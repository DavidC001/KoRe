#!/usr/bin/env python3
"""
Comprehensive evaluation script for KG-LFM model.

This script loads the best trained model and evaluates it on various metrics
relevant for KG-augmented generation including:
- Perplexity
- Top-k accuracy for KG predictions (Hit@k metrics)
- Graph embedding coherence metrics
- Knowledge utilization metrics
"""

import logging
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import gc
import torch, logging
from typing import List, Tuple

import torch
from tqdm.auto import tqdm

# Project imports
from KG_LM.configuration import load_yaml_config, ProjectConfig, IGNORE_INDEX, SPECIAL_KG_TOKEN
from KG_LM.model.KG_LM_arch import KG_LM, KG_LMConfig
from KG_LM.utils.Dataloader import create_dataloader

from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.utils import set_seed, broadcast_object_list

from copy import deepcopy

def align_logits_with_labels(
    logits: torch.Tensor, labels: torch.Tensor, num_quantizers: int, special_kg_token: int
) -> torch.Tensor:
    """
    Make logits length match labels length by *compressing* each KG block
    of size `num_quantizers` in logits down to a single row aligned with
    the single SPECIAL_KG_TOKEN in labels.

    Handles multiple KG tokens per sequence and ragged tails safely.
    """
    # Already aligned
    if logits.size(1) == labels.size(1):
        return logits

    B, Llab = labels.shape
    _, Llog, V = logits.shape
    out = logits.new_empty(B, Llab, V)

    for b in range(B):
        lab = labels[b]
        log = logits[b]
        write = 0  # index in labels / out
        read = 0   # index in logits

        while write < Llab and read < Llog:
            if lab[write].item() == special_kg_token:
                # compress the KG block [read : read + num_quantizers)
                block_end = min(read + num_quantizers, Llog)
                if block_end > read:
                    out[b, write] = log[read:block_end].mean(dim=0)
                    read = block_end
                else:
                    # degenerate case: no room left; repeat last valid row
                    last = max(read - 1, 0)
                    out[b, write] = log[last]
                write += 1
            else:
                out[b, write] = log[read]
                read += 1
                write += 1

        # If labels still have tail (e.g., logits shorter), pad with last valid row to keep shape
        if write < Llab:
            last_row = out[b, max(write - 1, 0)]
            out[b, write:] = last_row.expand(Llab - write, V)

    return out


def compute_hit_k(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    k_values: List[int],
    object_boundaries: List[Tuple[int, int]],
    num_quantizers: int,
    attention_mask: torch.Tensor,
    special_token: int,
    tokenizer,
    return_individual_ranks: bool = False
):
    logger = logging.getLogger()

    # 1) Align & shift for causal LM
    logits = align_logits_with_labels(logits, input_ids, num_quantizers, special_token)
    shift_logits = logits[..., :-1, :]           # (B, L-1, V)
    shift_labels = input_ids[..., 1:]            # (B, L-1)
    B, Lm1, V = shift_logits.shape
    device = shift_logits.device

    # 2) Build a boolean mask for object token positions (shifted by -1 due to causal LM)
    object_mask = torch.zeros((B, Lm1), dtype=torch.bool, device=device)
    for b, (start, end) in enumerate(object_boundaries):
        # Shift boundaries by -1 since we're working with shift_logits and shift_labels
        # which are offset by 1 position due to causal LM prediction
        if start is None or end is None:
            logger.warning(f"Invalid object boundaries for sample {b}: start={start}, end={end}")
            continue
            
        s = max(0, start - 1)  # shift start by -1, ensure >= 0
        e = min(Lm1, max(start, end - 1))  # shift end by -1, ensure <= Lm1, and >= start
        
        if s < e:
            object_mask[b, s:e] = True
        else:
            logger.warning(f"No valid object tokens found for sample {b} after shifting: start={start}->{s}, end={end}->{e}")

    has_object = object_mask.any(dim=1)          # (B,)
    total_objects = int(has_object.sum().item())

    # 3) (Same semantics as your running sum; caller can divide later if needed)
    tot_num_tokens = int(
        attention_mask.eq(1).sum().item()
        + input_ids.eq(special_token).sum().item() * num_quantizers
    )

    if total_objects == 0:
        return {k: 0 for k in k_values}, tot_num_tokens, total_objects

    with torch.no_grad():
        # 4) True logits and vectorized rank computation
        #     true_logits: (B, L-1)
        true_logits = shift_logits.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        #     ranks: 1 + count of logits strictly greater than the true logit
        ranks = 1 + (shift_logits > true_logits.unsqueeze(-1)).sum(dim=-1)  # (B, L-1), int64

        # 5) Sequence-level rank = max rank across object positions
        big = torch.iinfo(ranks.dtype).max
        seq_ranks = torch.full((B,), big, dtype=ranks.dtype, device=device)
        # Mask out non-object positions to 0 so max() only sees object tokens
        masked_max = ranks.masked_fill(~object_mask, 0).amax(dim=1)
        seq_ranks = torch.where(has_object, masked_max, seq_ranks)          # (B,)

    # 6) Hit@k counts (sequence is a hit if its max rank <= k)
    k_values_sorted = sorted(set(k_values))
    hit_k_batch_correct = {k: int((seq_ranks <= k).sum().item()) for k in k_values_sorted}

    # Optional: debug decode (kept cheap; only runs if DEBUG is enabled)
    if logger.isEnabledFor(logging.DEBUG):
        for b in range(B):
            if has_object[b]:
                obj_pos = object_mask[b]
                labels_b = shift_labels[b][obj_pos]
                logits_b = shift_logits[b][obj_pos]
                logger.debug(f"Decoded tokens[{b}]: {tokenizer.batch_decode(labels_b.tolist())}")
                logger.debug(f"Decoded argmax[{b}]: {tokenizer.batch_decode(logits_b.argmax(dim=-1).tolist())}")

    # Return individual ranks if requested (for WebQSP evaluation)
    if return_individual_ranks:
        individual_ranks = []
        for b in range(B):
            if has_object[b]:
                individual_ranks.append(seq_ranks[b].item())
            else:
                individual_ranks.append(float('inf'))  # No valid object found
        
        # DEBUG: Verify that manual aggregation of individual ranks matches hit_k_batch_correct
        manual_hits = {}
        for k in k_values:
            manual_count = sum(1 for rank in individual_ranks if rank != float('inf') and rank <= k)
            manual_hits[k] = manual_count
            if manual_hits[k] != hit_k_batch_correct[k]:
                logger.warning(f"MISMATCH for k={k}: manual={manual_hits[k]}, original={hit_k_batch_correct[k]}")
        
        return individual_ranks, tot_num_tokens, total_objects
    
    return hit_k_batch_correct, tot_num_tokens, total_objects


class BaseLLM:
    """Base class for LLM evaluators."""
    def __init__(self,model):
        self.model = model
        
    def __call__(self, **kwds):
        """Return the model output without LoRA layers."""
        with self.model.llm.disable_adapter():
            results = self.model(**kwds)

        return results

    def generate(self, **kwds):
        """Generate text using the model."""
        with self.model.llm.disable_adapter():
            generated = self.model.generate(**kwds)
        return generated


class KGLFMEvaluator:
    """Comprehensive evaluator for KG-LFM model."""
    
    def __init__(
        self,
        config_path: str,
        batch_size: int = 8,
        max_samples: Optional[int] = None,
        no_baseline: bool = False,
        no_text: bool = False,
        only_baselines: bool = False,
        split: str = "test",
        corrupt: bool = False,
    ):
        self.config_path = config_path
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.no_baseline = no_baseline
        self.no_text = no_text
        self.only_baselines = only_baselines

        assert not (self.no_baseline and self.only_baselines), "Conflicting options: --no_baseline and --only_baselines"

        # Initialize accelerator
        self.accelerator = Accelerator()
        
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config : ProjectConfig = load_yaml_config(config_path)
        self.model_path = Path(self.config.train_conf.start_from_checkpoint)
        
        self.config.train_conf.dataloader.batch_size = batch_size
        self.config.dataset.corrupt = corrupt
        
        set_seed(self.config.seed)
        
        # Initialize model
        self.model = None
        self.tokenizer = None
        
        self.split = split
        
        # Initialize metrics storage
        self.results = defaultdict(dict)
        
        self.evaluations = {
            # 'perplexity': self.compute_perplexity,
            'hit_at_k': self.compute_hit_k_metrics,
        }
        
    def remove_kg_stuff(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Remove KG-related tokens from input_ids and attention_mask."""
        
        batch_sentences = []
        conversations = []
        for sample_idx in range(len(batch["input_ids"])):
            conversation = batch["conversations"][sample_idx]
            
            # remove the one with the special token
            kg_string = SPECIAL_KG_TOKEN
            conversation = [turn for turn in conversation if kg_string not in turn['content']]

            conversations.append(conversation)

            # If tokenizer has a apply_chat_template method, use it
            try:
                sentence = self.clean_tokenizer.apply_chat_template(
                    conversation=conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
            except ValueError:
                sentence = ""
                for turn in conversation:
                    sentence += turn['role'] + ": " + turn['content'] + "\n"
                sentence = sentence.strip()

            obj_start, obj_end = batch["objects"][sample_idx]["boundaries"]
            obj_str = batch["sentences"][sample_idx][obj_start:obj_end]

            new_obj_start = sentence.rfind(obj_str)
            new_obj_end = new_obj_start + len(obj_str)
            batch["objects"][sample_idx]["boundaries"] = (new_obj_start, new_obj_end)
            batch_sentences.append(sentence)
        
        tokenized = self.clean_tokenizer(
            batch_sentences,
            padding=True,
            return_tensors='pt',
        )
        batch["labels"] = torch.full(tokenized['input_ids'].shape, IGNORE_INDEX, dtype=tokenized['input_ids'].dtype)

        for sample_idx in range(len(batch["sentences"])):
            obj_start, obj_end = batch["objects"][sample_idx]["boundaries"]
            tok_start = tokenized.char_to_token(sample_idx, obj_start)
            tok_end = tokenized.char_to_token(sample_idx, obj_end)
            batch["objects"][sample_idx]["token_boundaries"] = (tok_start, tok_end)

            batch["labels"][sample_idx][tok_start:tok_end] = tokenized['input_ids'][sample_idx][tok_start:tok_end]

        out = {
            "question_ids": batch["question_ids"],
            "conversations": conversations,
            "sentences": batch_sentences,
            "objects": batch["objects"],
            "input_ids": tokenized['input_ids'],
            "attention_mask": tokenized['attention_mask'],
            "graphs": batch["graphs"]
        }
        
        return out

    def kg_textualization(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert KG graphs to textual format for model input."""
        # Convert graphs to textual representation
        
        # deep copy to avoid modifying original batch
        batch = deepcopy(batch)
        
        for sample_idx in range(len(batch["sentences"])):
            graph = batch["graphs"][sample_idx]
            
            graph_text = " Information from the knowledge graph: "
            
            central_node_label = graph["central_node_label"]
            neighbors = graph["neighbour_node_labels"]
            edges = graph["edge_labels"]
            
            for neighbor, edge in zip(neighbors, edges):
                # Create a textual representation of the graph
                graph_text += f"{central_node_label} {edge} {neighbor}. "

            graph_text += "\n"

            # in the turn with SPECIAL_KG_TOKEN substitute it with <SPECIAL_KG_TOKEN>
            special_tok = SPECIAL_KG_TOKEN
            num_turns = len(batch["conversations"][sample_idx])
            batch["conversations"][sample_idx] = [{
                "role": batch["conversations"][sample_idx][i]["role"],
                "content": batch["conversations"][sample_idx][i]["content"].replace(special_tok, graph_text)
            } for i in range(num_turns)]

        return self.remove_kg_stuff(batch)

    def load_model(self):
        """Load the best trained model."""
        if self.accelerator.is_main_process:
            self.logger.info(f"Loading model from {self.model_path}")
        
        self.tests = {}
        
        try:
            self.model_config : KG_LMConfig = KG_LMConfig.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path / "llm")
            
            self.special_kg_token_id = self.tokenizer.convert_tokens_to_ids(SPECIAL_KG_TOKEN)
            
            if not self.only_baselines:
                self.model = KG_LM.from_pretrained(self.model_path)
                self.model.eval()
                
                self.tests.update({
                    "KG_LM": (lambda x: x, self.model),  # No preprocessing needed for KG_LM
                })

            # if the config requires to tune the model also load clean model
            if  not self.no_baseline:
                if self.only_baselines or self.model_config.tune_language_model:
                    self.clean_model = AutoModelForCausalLM.from_pretrained(
                        self.model_config.llm_model_name,
                    )
                    self.clean_tokenizer = AutoTokenizer.from_pretrained(
                        self.model_config.llm_model_name,
                    )
                    self.clean_model.eval()

                elif self.model_config.use_lora:
                    self.clean_model = BaseLLM(self.model)
                    self.clean_tokenizer = self.tokenizer
                else:
                    self.clean_model = self.model
                    self.clean_tokenizer = self.tokenizer
                
                if self.accelerator.is_main_process:
                    self.logger.info("Model loaded successfully")
                    self.logger.info("Clean model loaded successfully")

                self.tests.update({
                    "original_LLM": (self.remove_kg_stuff, self.clean_model),
                })

                if not self.no_text:
                    self.tests.update({
                        "textualization": (self.kg_textualization, self.clean_model),
                    })
        except Exception as e:
            if self.accelerator.is_main_process:
                self.logger.error(f"Error loading model: {e}")
            raise
    
    def setup_data(self):
        """Setup data loaders for evaluation."""
        if self.accelerator.is_main_process:
            self.logger.info(f"Setting up {self.split} data loader")

        self.dataloader = create_dataloader(
            self.config.dataset,
            self.config.train_conf.dataloader,
            tokenizer=self.tokenizer,
            split=self.split
        )

        # disable shuffling for evaluation
        self.dataloader.shuffle = False

        if self.accelerator.is_main_process:
            self.logger.info(f"Data loader setup complete. {len(self.dataloader)} batches available.")
    
    def prepare_accelerator(self):
        """Prepare the accelerator for distributed training."""
        if self.accelerator.is_main_process:
            self.logger.info("Preparing accelerator...")
        
        # Prepare model and dataloader
        if self.only_baselines:
            self.clean_model, self.dataloader = self.accelerator.prepare(
                self.clean_model, self.dataloader
            )
        if self.model_config.tune_language_model:
            self.model, self.clean_model, self.dataloader = self.accelerator.prepare(
                self.model, self.clean_model, self.dataloader
            )
        else:
            self.model, self.dataloader = self.accelerator.prepare(
                self.model, self.dataloader
            )
            self.clean_model = self.model  # Use the same model if not tuning

    def compute_perplexity(self) -> Dict[str, float]:
        """Compute perplexity for all models on the evaluation dataset.
        
        Perplexity is computed as exp(average negative log-likelihood) over all
        valid token positions (non-ignored tokens).
        """
        if self.accelerator.is_main_process:
            self.logger.info("Computing perplexity metrics...")
        
        results = {}
        
        for name, (preprocess_func, model) in self.tests.items():
            if self.accelerator.is_main_process:
                self.logger.info(f"Evaluating {name} model for perplexity...")
            
            total_loss = 0.0
            total_tokens = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(self.dataloader, desc=f"Computing perplexity for {name}", disable=not self.accelerator.is_main_process)):
                    if self.max_samples and (batch_idx * self.batch_size * self.accelerator.num_processes) >= self.max_samples:
                        break

                    batch = preprocess_func(batch)
                    
                    batch['input_ids'] = batch['input_ids'].to(self.accelerator.device)
                    batch['attention_mask'] = batch['attention_mask'].to(self.accelerator.device)
                    
                    # Use the proper labels from the dataset (only assistant response tokens)
                    labels = batch.get('labels', batch['input_ids'].clone())
                    labels = labels.to(self.accelerator.device)
                    
                    model_input = {
                        'input_ids': batch['input_ids'],
                        'attention_mask': batch['attention_mask'],
                        'labels': labels
                    }
                    if batch.get('graphs'): 
                        model_input['graphs'] = batch['graphs']

                    outputs = model(**model_input)
                    
                    # Get the loss and number of tokens
                    loss = outputs.loss
                    
                    # Count valid tokens (non-ignored tokens in labels)
                    # Only count tokens that are not IGNORE_INDEX - these are the assistant response tokens
                    valid_tokens = (labels != IGNORE_INDEX).sum().item()
                    
                    total_loss += loss.item() * valid_tokens
                    total_tokens += valid_tokens

            # Gather losses and token counts from all processes
            total_loss_tensor = torch.tensor(total_loss, device=self.accelerator.device)
            total_tokens_tensor = torch.tensor(total_tokens, device=self.accelerator.device)
            
            total_loss_gathered = self.accelerator.gather(total_loss_tensor).sum().item()
            total_tokens_gathered = self.accelerator.gather(total_tokens_tensor).sum().item()

            # Synchronize across processes
            self.accelerator.wait_for_everyone()
            
            # Compute perplexity on main process
            if self.accelerator.is_main_process:
                if total_tokens_gathered > 0:
                    avg_loss = total_loss_gathered / total_tokens_gathered
                    perplexity = math.exp(avg_loss)
                    
                    results[name] = {
                        'perplexity': perplexity,
                        'average_loss': avg_loss,
                        'total_tokens': total_tokens_gathered
                    }
                    
                    self.logger.info(f"Perplexity for {name}: {perplexity:.4f} (avg_loss: {avg_loss:.4f}, tokens: {total_tokens_gathered})")
                else:
                    self.logger.warning(f"No valid tokens found for perplexity computation for {name}")
                    results[name] = {
                        'perplexity': float('inf'),
                        'average_loss': float('inf'),
                        'total_tokens': 0
                    }
            else:
                # Non-main processes set dummy values
                results[name] = {
                    'perplexity': 0.0,
                    'average_loss': 0.0,
                    'total_tokens': 0
                }
                    
        # Broadcast results to all processes
        if self.accelerator.is_main_process:
            results_to_broadcast = results
        else:
            results_to_broadcast = None
        
        results = broadcast_object_list([results_to_broadcast])[0]
        
        return results

    def compute_hit_k_metrics(self, k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """Compute Hit@k metrics for object label prediction.
        
        To quantify recall, we adopt the widely used Hit@k metric. For an object label 
        split into T tokens, we record the rank (r_t) of each token t in the model's 
        output logits. The sequence rank is taken as r = max{r_1,...,r_T}, and it counts 
        as a "hit" if r ≤ k (i.e., all tokens appear in the top-k predictions at their 
        respective timesteps). This approach is robust to multi-token entities, a common 
        challenge in IR tasks involving named entities ("New York Times" vs. "NYT").
        """
        if self.accelerator.is_main_process:
            self.logger.info(f"Computing Hit@k metrics for k={k_values}...")
        
        results = {}
        
        for name, (preprocess_func, model) in self.tests.items():
            if self.accelerator.is_main_process:
                self.logger.info(f"Evaluating {name} model for Hit@k metrics...")
            
            hit_k_correct = {k: 0 for k in k_values}
            total_objects = 0
            total_samples = 0
            average_num_tokens = 0
            
            # For WebQSP: collect all predictions per question across all batches  
            is_webqsp = self.config.dataset.name == "web-qsp"
            
            # TEMPORARY: Force disable WebQSP mode to test individual ranks method
            if self.accelerator.is_main_process and is_webqsp:
                self.logger.info("Using WebQSP evaluation mode: computing both per-sample and per-question metrics")
            
            # Collect all question predictions across batches for WebQSP
            all_question_ranks = defaultdict(list) if is_webqsp else None

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(self.dataloader, desc=f"Computing Hit@k metrics for {name}", disable=not self.accelerator.is_main_process)):
                    if self.max_samples and (batch_idx * self.batch_size * self.accelerator.num_processes) >= self.max_samples:
                        break

                    batch = preprocess_func(batch)
                    # No need to move to device manually - accelerator handles this

                    object_boundaries = [obj["token_boundaries"] for obj in batch['objects']]

                    batch['input_ids'] = batch['input_ids'].to(self.accelerator.device)
                    batch['attention_mask'] = batch['attention_mask'].to(self.accelerator.device)
                    batch['labels'] = batch['input_ids'].to(self.accelerator.device)
                    
                    model_input = {
                        'input_ids': batch['input_ids'],
                        'attention_mask': batch['attention_mask'],
                    }
                    if batch['graphs']: model_input['graphs'] = batch['graphs']

                    outputs = model(**model_input)
                    # Get logits and labels
                    logits = outputs.logits
                    input_ids = batch['input_ids']

                    
                    if not is_webqsp:
                        # original compute_hit_k for non-WebQSP (should be identical to old results)
                        hit_k_correct_batch, batch_avg_num_tokens, new_objects = compute_hit_k(
                            logits, input_ids, k_values,
                            object_boundaries, self.model_config.num_quantizers + (2 if hasattr(self.model_config, 'bounding_tokens') and self.model_config.bounding_tokens else 0),
                            batch['attention_mask'],
                            special_token=self.special_kg_token_id, tokenizer=self.tokenizer,
                            return_individual_ranks=False
                        )
                        
                        for k in k_values:
                            hit_k_correct[k] += hit_k_correct_batch[k]
                            
                        total_objects += new_objects
                    else:
                        # new individual ranks method
                        individual_ranks, batch_avg_num_tokens, new_objects = compute_hit_k(
                            logits, input_ids, k_values,
                            object_boundaries, self.model_config.num_quantizers + (2 if hasattr(self.model_config, 'bounding_tokens') and self.model_config.bounding_tokens else 0),
                            batch['attention_mask'],
                            special_token=self.special_kg_token_id, tokenizer=self.tokenizer,
                            return_individual_ranks=True
                        )
                        
                        question_ids = batch.get('question_ids', None)
                        
                        if question_ids is None:
                            # Create unique question IDs across processes to avoid conflicts
                            batch_size = len(batch['input_ids'])
                            # Use process_index and batch_idx to create unique IDs
                            base_id = batch_idx * self.accelerator.num_processes * batch_size + self.accelerator.process_index * batch_size
                            question_ids = list(range(base_id, base_id + batch_size))
                            
                            # Log warning on first batch if no proper question IDs
                            if batch_idx == 0 and self.accelerator.is_main_process:
                                self.logger.warning(f"No 'question_ids' found in batch data. Using generated IDs starting from {base_id}. This may not represent actual WebQSP questions.")
                        
                        # For WebQSP, collect ranks per question for later aggregation
                        for qid, rank in zip(question_ids, individual_ranks):
                            if rank != float('inf'):  # Only collect valid ranks
                                all_question_ranks[qid].append(rank)
                        
                        # Debug: Log question IDs to understand what we're seeing
                        if batch_idx < 3:
                            # Log raw question_ids from batch
                            self.logger.info(f"Process {self.accelerator.process_index}, Batch {batch_idx}: Raw question_ids from batch: {question_ids[:3] if question_ids else 'None'}")
                            # Log ranks
                            valid_qids_and_ranks = [(qid, rank) for qid, rank in zip(question_ids, individual_ranks) if rank != float('inf')]
                            if valid_qids_and_ranks:
                                sample_data = valid_qids_and_ranks[:3]
                                self.logger.info(f"Process {self.accelerator.process_index}, Batch {batch_idx}: Valid (qid, rank) pairs: {sample_data}")
                                
                    average_num_tokens += batch_avg_num_tokens
                    total_samples += new_objects

            # For WebQSP, we need to defer the per-question aggregation until after gathering all data
            if is_webqsp and all_question_ranks is not None:
                # Debug: Log local question ranks before gathering
                local_question_count = len(all_question_ranks)
                if local_question_count > 0:
                    sample_qids = list(all_question_ranks.keys())[:5]
                    self.logger.info(f"Process {self.accelerator.process_index}: Found {local_question_count} local questions, sample IDs: {sample_qids}")
                else:
                    self.logger.info(f"Process {self.accelerator.process_index}: No local questions found")
                
                # Manually gather question ranks from all processes using individual broadcasts
                gathered_question_ranks = []
                
                for process_idx in range(self.accelerator.num_processes):
                    # Prepare data to broadcast (only the current process has real data)
                    if process_idx == self.accelerator.process_index:
                        process_data = dict(all_question_ranks)
                    else:
                        process_data = {}
                    
                    # Broadcast this process's data to all processes
                    data_to_broadcast = [process_data]
                    broadcast_object_list(data_to_broadcast, from_process=process_idx)
                    
                    # All processes now have this process's data
                    gathered_question_ranks.append(data_to_broadcast[0])
                
                if self.accelerator.is_main_process:
                    # Debug: Log what each process contributed
                    for idx, process_data in enumerate(gathered_question_ranks):
                        self.logger.info(f"Process {idx} contributed {len(process_data)} questions: {list(process_data.keys())[:10]}...")
                    
                    # Merge question ranks from all processes
                    merged_question_ranks = defaultdict(list)
                    for process_data in gathered_question_ranks:
                        for qid, ranks in process_data.items():
                            merged_question_ranks[qid].extend(ranks)
                    
                    self.logger.info(f"Total unique questions after merging: {len(merged_question_ranks)}")
                    
                    if merged_question_ranks:
                        self.logger.info(f"WebQSP proper question IDs found - using per-question evaluation")
                    else:
                        self.logger.warning("No WebQSP question IDs found across all processes")
                    
                    # Compute per-sample metrics for comparison
                    sample_hits = {k: 0 for k in k_values}
                    total_samples = 0
                    for qid, ranks in merged_question_ranks.items():
                        for rank in ranks:
                            total_samples += 1
                            for k in k_values:
                                if rank <= k:
                                    sample_hits[k] += 1
                    
                    # Compute per-question metrics (WebQSP standard)
                    question_hits = {k: 0 for k in k_values}
                    valid_questions = 0
                    for qid, ranks in merged_question_ranks.items():
                        if ranks:  # Only process questions with valid ranks
                            best_rank = min(ranks)  # Get the best (minimum) rank for this question
                            valid_questions += 1
                            for k in k_values:
                                if best_rank <= k:
                                    question_hits[k] += 1
                    
                    self.logger.info(f"WebQSP evaluation: {valid_questions} questions, {total_samples} total samples")
                    for k in k_values:
                        sample_rate = sample_hits[k] / total_samples if total_samples > 0 else 0
                        question_rate = question_hits[k] / valid_questions if valid_questions > 0 else 0
                        self.logger.info(f"Hit@{k}: per-sample={sample_rate:.4f}, per-question={question_rate:.4f}")
                    
                    # Update hit_k_correct and total_objects with WebQSP per-question metrics
                    hit_k_correct = question_hits
                    total_objects = valid_questions
                    

            # Gather hit counts and total objects from all processes (after WebQSP processing)
            if not is_webqsp:
                # For non-WebQSP datasets, gather results normally
                hit_k_tensors = {}
                for k in k_values:
                    hit_k_tensor = torch.tensor(hit_k_correct[k], device=self.accelerator.device)
                    hit_k_tensors[k] = self.accelerator.gather(hit_k_tensor).sum().item()
                
                total_objects_tensor = torch.tensor(total_objects, device=self.accelerator.device)
                total_objects_gathered = self.accelerator.gather(total_objects_tensor).sum().item()
            else:
                # For WebQSP, results are already computed on main process, just broadcast them
                if self.accelerator.is_main_process:
                    hit_k_tensors = {k: hit_k_correct[k] for k in k_values}
                    total_objects_gathered = total_objects
                else:
                    hit_k_tensors = {k: 0 for k in k_values}
                    total_objects_gathered = 0
                
                # Broadcast the final results to all processes for consistent state
                results_to_broadcast = [hit_k_tensors, total_objects_gathered]
                broadcast_object_list(results_to_broadcast, from_process=0)
                hit_k_tensors, total_objects_gathered = results_to_broadcast
            
            total_samples_tensor = torch.tensor(total_samples, device=self.accelerator.device)
            total_samples_gathered = self.accelerator.gather(total_samples_tensor).sum().item()

            # Gather average_num_tokens from all processes
            average_num_tokens_gathered = self.accelerator.gather(torch.tensor(average_num_tokens, device=self.accelerator.device)).sum().item()

            # Synchronize across processes
            self.accelerator.wait_for_everyone()
            
            # Compute Hit@k metrics on main process
            metrics = {}
            if self.accelerator.is_main_process:
                if total_objects_gathered > 0:
                    average_num_tokens_gathered /= total_samples_gathered
                    metrics['average_num_tokens'] = average_num_tokens_gathered
                    
                    for k in k_values:
                        metrics[f'hit_at_{k}'] = hit_k_tensors[k] / total_objects_gathered
                    
                    # Only show summary logging for non-WebQSP (WebQSP already logged detailed results)
                    if not is_webqsp:
                        self.logger.info(f"Hit@k computed on {total_objects_gathered} objects ({total_samples_gathered} total samples) with average {average_num_tokens_gathered:.2f} tokens per sample.")
                        for k in k_values:
                            self.logger.info(f"Hit@{k}: {metrics[f'hit_at_{k}']:.4f}")
                    else:
                        self.logger.info(f"Hit@k computed on {total_objects_gathered} questions ({total_samples_gathered} total samples) with average {average_num_tokens_gathered:.2f} tokens per sample.")
                        for k in k_values:
                            self.logger.info(f"Hit@{k}: {metrics[f'hit_at_{k}']:.4f}")
                else:
                    self.logger.warning("No valid objects found for Hit@k computation")
                    for k in k_values:
                        metrics[f'hit_at_{k}'] = 0.0
            else:
                # Non-main processes need empty metrics
                metrics = {}
   
            # Broadcast metrics to all processes so they all have the same results
            metrics = broadcast_object_list([metrics])[0]
            
            # Store results for this preprocessing method
            results[name] = metrics

        # Broadcast results to all processes
        results = broadcast_object_list([results])[0]
        
        return results

    def evaluate(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Run all evaluation metrics and return comprehensive results."""
        if self.accelerator.is_main_process:
            self.logger.info("Starting comprehensive evaluation...")
        
        # Load model and setup data
        self.load_model()
        self.setup_data()
        self.prepare_accelerator()
        
        # Run all evaluations
        evaluations = self.evaluations
        
        # Run evaluations
        for eval_name, eval_func in evaluations.items():
            if self.accelerator.is_main_process:
                self.logger.info(f"Running {eval_name}...")
                
            self.results[eval_name] = eval_func()
            
            # Clear GPU memory after each evaluation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        # Add metadata (only on main process)
        if self.accelerator.is_main_process:
            self.results['metadata'] = {
                'model_path': str(self.model_path),
                'config_path': self.config_path,
                'batch_size': self.batch_size,
                'max_samples': self.max_samples,
                'num_processes': self.accelerator.num_processes,
                'process_index': self.accelerator.process_index,
                'split': self.split,
                'corrupt': self.config.dataset.corrupt,
            }
        
        # Save results if output file specified (only on main process)
        if output_file and self.accelerator.is_main_process:
            self.save_results(output_file)
        
        return dict(self.results)
    
    def save_results(self, output_file: str):
        """Save evaluation results to a JSON file."""
        if not self.accelerator.is_main_process:
            return
            
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def print_summary(self):
        """Print a summary of evaluation results."""
        if not self.accelerator.is_main_process:
            return
            
        print("\n" + "="*80)
        print("KG-LFM EVALUATION SUMMARY")
        print("="*80)
        
        for category, metrics in self.results.items():
            if category == 'metadata':
                continue
            
            print(f"\n{category.upper().replace('_', ' ')}:")
            print("-" * 40)
            
            if isinstance(metrics, dict) and 'error' not in metrics:
                # Handle nested results (e.g., different models for perplexity)
                if any(isinstance(v, dict) for v in metrics.values()):
                    for model_name, model_metrics in metrics.items():
                        if isinstance(model_metrics, dict):
                            print(f"  {model_name}:")
                            for metric_name, value in model_metrics.items():
                                if isinstance(value, float):
                                    if metric_name == 'perplexity':
                                        print(f"    {metric_name}: {value:.4f}")
                                    else:
                                        print(f"    {metric_name}: {value:.6f}")
                                else:
                                    print(f"    {metric_name}: {value}")
                        else:
                            if isinstance(model_metrics, float):
                                print(f"  {model_name}: {model_metrics:.4f}")
                            else:
                                print(f"  {model_name}: {model_metrics}")
                else:
                    for metric_name, value in metrics.items():
                        if isinstance(value, float):
                            print(f"  {metric_name}: {value:.4f}")
                        else:
                            print(f"  {metric_name}: {value}")
            elif 'error' in metrics:
                print(f"  Error: {metrics['error']}")
        
        print("\n" + "="*80)