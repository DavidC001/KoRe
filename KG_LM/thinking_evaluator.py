import logging

import nltk
from KG_LM.evaluator import KGLFMEvaluator, compute_hit_k
import torch
from tqdm.auto import tqdm
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from copy import deepcopy

from fuzzywuzzy import fuzz

from accelerate.utils import broadcast_object_list

class KGLFMThinkingEvaluator(KGLFMEvaluator):
    
    SYSTEM_PROMPT = {
        "role": "system",
        "content": "You are a helpful assistant designed to answer user queries. First think and then answer with \"the answer is: entity\""
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized KGLFMThinkingEvaluator.")

        # Generation parameters
        self.max_new_tokens = 256
        self.temperature = 0.6
        self.top_p = 0.95
        self.top_k = 20
        self.repetition_penalty = 1.2
        
        # Think tokens for reasoning
        self.think_start_token = "<think>"
        self.think_end_token = "</think>"
        
        # Available evaluation metrics
        self.evaluations = {
            'hit_at_k': self.compute_hit_k_metrics,
            "fuzzy_entity_recognition": self.compute_fuzzy_metrics
        }

    # ---------- Prompt construction ----------
    def _format_chat_messages(self, messages: List[Dict[str, str]], 
                             add_generation_prompt: bool = True) -> str:
        """
        Format chat messages using the tokenizer's chat template.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            add_generation_prompt: Whether to add the generation prompt for inference
            
        Returns:
            Formatted prompt string
        """
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            )
        except Exception as e:
            self.logger.debug(f"Chat template failed, using fallback: {e}")
            return self._fallback_chat_format(messages)

    def _fallback_chat_format(self, messages: List[Dict[str, str]]) -> str:
        """Fallback formatting when tokenizer's chat template fails."""
        parts = []
        
        # Handle system message
        if messages[0]["role"] != "system":
            parts.extend([f"{msg['role']}: {msg['content']}" for msg in messages])
        else:
            system_content = messages[0]["content"]
            parts.append(f"system: {system_content}")
            parts.extend([f"{msg['role']}: {msg['content']}" for msg in messages[1:]])
        
        parts.append("assistant:")
        return "\n".join(parts)


    # ---------- Helper methods for distributed evaluation ----------
    def _gather_metrics_across_processes(self, local_value: int) -> int:
        """Gather and sum a metric value across all processes."""
        tensor = torch.tensor(local_value, device=self.accelerator.device)
        gathered = self.accelerator.gather(tensor)
        return gathered.sum().item()
    
    def _gather_webqsp_question_ranks(self, local_question_ranks: Dict[Any, List[float]]) -> Dict[Any, List[float]]:
        """
        Gather question ranks from all processes for WebQSP evaluation.
        
        Args:
            local_question_ranks: Dictionary mapping question IDs to rank lists (local to this process)
            
        Returns:
            Merged dictionary with ranks from all processes (only on main process)
        """
        merged_ranks = defaultdict(list)
        
        # Broadcast from each process
        for process_idx in range(self.accelerator.num_processes):
            if process_idx == self.accelerator.process_index:
                process_data = dict(local_question_ranks)
            else:
                process_data = {}
            
            data_to_broadcast = [process_data]
            broadcast_object_list(data_to_broadcast, from_process=process_idx)
            
            # Merge on all processes
            for question_id, ranks in data_to_broadcast[0].items():
                merged_ranks[question_id].extend(ranks)
        
        return merged_ranks
    
    def _compute_webqsp_metrics(self, question_ranks: Dict[Any, List[float]], k_values: List[int]) -> Dict[str, Any]:
        """
        Compute Hit@K metrics for WebQSP dataset (per-question aggregation).
        
        For each question, we take the best (minimum) rank across all linked entities.
        
        Args:
            question_ranks: Dictionary mapping question IDs to list of ranks
            k_values: List of k values for Hit@K
            
        Returns:
            Dictionary of metrics
        """
        if not question_ranks:
            return {f'hit@{k}': 0.0 for k in k_values}
        
        # Take best rank for each question
        question_best_ranks = {
            qid: min(ranks)
            for qid, ranks in question_ranks.items()
        }
        
        # Compute Hit@K
        metrics = {}
        num_questions = len(question_best_ranks)
        
        for k in k_values:
            hits = sum(1 for rank in question_best_ranks.values() if rank <= k)
            metrics[f'hit@{k}'] = hits / num_questions if num_questions > 0 else 0.0
        
        metrics['questions'] = num_questions
        return metrics
    
    
    
    # ---------- Finding object label token boundaries ----------
    def _find_object_boundaries(self, prompts: List[str], 
                               objects: List[Dict[str, Any]],
                               tokenizer_encoding) -> List[Tuple[Optional[int], Optional[int]]]:
        """
        Find token boundaries for object labels in prompts.
        
        Args:
            prompts: List of formatted prompt strings
            objects: List of object dictionaries with 'label' field
            tokenizer_encoding: Tokenizer output with char_to_token method
            
        Returns:
            List of (start_token, end_token) tuples for each sample
        """
        boundaries = []
        
        for i, prompt in enumerate(prompts):
            obj_text = objects[i]["label"]
            
            # Find the last occurrence of the object text (in gold answer)
            obj_pos = prompt.rfind(obj_text)
            
            if obj_pos >= 0:
                char_start = obj_pos
                char_end = char_start + len(obj_text)
                
                token_start = tokenizer_encoding.char_to_token(i, char_start)
                token_end = tokenizer_encoding.char_to_token(i, char_end)
                
                boundaries.append((token_start, token_end))
            else:
                self.logger.warning(f"Could not find object text '{obj_text}' in prompt for sample {i}")
                boundaries.append((None, None))
        
        return boundaries


    # ==========================
    # Thinking-based generation and logits computation
    # ==========================
    
    
    # -------------------------
    # KG-LM with thinking
    # -------------------------
    def _generate_with_thinking_batched(self, 
                                        batch: Dict[str, Any], 
                                        stop_strings: Optional[List[str]] = None
                                    ) -> List[str]:
        """
        Generate answers using thinking .
        Args:
            batch: Batch dictionary with input data
            stop_strings: Optional list of stop strings for generation

        Returns:
            List of final generated answer texts (one per sample)
        """
        # Initialize conversation state
        conversations = [
            [
                self.SYSTEM_PROMPT,
                batch["conversations"][i][0],  # User question
                batch["conversations"][i][1],  # Tool answer
            ] for i in range(len(batch["conversations"]))
        ]
        
        prompts = [
            self._format_chat_messages(
                conversations[i], add_generation_prompt=True
            ) for i in range(len(conversations))
        ]
        
        # Tokenize prompts
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side="left",
        ).to(self.accelerator.device)
        
        # Generate responses with </think> as stop token
        generated = self.model.generate(
            **encoded,
            graphs=batch["graphs"],
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            top_k=self.top_k,
            stop_strings=stop_strings,
            tokenizer=self.tokenizer,
        )
        
        # Decode generated tokens
        decoded_texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        return decoded_texts

    def _compute_thinking_logits_batched(self, batch: Dict[str, Any]) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, list[tuple[int, int]] | None]:
        """
        Compute logits for thinking model by generating then evaluating on gold answers.
        
        This two-step process:
        1. Generates answers using thinking
        2. Appends GOLD answer to conversation and computes logits
        
        Args:
            batch: Input batch dictionary
            
        Returns:
            Tuple of (logits, input_ids, attention_mask, object_boundaries)
        """
        batch_size = len(batch["sentences"])
        
        # Step 1: Generate answers using the full thinking process
        generated_texts = self._generate_with_thinking_batched(batch, stop_strings=[self.think_end_token])
        if not generated_texts or len(generated_texts) != batch_size:
            self.logger.warning("Failed to generate thinking answers for batch")
            return None, None, None, None
        
        # Step 2: Build conversations with generated answers followed by GOLD answers
        conversations = []
        entity_ids_per_sample = []
        
        for i in range(batch_size):
            conversation = [
                self.SYSTEM_PROMPT,
                batch["conversations"][i][0],  # User question
                batch["conversations"][i][1],  # Tool answer
                {"role": "assistant", "content": generated_texts[i] + "\n</think>" + batch["conversations"][i][2]["content"]}
            ]
            conversations.append(conversation)
            entity_ids_per_sample.append([batch["subjects"][i]["id"]])
        
        # breakpoint()
        
        # Format prompts with gold answers
        prompts = [
            self._format_chat_messages(conv, add_generation_prompt=False)
            for conv in conversations
        ]
        
        # debug print first message
        self.logger.debug(f"First prompt with gold answer:\n{prompts[0]}")
        
        # Tokenize with offset mapping for boundary detection
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side="left",
            return_offsets_mapping=True,
        ).to(self.accelerator.device)
        
        # Compute logits
        outputs = self.model(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
            graphs=batch["graphs"],
        )
        
        # Find token boundaries for gold object labels
        object_boundaries = self._find_object_boundaries(
            prompts, batch["objects"], encoded
        )
        
        return outputs.logits, encoded['input_ids'], encoded['attention_mask'], object_boundaries


    # -------------------------
    # Baseline models with thinking
    # -------------------------
    def _generate_baseline_answers_batched(
        self, 
        batch: Dict[str, Any], 
        preprocess_func, model,
        stop_strings: List[str] = None
        ) -> List[str]:
        """
        Generate answers for baseline models with thinking tokens.
        
        Applies appropriate preprocessing and generates responses with </think> as stop token.
        
        Args:
            batch: Input batch dictionary
            preprocess_func: Preprocessing function for this baseline
            model: The baseline model
            stop_strings: Optional list of stop strings for generation
        Returns:
            Tuple of (full_thinking_texts, extracted_answers)
            - full_thinking_texts: Full generated text including thinking (for logits computation)
            - extracted_answers: Just the answer part after </think> (for fuzzy evaluation)
        """
        # Preprocess and prepare inputs
        batch_processed = preprocess_func(deepcopy(batch))
        batch_processed['input_ids'] = batch_processed['input_ids'].to(self.accelerator.device)
        batch_processed['attention_mask'] = batch_processed['attention_mask'].to(self.accelerator.device)
        
        model_input = {
            'input_ids': batch_processed['input_ids'],
            'attention_mask': batch_processed['attention_mask'],
        }
        
        # Add graphs if available (for textualization baseline)
        if batch_processed.get('graphs'):
            model_input['graphs'] = batch_processed['graphs']
        
        # Select tokenizer
        tokenizer = self.clean_tokenizer if hasattr(self, 'clean_tokenizer') else self.tokenizer
        
        # Generate with </think> as stop token
        with torch.no_grad():
            try:
                generated_tokens = model.generate(
                    **model_input,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
                    repetition_penalty=self.repetition_penalty,
                    top_k=self.top_k,
                    stop_strings=stop_strings,
                    tokenizer=tokenizer,
                )
                
                # Decode full generated text (including thinking)
                full_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                
            except Exception as e:
                self.logger.error(f"Error in baseline generation: {e}")
                # Fallback to empty strings
                batch_size = batch_processed['input_ids'].shape[0]
                full_texts = [""] * batch_size
        
        return full_texts

    def _compute_baseline_logits_batched(self, batch: Dict[str, Any], preprocess_func, model) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, list[tuple[int, int]] | None]:
        """
        Compute logits for baseline model by generating then evaluating on gold answers.
        
        This two-step process:
        1. Generates answers for baseline model (for fuzzy evaluation)
        2. Appends GOLD answer to the generated conversation and computes logits (for Hit@K evaluation)
        
        Args:
            batch: Input batch dictionary
            preprocess_func: Preprocessing function for this baseline
            model: The baseline model
            
        Returns:
            Tuple of (logits, input_ids, attention_mask, object_boundaries)
        """
        # Preprocess batch for baseline model
        batch_processed = preprocess_func(deepcopy(batch))
        batch_size = len(batch_processed["sentences"])
        
        # Step 1: Generate answers (get both full thinking text and extracted answers)
        full_thinking_texts = self._generate_baseline_answers_batched(
            batch, preprocess_func, model, stop_strings=[self.think_end_token]
        )
        if not full_thinking_texts or len(full_thinking_texts) != batch_size:
            self.logger.warning("Failed to generate baseline answers for batch")
            return None, None, None, None
        
        # Select appropriate tokenizer for this baseline
        tokenizer = self.clean_tokenizer if hasattr(self, 'clean_tokenizer') else self.tokenizer
        
        # Step 2: Build conversations with full thinking text + gold answers
        # Use the full thinking text (including </think> token) followed by gold answer
        conversations = []
        for i in range(batch_size):
            conversation = [
                self.SYSTEM_PROMPT,
                batch_processed["conversations"][i][0],  # User question
            ]
            
            if len(batch_processed["conversations"][i]) > 2:
                conversation += [
                    batch_processed["conversations"][i][1],  # tool answer
                    {"role": "assistant", "content": full_thinking_texts[i]+"\n</think>"+batch_processed["conversations"][i][2]["content"]},  # Generated + Gold
                ]
            else:
                conversation += [
                    {
                        "role": "assistant", 
                        "content": full_thinking_texts[i]+"\n</think>"+ batch_processed["conversations"][i][1]["content"]
                    },  # Generated + Gold
                ]
            conversations.append(conversation)
        
        # Format prompts
        prompts = [
            self._format_chat_messages(conv, add_generation_prompt=False)
            for conv in conversations
        ]
        
        # Tokenize
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side="left",
            return_offsets_mapping=True,
        ).to(self.accelerator.device)
        
        # Compute logits
        outputs = model(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
        )
        
        # Find object boundaries
        object_boundaries = self._find_object_boundaries(
            prompts, batch["objects"], encoded
        )
        
        return outputs.logits, encoded['input_ids'], encoded['attention_mask'], object_boundaries

    

    # ==========================
    # Fuzzy-matching evaluation using grammar parsing
    # ========================
    def _extract_gold_answers_for_sample(self, batch: Dict[str, Any], sample_index: int) -> List[str]:
        """
        Extract gold answer strings from various possible batch fields.
        
        Different datasets store answers in different formats. This method tries
        to find answers in common locations: object labels, answer lists, and metadata.
        
        Args:
            batch: The data batch
            sample_index: Index of the sample within the batch
            
        Returns:
            List of unique gold answer strings
        """
        answers: List[str] = []
        
        # 1. Check object label fields
        obj = batch["objects"][sample_index]
        for key in ("object_label", "label", "answer"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                answers.append(value.strip())
        
        # 2. Check sample-level answer lists
        sample_answers = None
        if isinstance(batch.get("answers"), list):
            sample_answers = batch["answers"][sample_index]
        elif isinstance(batch.get("gold_answers"), list):
            sample_answers = batch["gold_answers"][sample_index]
        
        if sample_answers:
            if isinstance(sample_answers, (list, tuple)):
                answers.extend([str(ans).strip() for ans in sample_answers if str(ans).strip()])
            elif isinstance(sample_answers, str):
                answers.append(sample_answers.strip())
        
        # 3. Check sentence metadata
        if isinstance(batch.get("sentences_meta"), list):
            meta = batch["sentences_meta"][sample_index]
            answer = meta.get("answer")
            if isinstance(answer, str) and answer.strip():
                answers.append(answer.strip())
        
        # De-duplicate while preserving order
        unique_answers = list(dict.fromkeys(answers))
        return unique_answers

    def _extract_noun_phrases(self, text: str) -> List[str]:
        """Extract noun phrases and proper nouns from text using NLTK grammar parsing."""
        try:
            # Tokenize and POS tag
            tokens = nltk.word_tokenize(text)
            if not tokens:
                return []
            
            tagged_sentence = nltk.pos_tag(tokens)
            
            # Grammar for extracting noun phrases and proper nouns
            grammar = r"""
                NP:{<JJ|NN><POS|IN>?<NN>+}
                PP:{<NN|NNS|NNP|NNPS>}
            """
            parser = nltk.RegexpParser(grammar)
            parsed_result = parser.parse(tagged_sentence)
            
            # Extract phrases from parsed tree
            phrases = []
            for subtree in parsed_result.subtrees():
                if subtree.label() in ['NP', 'PP']:
                    phrase = ' '.join(word for word, tag in subtree)
                    phrases.append(phrase.strip())
            
            return phrases
        except Exception as e:
            self.logger.warning(f"Error extracting noun phrases: {e}")
            return []
    
    def _check_fuzzy_match(self, candidate: str, gold_answers: List[str], 
                          exact_threshold: int = 50, partial_threshold: int = 70) -> bool:
        """Check if candidate matches any gold answer using fuzzy matching."""
        if not candidate:
            return False
        
        candidate_lower = candidate.lower().strip()
        for answer in gold_answers:
            answer_lower = answer.lower().strip()
            
            # Check exact fuzzy ratio
            if fuzz.ratio(answer_lower, candidate_lower) > exact_threshold:
                return True
            
            # Check partial ratio for substring matches
            if fuzz.partial_ratio(answer_lower, candidate_lower) > partial_threshold:
                return True
        
        return False
    
    def parse_answer_with_grammar(self, generated_text: str, gold_answers: List[str]) -> int:
        """
        Parse generated text and check for fuzzy matches with gold answers.
        
        Args:
            generated_text: The text generated by the model
            gold_answers: List of acceptable gold answer strings
            
        Returns:
            1 if a match is found, 0 otherwise
        """
        if not generated_text or not gold_answers:
            return 0
        
        try:
            normalized_text = generated_text.lower().strip()
            
            # Check for uncertainty expressions
            if "i'm sorry" in normalized_text or "i don't know" in normalized_text:
                return 0
            
            # Extract noun phrases using grammar parsing
            phrases = self._extract_noun_phrases(normalized_text)
            
            # Add the full text for direct matching
            phrases.append(normalized_text)
            
            # Check all extracted phrases against gold answers
            for phrase in phrases:
                if self._check_fuzzy_match(phrase, gold_answers):
                    return 1
                    
        except Exception as e:
            self.logger.warning(f"Error in grammar-based answer parsing: {e}")
        
        return 0
    


    # ==========================
    # EVALUATION FUNCTIONS
    # =========================
    
    # -------------------------
    # Hit@k evaluation using logits prediction
    # -------------------------
    def compute_hit_k_metrics(self, k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        Compute Hit@k metrics for object label prediction across all model types.
        
        For all models, we generate final answers and then compute logits.
        This ensures fair comparison using the same generation-then-logits pipeline.
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
            
            if self.accelerator.is_main_process and is_webqsp:
                self.logger.info("Using WebQSP evaluation mode: computing both per-sample and per-question metrics")
            
            # Collect all question predictions across batches for WebQSP
            all_question_ranks = defaultdict(list) if is_webqsp else None

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(self.dataloader, desc=f"Computing Hit@k metrics for {name}", disable=not self.accelerator.is_main_process)):
                    if self.max_samples and (batch_idx * self.batch_size * self.accelerator.num_processes) >= self.max_samples:
                        break

                    # Handle different model types
                    if name == "KG_LM":
                        # For thinking model, run the thinking loop and get logits
                        logits, input_ids, attention_mask, object_boundaries = self._compute_thinking_logits_batched(batch)
                        if logits is None:
                            continue
                    else:
                        # For baseline models, generate answers and then compute logits
                        logits, input_ids, attention_mask, object_boundaries = self._compute_baseline_logits_batched(batch, preprocess_func, model)
                        if logits is None:
                            continue
                    
                    if not is_webqsp:
                        # Standard hit@k computation for non-WebQSP datasets
                        hit_k_correct_batch, batch_avg_num_tokens, new_objects = compute_hit_k(
                            logits, input_ids, k_values,
                            object_boundaries, 
                            self.model_config.num_quantizers + (2 if hasattr(self.model_config, 'bounding_tokens') and self.model_config.bounding_tokens else 0),
                            attention_mask,
                            special_token=self.special_kg_token_id, 
                            tokenizer=self.tokenizer,
                            return_individual_ranks=False
                        )
                        
                        for k in k_values:
                            hit_k_correct[k] += hit_k_correct_batch[k]
                            
                        total_objects += new_objects
                    else:
                        # WebQSP evaluation with per-question aggregation
                        individual_ranks, batch_avg_num_tokens, new_objects = compute_hit_k(
                            logits, input_ids, k_values,
                            object_boundaries, 
                            self.model_config.num_quantizers + (2 if hasattr(self.model_config, 'bounding_tokens') and self.model_config.bounding_tokens else 0),
                            attention_mask,
                            special_token=self.special_kg_token_id, 
                            tokenizer=self.tokenizer,
                            return_individual_ranks=True
                        )
                        
                        question_ids = batch.get('question_ids', None)
                        
                        if question_ids is None:
                            # Create unique question IDs across processes to avoid conflicts
                            batch_size = len(input_ids)
                            base_id = batch_idx * self.accelerator.num_processes * batch_size + self.accelerator.process_index * batch_size
                            question_ids = list(range(base_id, base_id + batch_size))
                            
                            if batch_idx == 0 and self.accelerator.is_main_process:
                                self.logger.warning(f"No 'question_ids' found in batch data. Using generated IDs starting from {base_id}.")
                        
                        # For WebQSP, collect ranks per question for later aggregation
                        for qid, rank in zip(question_ids, individual_ranks):
                            if rank != float('inf'):  # Only collect valid ranks
                                all_question_ranks[qid].append(rank)
                                
                    average_num_tokens += batch_avg_num_tokens
                    total_samples += new_objects

            # Handle WebQSP aggregation (same as base evaluator)
            if is_webqsp and all_question_ranks is not None:
                # Gather question ranks from all processes
                gathered_question_ranks = []
                
                for process_idx in range(self.accelerator.num_processes):
                    if process_idx == self.accelerator.process_index:
                        process_data = dict(all_question_ranks)
                    else:
                        process_data = {}
                    
                    data_to_broadcast = [process_data]
                    broadcast_object_list(data_to_broadcast, from_process=process_idx)
                    gathered_question_ranks.append(data_to_broadcast[0])
                
                if self.accelerator.is_main_process:
                    # Merge question ranks from all processes
                    merged_question_ranks = defaultdict(list)
                    for process_data in gathered_question_ranks:
                        for qid, ranks in process_data.items():
                            merged_question_ranks[qid].extend(ranks)
                    
                    self.logger.info(f"Total unique questions after merging: {len(merged_question_ranks)}")
                    
                    if merged_question_ranks:
                        # For each question, take the best (minimum) rank across all linked entities
                        question_best_ranks = {qid: min(ranks) for qid, ranks in merged_question_ranks.items()}
                        
                        # Compute hit@k based on question-level best ranks
                        hit_k_tensors = {}
                        for k in k_values:
                            hits = sum(1 for rank in question_best_ranks.values() if rank <= k)
                            hit_k_tensors[k] = hits
                        
                        total_objects_gathered = len(merged_question_ranks)
                    else:
                        hit_k_tensors = {k: 0 for k in k_values}
                        total_objects_gathered = 0
                else:
                    hit_k_tensors = {k: 0 for k in k_values}
                    total_objects_gathered = 0
                
                # Broadcast the final results to all processes for consistent state
                results_to_broadcast = [hit_k_tensors, total_objects_gathered]
                broadcast_object_list(results_to_broadcast, from_process=0)
                hit_k_tensors, total_objects_gathered = results_to_broadcast
            else:
                # For non-WebQSP datasets, gather results normally
                hit_k_tensors = {}
                for k in k_values:
                    hit_k_tensor = torch.tensor(hit_k_correct[k], device=self.accelerator.device)
                    hit_k_tensors[k] = self.accelerator.gather(hit_k_tensor).sum().item()
                
                total_objects_tensor = torch.tensor(total_objects, device=self.accelerator.device)
                total_objects_gathered = self.accelerator.gather(total_objects_tensor).sum().item()
            
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
                    for k in k_values:
                        hit_rate = hit_k_tensors[k] / total_objects_gathered
                        metrics[f'hit@{k}'] = hit_rate
                        self.logger.info(f"Hit@{k} for {name}: {hit_rate:.4f} ({hit_k_tensors[k]}/{total_objects_gathered})")
                    
                    if is_webqsp:
                        metrics['questions'] = total_objects_gathered
                    else:
                        metrics['samples'] = total_samples_gathered
                        
                    if average_num_tokens_gathered > 0:
                        metrics['average_num_tokens'] = average_num_tokens_gathered / total_samples_gathered
                else:
                    self.logger.warning(f"No valid objects found for Hit@k computation for {name}")
                    for k in k_values:
                        metrics[f'hit@{k}'] = 0.0
                    metrics['samples'] = 0
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
    
    
    # -------------------------
    # Fuzzy-matching evaluation (string-level)
    # -------------------------
    def compute_fuzzy_metrics(
        self,
        threshold: int = 50,          # fuzzy ratio threshold to count as hit
        k_values: List[int] = [1,3,5] # optional @k reporting on top fuzzy candidates
    ) -> Dict[str, float]:
        """
        Generate final answers with the thinking loop and evaluate them by fuzzy string matching
        against gold answers. For WebQSP, aggregate per question (best among linked entities).
        """
        if self.accelerator.is_main_process:
            self.logger.info(f"Computing fuzzy-matching metrics (threshold={threshold}) with thinking...")

        results = {}
        for name, (preprocess_func, model) in self.tests.items():
            # Handle all models, but use different generation strategies
            if self.accelerator.is_main_process:
                self.logger.info(f"Computing fuzzy metrics for {name}...")

            total = 0
            correct = 0

            # WebQSP aggregation
            is_webqsp = (self.config.dataset.name == "web-qsp")
            per_question_best: Dict[Any, float] = defaultdict(float) if is_webqsp else None

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(self.dataloader, desc=f"Computing fuzzy metrics for {name}", disable=not self.accelerator.is_main_process)):
                    if self.max_samples and (batch_idx * self.batch_size * self.accelerator.num_processes) >= self.max_samples:
                        break

                    # Handle different model types for text generation
                    try:
                        if name == "KG_LM":
                            # Run the full thinking pass and get final decoded texts
                            final_texts = self._generate_with_thinking_batched(batch)
                        else:
                            # For baseline models, generate answers directly
                            final_texts = self._generate_baseline_answers_batched(batch, preprocess_func, model)
                        
                        if not final_texts:
                            self.logger.warning(f"No texts generated for batch {batch_idx} with {name}")
                            continue
                            
                    except Exception as e:
                        self.logger.error(f"Error generating texts for {name} at batch {batch_idx}: {e}")
                        continue

                    # Question ids (for WebQSP)
                    question_ids = batch.get('question_ids', None)
                    if question_ids is None:
                        # build stable synthetic ids
                        bs = len(final_texts)
                        base_id = batch_idx * self.accelerator.num_processes * bs + self.accelerator.process_index * bs
                        question_ids = list(range(base_id, base_id + bs))

                    # For each sample: fuzzy-match generated text vs gold answers
                    for i, gen_text in enumerate(final_texts):
                        if i >= len(batch["sentences"]):
                            break
                            
                        golds = self._extract_gold_answers_for_sample(batch, i)
                        if not golds:
                            continue
                        
                        # Use grammar-based parsing to check for answer match
                        try:
                            correct_s = self.parse_answer_with_grammar(gen_text, golds)
                        except Exception as e:
                            self.logger.warning(f"Error in grammar parsing for sample {i}: {e}")
                            correct_s = 0
                            
                        if is_webqsp:
                            qid = question_ids[i]
                            per_question_best[qid] = max(per_question_best[qid], correct_s)
                        else:
                            total += 1
                            correct += correct_s

            # Gather across processes
            if not is_webqsp:
                total_t = torch.tensor(total, device=self.accelerator.device)
                correct_t = torch.tensor(correct, device=self.accelerator.device)
                total_g = self.accelerator.gather(total_t).sum().item()
                correct_g = self.accelerator.gather(correct_t).sum().item()
                self.accelerator.wait_for_everyone()

                metrics = {}
                if self.accelerator.is_main_process:
                    acc = (correct_g / total_g) if total_g > 0 else 0.0
                    metrics["fuzzy@%d" % threshold] = acc
                    self.logger.info(f"Fuzzy accuracy @ {threshold}: {acc:.4f} over {total_g} samples")
                results[name] = metrics
            else:
                # Broadcast dicts & merge on main
                gathered = []
                for p in range(self.accelerator.num_processes):
                    local = dict(per_question_best) if p == self.accelerator.process_index else {}
                    payload = [local]
                    broadcast_object_list(payload, from_process=p)
                    gathered.append(payload[0])

                if self.accelerator.is_main_process:
                    merged: Dict[Any, float] = {}
                    for d in gathered:
                        for qid, score in d.items():
                            merged[qid] = max(merged.get(qid, 0.0), score)
                    valid_q = len(merged)
                    
                    acc = sum(merged.values()) / valid_q if valid_q > 0 else 0.0
                    metrics = {"fuzzy@%d" % threshold: acc, "questions": valid_q}
                else:
                    metrics = {}
                
                # broadcast final metrics
                blob = [metrics]
                broadcast_object_list(blob, from_process=0)
                results[name] = blob[0]

        return results
    
    