import copy
import logging
import os
import os.path as osp
import warnings

from abc import ABC
from collections import OrderedDict
from typing import Optional, Tuple, Union, List

from transformers import (
    AutoConfig, AutoModel, AutoModelForCausalLM,
    AutoTokenizer, PreTrainedTokenizer,
    PreTrainedModel, GenerationConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

import torch
from KG_LM.model.KG_encoder import KGEncoder
from KG_LM.model.KG_decoder import KGDecoder

from torch_geometric.data import Batch

# PEFT imports for LoRA support
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# Constants
from KG_LM.configuration import IGNORE_INDEX, SPECIAL_KG_TOKEN, ModelConfig

def infer_stop_tokens(tokenizer):
    """Simple implementation of infer_stop_tokens for KG_LM"""
    stop_tokens = []
    if tokenizer.eos_token:
        stop_tokens.append(tokenizer.eos_token)
    return stop_tokens

class KG_LMConfig(AutoConfig):
    """
    Configuration class for the Knowledge Graph Language Foundation Model (KG_LM).
    This class extends AutoConfig to include specific parameters for the KG_LM model.
    
    """
    
    model_type = "KG_LM"
    llm_model_name: str = "Qwen/Qwen3-8B"
    "Name of the LLM model"

    node_embedding_dim: int = 1024  
    "Dimension of node embeddings"
    edge_embedding_dim: int = 1024  
    "Dimension of edge embeddings"
    graph_pooling: bool = False
    "Whether to apply global mean pooling to the graph representations"
    q_former: bool = False
    """Whether to use a Q-Former architecture to pool graph representations. Defaults to True."""

    dropout: float = 0.2  
    "Dropout rate for the model"
    num_heads: int = 1  
    "Number of attention heads in GATv2Conv"
    gat_layers: int = 1
    """Number of GATv2 layers to apply."""
    num_quantizers: int = 3  
    "Number of quantizers for residual vector quantization"

    codebook_size: int = 512  
    "Size of the codebook for vector quantization"
    codebook_dim: int = 0  
    "Dimension of the downsampled codebook. If 0, no downsampling is applied."
    shared_codebook: bool = False  
    "Whether to use a shared codebook across quantizers"
    dead_codebook_threshold: float = 0.5
    "Threshold for dead codebook entries"

    tune_language_model: bool = False  
    "Whether to tune the language model"
    tune_kg_encoder: bool = False  
    "Whether to tune the knowledge graph encoder"
    tune_kg_decoder: bool = False  
    "Whether to tune the knowledge graph decoder"

    bounding_tokens: bool = False
    "Whether to use special bounding tokens around the KG embeddings in the input sequence"

    use_lora: bool = True  
    "Whether to use LoRA for training"
    lora_r: int = 8  
    "Rank for LoRA"
    lora_alpha: int = 16  
    "Alpha for LoRA scaling"
    lora_target_modules: List[str] = ["q_proj", "k_proj"]

    use_kg_decoder: bool = False  
    "Whether to use the KG decoder for prediction tasks"
    
    use_kg_encoder: bool = True
    "Whether to use the KG encoder for embedding the input graphs"

def set_KGLM_model_args(config :KG_LMConfig, model_args: ModelConfig):
    """
    Set the model arguments from a ModelConfig instance.
    
    Args:
        config (KG_LMConfig): An instance of KG_LMConfig to be updated with model parameters.
        model_args (ModelConfig): An instance of ModelConfig containing model parameters.
    """
    # get config for the LLM model to know the shape of the embeddings
    graph_nodes_embedding_model = model_args.graph_nodes_embedding_model
    graph_llm_config = AutoConfig.from_pretrained(
        graph_nodes_embedding_model,
    )
    # Check if the model is compatible with KG_LM
    if not hasattr(graph_llm_config, 'hidden_size'):
        raise ValueError(f"The model {graph_nodes_embedding_model} does not have a 'hidden_size' attribute. "
                         "Please use a compatible model for KG_LM.")
    
    config.model_type = "KG_LM"    
    
    config.node_embedding_dim = graph_llm_config.hidden_size
    config.edge_embedding_dim = graph_llm_config.hidden_size
    
    config.llm_model_name = model_args.llm_model_name
    
    config.graph_pooling = model_args.graph_pooling
    config.q_former = model_args.q_former
    config.dropout = model_args.dropout
    config.num_heads = model_args.num_heads
    config.gat_layers = model_args.gat_layers
    config.num_quantizers = model_args.num_quantizers
    config.codebook_size = model_args.codebook_size
    config.codebook_dim = model_args.codebook_dim
    config.shared_codebook = model_args.shared_codebook
    config.dead_codebook_threshold = model_args.dead_codebook_threshold
    config.tune_language_model = model_args.tune_language_model
    config.tune_kg_encoder = model_args.tune_kg_encoder
    config.tune_kg_decoder = model_args.tune_kg_decoder
    config.use_lora = model_args.use_lora
    config.lora_r = model_args.lora_r
    config.lora_alpha = model_args.lora_alpha
    config.lora_target_modules = model_args.lora_target_modules
    config.use_kg_decoder = model_args.use_kg_decoder
    config.bounding_tokens = model_args.bounding_tokens
    config.use_kg_encoder = model_args.use_kg_encoder
    
    return config


class KG_LMMetaModel(ABC):
    def init_KGLM(self, config: KG_LMConfig = None, *args, **kwargs):
        if hasattr(self, "llm") or hasattr(self, "kg_encoder"):
            return 

        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_model_name, trust_remote_code=True)
        
        # Freeze layers if configured
        if not config.tune_language_model:
            logging.info("Freezing language model parameters.")
            for param in self.llm.parameters():
                param.requires_grad = False
        
        # Apply lora if configured
        if hasattr(config, "use_lora") and config.use_lora:
            logging.info("Applying LoRA to the language model.")
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                task_type=TaskType.CAUSAL_LM,
                lora_dropout=config.dropout,
            )
            self.llm : PeftModel = get_peft_model(self.llm, lora_config)
        
        self.llm_embedding_dim = self.llm.config.hidden_size
        
        self.kg_encoder = None
        # condition to be compatible with older configs that do not have use_kg_encoder
        if not hasattr(config, "use_kg_encoder") or config.use_kg_encoder:
            logging.info("Initializing KG Encoder.")
            # Initialize the KGEncoder with the specified dimensions
            self.kg_encoder = KGEncoder(
                node_embedding_dim=config.node_embedding_dim,
                edge_embedding_dim=config.edge_embedding_dim,
                final_embedding_dim=self.llm_embedding_dim,
                dropout=config.dropout,
                num_heads=config.num_heads,
                gat_layers=config.gat_layers if hasattr(config, "gat_layers") else 1,
                num_quantizers=config.num_quantizers,
                codebook_size=config.codebook_size,
                codebook_dim=config.codebook_dim if hasattr(config, "codebook_dim") else 0,
                shared_codebook=config.shared_codebook,
                graph_pooling=config.graph_pooling,
                q_former=config.q_former if hasattr(config, "q_former") else False,
                beta_commit=0.25,
                std_tokens=self.llm.get_input_embeddings().weight.std().item(),
                dead_codebook_threshold=config.dead_codebook_threshold if hasattr(config, "dead_codebook_threshold") else 0.5,
                bounding_tokens=config.bounding_tokens if hasattr(config, "bounding_tokens") else False
            )

        # Initialize the KGDecoder if configured
        self.kg_decoder = None
        if hasattr(config, "use_kg_decoder") and config.use_kg_decoder:
            logging.info("Initializing KG Decoder.")
            self.kg_decoder = KGDecoder(
                node_embedding_dim=config.node_embedding_dim,
                edge_embedding_dim=config.edge_embedding_dim,
                final_embedding_dim=self.llm_embedding_dim,
                dropout=config.dropout,
                num_quantizers=config.num_quantizers,
            )
            
            # Freeze decoder if configured
            if not getattr(config, "tune_kg_decoder", False):
                logging.info("Freezing KG Decoder parameters.")
                for param in self.kg_decoder.parameters():
                    param.requires_grad = False

        assert (
            self.llm is not None
        ), "KG_LM model initialization failed. Please check the configuration and ensure that the LLM and KGEncoder are properly initialized."

        self.is_loaded = True
        
    @classmethod
    def load_pretrained(cls, model_path_or_config, *args, **kwargs):
        """
        Loads a pretrained KG_LM model from a specified directory.

        This method reconstructs the model by:
        1. Loading the KG_LM configuration.
        2. Initializing the model architecture (`cls`).
        3. Loading the pretrained weights for the language model (`llm`).
        4. Loading the pretrained weights for the knowledge graph encoder (`kg_encoder`).
        5. Loading the correct tokenizer.
        """
        if not osp.isdir(model_path_or_config):
            raise ValueError(f"The path '{model_path_or_config}' is not a valid directory.")

        # Load the main configuration file for the KG_LM model
        config : KG_LMConfig = KG_LMConfig.from_pretrained(model_path_or_config, **kwargs)

        # Initialize the model instance (e.g., KG_LM) using the loaded config.
        # This will call `init_KGLM` and create the base architecture.
        model = cls(config, *args, **kwargs)

        # Load the pretrained language model from the 'llm' subdirectory
        llm_path = osp.join(model_path_or_config, "llm")
        if osp.exists(llm_path):
            logging.info(f"Loading pretrained LLM from {llm_path}")
            model.llm = AutoModelForCausalLM.from_pretrained(llm_path, trust_remote_code=True)
            if config.use_lora:
                model.llm = PeftModel.from_pretrained(model.llm, llm_path, trust_remote_code=True)
                # make require grad True for all parameters in the LoRA layers
                for name, param in model.llm.named_parameters():
                    if "lora" in name or config.tune_language_model:
                        param.requires_grad = True
        else:
            warnings.warn(
                f"LLM directory not found at {llm_path}. "
                "The model will use the base LLM initialized from the config's `llm_model_name`."
            )
            
        # Load the tokenizer from the 'llm' subdirectory to ensure it matches the saved model
        if osp.exists(llm_path):
            logging.info(f"Loading tokenizer from {llm_path}")
            model.tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)

        # Load the state dictionary for the KG Encoder
        kg_encoder_path = osp.join(model_path_or_config, "kg_encoder.pth")
        if osp.exists(kg_encoder_path):
            logging.info(f"Loading pretrained KG Encoder from {kg_encoder_path}")
            kg_encoder_state_dict = torch.load(kg_encoder_path, map_location='cpu')
            model.kg_encoder.load_state_dict(kg_encoder_state_dict)
        else:
            warnings.warn(
                f"KG Encoder weights not found at {kg_encoder_path}. "
                "The model will use a randomly initialized KG Encoder."
            )

        # Load the state dictionary for the KG Decoder if it exists
        kg_decoder_path = osp.join(model_path_or_config, "kg_decoder.pth")
        if osp.exists(kg_decoder_path) and hasattr(model, 'kg_decoder') and model.kg_decoder is not None:
            logging.info(f"Loading pretrained KG Decoder from {kg_decoder_path}")
            kg_decoder_state_dict = torch.load(kg_decoder_path, map_location='cpu')
            model.kg_decoder.load_state_dict(kg_decoder_state_dict)
        elif hasattr(model, 'kg_decoder') and model.kg_decoder is not None:
            warnings.warn(
                f"KG Decoder weights not found at {kg_decoder_path}. "
                "The model will use a randomly initialized KG Decoder."
            )

        model.is_loaded = True
        return model


    def save_pretrained(self, output_dir, state_dict=None):
        if state_dict is None:
            state_dict = self.state_dict()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Define specific paths for components
        llm_dir = osp.join(output_dir, "llm")
        kg_encoder_path = osp.join(output_dir, "kg_encoder.pth")
        kg_decoder_path = osp.join(output_dir, "kg_decoder.pth")
        
        # Save tokenizer and llm config/weights
        if self.get_llm():
            logging.info(f"Saving LLM and tokenizer to {llm_dir}")
            # The tokenizer is part of the class, save it to the same directory as the llm
            if getattr(self, "tokenizer", None):
                self.tokenizer.save_pretrained(llm_dir)
            
            self.llm.config._name_or_path = llm_dir
            self.llm.save_pretrained(llm_dir)

        # Save KG encoder weights
        if self.get_kg_encoder():
            logging.info(f"Saving KG Encoder to {kg_encoder_path}")
            kg_encoder_state_dict = OrderedDict(
                {k.split("kg_encoder.")[-1]: v for k, v in state_dict.items() if "kg_encoder." in k}
            )
            torch.save(kg_encoder_state_dict, kg_encoder_path)

        # Save KG decoder weights
        if self.get_kg_decoder():
            logging.info(f"Saving KG Decoder to {kg_decoder_path}")
            kg_decoder_state_dict = OrderedDict(
                {k.split("kg_decoder.")[-1]: v for k, v in state_dict.items() if "kg_decoder." in k}
            )
            torch.save(kg_decoder_state_dict, kg_decoder_path)

        # Save the main model config
        self.config._name_or_path = output_dir
        self.config.architectures = [self.__class__.__name__]
        self.config.save_pretrained(output_dir)

    def get_llm(self):
        llm = getattr(self, "llm", None)
        return llm

    def get_lm_head(self):
        lm_head = getattr(self.get_llm(), "lm_head", None)
        return lm_head

    def get_kg_encoder(self):
        kg_encoder = getattr(self, "kg_encoder", None)
        return kg_encoder

    def get_kg_decoder(self):
        kg_decoder = getattr(self, "kg_decoder", None)
        return kg_decoder

    def correct_train(self):
        '''
        To ensure the expected behaviors for modules like dropout, batchnorm, etc., we need to call model.eval() for the freezed modules.
        '''
        self.train()
        
        if self.get_llm() and not getattr(self.config, "tune_language_model", False):
            self.get_llm().eval()
            logging.debug("Freezed llm model to eval mode.")
        if self.get_kg_encoder() and not getattr(self.config, "tune_kg_encoder", False):
            self.get_kg_encoder().eval()
            logging.debug("Freezed kg_encoder model to eval mode.")
        if self.get_kg_decoder() and not getattr(self.config, "tune_kg_decoder", False):
            self.get_kg_decoder().eval()
            logging.debug("Freezed kg_decoder model to eval mode.")
    
    def encode_graphs(self, graphs):
        kg_encoder = self.get_kg_encoder()
        if kg_encoder is None:
            return None, None, None
        
        graph_features, indexes, RVQ_loss = kg_encoder(graphs)
        return graph_features, indexes, RVQ_loss
    
    def decode_graphs(
        self, 
        quantized_tokens, 
        original_graphs=None,
        mode="both"
    ):
        """
        Perform edge prediction and/or object prediction using the KG decoder.
        
        Args:
            quantized_tokens: Quantized token representations from encoder [B, Q, D]
            original_graphs: Original graph batch for context (optional)
            mode: Prediction mode - "edge_prediction", "object_prediction", or "both"
            
        Returns:
            Decoder output dict or None if decoder not available
        """
        kg_decoder = self.get_kg_decoder()
        if kg_decoder is None:
            return None
            
        return kg_decoder(
            quantized_tokens=quantized_tokens,
            original_graphs=original_graphs,
            mode=mode
        )
    
    def _temporary_reorder_cache(self, past_key_values, sorted_idx):
        return self.get_llm()._temporary_reorder_cache(past_key_values, sorted_idx)

    def get_input_embeddings(self):
        return self.get_llm().get_input_embeddings()

    def get_output_embeddings(self):
        return self.get_llm().get_output_embeddings()


class KG_LMMetaForCausalLM(ABC):
    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        graphs,
    ):
        """
        Prepares the inputs and labels for multimodal training by integrating graph embeddings
        into the text sequence at positions marked by special KG tokens.
        
        This method handles the complex process of:
        1. Encoding graphs into embeddings
        2. Replacing special KG tokens with actual graph embeddings
        3. Properly padding and aligning sequences for batch processing
        4. Managing attention masks and position IDs for the modified sequences
        """
        RVQ_loss = None
        
        # Early return for cases where no multimodal processing is needed
        if graphs is None:
            logging.debug("No graphs provided, returning standard inputs and labels.")
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None, # Return standard embeddings
                labels,
                RVQ_loss,
                None,  # No indices for graph embeddings
                None  # No quantized tokens
            )

        logging.debug(f"Preparing inputs and labels for multimodal processing with {len(graphs)} graphs...")
        # Encode the knowledge graphs into embeddings that will replace KG tokens
        graph_features, indices, RVQ_loss = self.encode_graphs(graphs)
        
        if graph_features is not None:
            graph_features = graph_features.to(self.llm.dtype)

        processed_graph = 0
        
        if graph_features is not None:
            logging.debug(f"Graph features shape: {graph_features.shape}, RVQ loss: {RVQ_loss}")

        # Store original inputs for later reference
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        # If position IDs are not provided, create them based on input IDs
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(input_ids.shape[0], -1)

        # Create attention mask if not provided (all tokens are attended to)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        # Create labels if not provided (ignore all tokens in loss calculation)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # Get embeddings for text tokens, replacing KG tokens with padding token (0) temporarily
        # This prevents embedding lookup errors for the special KG token
        input_ids_copy = input_ids.clone()
        input_ids_copy[input_ids_copy == self.special_kg_token] = 0
        input_embeds = self.llm.get_input_embeddings()(input_ids_copy)

        # Extract valid tokens based on attention mask for each sample in the batch
        # This removes padding tokens from processing
        input_ids_unpadded = [
            cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        input_embeds_unpadded = [
            cur_input_embeds[cur_attention_mask]
            for cur_input_embeds, cur_attention_mask in zip(input_embeds, attention_mask)
        ]
        labels_unpadded = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # Lists to store the final processed embeddings and labels
        new_input_embeds = []
        new_labels = []
        
        # Process each sample in the batch individually
        for batch_idx, cur_input_ids in enumerate(input_ids_unpadded):
            # Count how many KG tokens are in this sample
            num_kg_tokens = (cur_input_ids == self.special_kg_token).sum()
            
            if num_kg_tokens == 0:
                # No KG tokens found, just use the original text embeddings
                new_input_embeds.append(input_embeds_unpadded[batch_idx])
                new_labels.append(labels_unpadded[batch_idx])
                continue

            # Get the current embeddings and labels for this sample
            cur_input_embeds = input_embeds_unpadded[batch_idx]
            
            # Find positions of KG tokens and create segments between them
            # Add -1 at start and sequence length at end to handle boundaries
            kg_token_indices = (
                [-1] + torch.where(cur_input_ids == self.special_kg_token)[0].tolist() + [cur_input_ids.shape[0]]
            )
            
            cur_labels = labels_unpadded[batch_idx]

            # Split the sequence into segments: text before first KG token, between KG tokens, after last KG token
            text_segments = []
            label_segments = []
            
            # Extract text segments
            for i in range(len(kg_token_indices) - 1):
                start_idx = kg_token_indices[i] + 1
                end_idx = kg_token_indices[i+1]
                text_segments.append(cur_input_embeds[start_idx:end_idx])
                label_segments.append(cur_labels[start_idx:end_idx])
            
            # Reconstruct the sequence by interleaving text segments with graph embeddings
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_kg_tokens):
                cur_new_input_embeds.append(text_segments[i])
                cur_new_labels.append(label_segments[i])
                
                # Insert graph embedding
                if graph_features is not None:
                    cur_graph_feature = graph_features[processed_graph]
                else:
                    cur_graph_feature = torch.Tensor([]).to(input_embeds.device)
                processed_graph += 1
                cur_new_input_embeds.append(cur_graph_feature)
                
                # Mark graph embedding positions to be ignored in loss calculation
                cur_new_labels.append(
                    torch.full(
                        (cur_graph_feature.shape[0],),
                        IGNORE_INDEX,
                        device=cur_labels.device,
                        dtype=cur_labels.dtype,
                    )
                )

            # Append the final text segment
            cur_new_input_embeds.append(text_segments[-1])
            cur_new_labels.append(label_segments[-1])

            # Concatenate all segments and graph embeddings into final sequence
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            cur_new_labels = torch.cat(cur_new_labels, dim=0)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Handle sequence length limits by truncating if necessary
        tokenizer_model_max_length = getattr(self.llm.config, "model_max_length", None)
        if tokenizer_model_max_length is not None:
            if any(len(x) > tokenizer_model_max_length for x in new_input_embeds):
                warnings.warn("Inputs truncated due to model_max_length!")
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Batch processing: pad all sequences to the same length
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        # Initialize padded tensors with appropriate default values
        new_input_embeds_padded = torch.zeros(
            (batch_size, max_len, new_input_embeds[0].shape[1]),
            dtype=new_input_embeds[0].dtype,
            device=new_input_embeds[0].device
        )
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,  # Padded positions should be ignored in loss
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        # Create new attention mask for the modified sequences
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=_attention_mask.dtype if _attention_mask is not None else torch.long,
            device=input_ids.device,
        )
        # Create new position IDs for the modified sequences
        position_ids = torch.zeros(
            (batch_size, max_len), 
            dtype=_position_ids.dtype if _position_ids is not None else torch.long, 
            device=input_ids.device
        )

        # Pad each sequence and update corresponding masks and position IDs
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            
            # Handle left padding (useful for some model architectures)
            if getattr(self.llm.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded[i, -cur_len:] = cur_new_embed
                new_labels_padded[i, -cur_len:] = cur_new_labels
                attention_mask[i, -cur_len:] = 1
                position_ids[i, -cur_len:] = torch.arange(0, cur_len, device=position_ids.device)
            else:
                # Handle right padding (default case)
                new_input_embeds_padded[i, :cur_len] = cur_new_embed
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = 1
                position_ids[i, :cur_len] = torch.arange(0, cur_len, device=position_ids.device)

        # Preserve None values for optional inputs that were originally None
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            # Ensure attention mask has the same dtype as the original
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        logging.debug(f"Processed input embeddings shape: {new_input_embeds_padded.shape}, Labels shape: {new_labels_padded.shape}, RVQ loss: {RVQ_loss}")

        # Return the processed inputs ready for the language model
        return (
            None,  # input_ids set to None since we're using input_embeds
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds_padded,  # The main output: text + graph embeddings
            new_labels,
            RVQ_loss,  # Return the RVQ loss for training purposes
            indices,
            graph_features,  # Return graph features for decoder
        )
    
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        graphs: Optional[Batch] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **generation_kwargs
    ) -> torch.Tensor:
        """
        Generate text based on the input IDs and (optionally) graphs.

        Args:
            input_ids (torch.Tensor): Input token IDs for the language model.
            graphs (Optional[Batch]): A batch of graphs from PyTorch Geometric. If None, falls back to pure text generation.
            attention_mask (Optional[torch.Tensor]): Attention mask for the input tokens.
            **generation_kwargs: Additional keyword arguments for generation.

        Returns:
            torch.Tensor: Generated token IDs (only the newly generated tokens, not including input).
        """
        
        gen_config = self.default_generation_config
        forward_only_kwargs = {}
        for key, value in list(generation_kwargs.items()):
            if hasattr(gen_config, key):
                setattr(gen_config, key, value)
            else:
                forward_only_kwargs[key] = value
        # Store original input length for consistent behavior
        original_input_length = input_ids.shape[1]
        
        # If no graphs provided, delegate directly to underlying LLM (pure text path)
        if graphs is None:
            full_outputs = self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config,
                **forward_only_kwargs
            )
            # Return only the newly generated tokens
            return full_outputs[:, original_input_length:]

        # Multimodal path: prepare combined embeddings
        (
            _,
            position_ids,
            attention_mask,
            _,
            inputs_embeds,
            _,
            _,
            _,
            _,  # quantized_tokens (not needed for generation)
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids, None, attention_mask, None, None, graphs
        )

        inputs_embeds = inputs_embeds.to(self.llm.dtype)
        
        #decode back input_embeds for inspection
        # emb_mat = self.llm.get_input_embeddings().weight
        # logits = inputs_embeds[0] @ emb_mat.T
        # decoded = logits.argmax(dim=-1)
        # decoded_text = self.tokenizer.batch_decode(decoded, skip_special_tokens=True)
        # print(decoded_text)

        # Call underlying LLM generate
        full_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            generation_config=gen_config,
            **forward_only_kwargs
        )

        return full_outputs
     
    @torch.inference_mode()
    def generate_graphs(
        self,
        input_ids: torch.Tensor,
        graphs: Batch,
        attention_mask: Optional[torch.Tensor] = None,
        **generation_kwargs
    ):
        """
        Etension of the previous method where the model can use the quantized graph embeddings to generate triplets.
        It will implement a <ASK> and <TELL> token added to the model to generate triplets from the graph embeddings.
        THIS IS NOT TO BE IMPLEMENTED YET, IT IS A FUTURE WORK.
        """
        raise NotImplementedError("The generate_graphs method is not implemented yet. Please implement the generate_graphs method for KG_LM.")
    
    @property
    def default_generation_config(self):
        """
        Returns the default generation configuration for the KG_LM model.
        
        This method retrieves the generation configuration from the underlying LLM model
        and sets sensible defaults if they are missing.
        """
        generation_config = copy.deepcopy(self.llm.generation_config)
        return generation_config


class KG_LM(KG_LMMetaModel, KG_LMMetaForCausalLM, PreTrainedModel):
    """
    Knowledge Graph Language Foundation Model (KG_LM) that combines a pre-trained language model 
    with a knowledge graph encoder.
    This model is designed to process both text and graph data, integrating them into a unified representation.
    
    """
    config_class = KG_LMConfig
    
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    
    def __init__(
        self,
        config: KG_LMConfig, 
    ):
        super(KG_LM, self).__init__(config)
        self.config = config
        
        # Initialize components. This will create self.llm and self.kg_encoder
        self.init_KGLM(config)
        
        # Initialize the tokenizer. It might be overwritten by load_pretrained if a custom one was saved.
        self.tokenizer : PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            config.llm_model_name, 
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.llm.config.pad_token_id

        if SPECIAL_KG_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL_KG_TOKEN]})
            # Important: resize token embeddings in the LLM to account for the new token
            self.llm.resize_token_embeddings(len(self.tokenizer))

        self.special_kg_token = self.tokenizer.convert_tokens_to_ids(SPECIAL_KG_TOKEN)
        
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load a pre-trained KG_LM model from the specified path.
        
        This method serves as a convenient entry point and delegates the core
        loading logic to the `load_pretrained` method implemented in the MetaModel.
        
        Args:
            pretrained_model_name_or_path (str): Path to the pre-trained model directory.
            *model_args: Additional positional arguments for the model.
            **kwargs: Additional keyword arguments for the model.
        
        Returns:
            KG_LM: An instance of the loaded KG_LM model.
        """
        return cls.load_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
    
    def forward(
        self, 
        input_ids : torch.Tensor, 
        graphs: Batch = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass of the KG_LM model.
        """
        
        RVQ_loss = None
        quantized_tokens = None
        quantized_indices = None
        if inputs_embeds is None:
            logging.debug("Preparing inputs and labels for multimodal processing...")
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                RVQ_loss,
                indices,
                quantized_tokens
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                graphs
            )
            
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(dtype=self.llm.dtype)

        logging.debug(f"Forward pass to LLM")
        out = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        logging.debug(f"Output from LLM: {out.keys()}")
        
        # Add decoder reconstruction loss if decoder is enabled and we have quantized tokens
        decoder_loss = None
        if (self.get_kg_decoder() is not None and 
            quantized_tokens is not None and 
            graphs is not None):
            
            # Run decoder for self-supervised learning
            # In training mode, we can create node pairs and relations from the original graphs
            # For now, we use a simple reconstruction objective
            decoder_results = self.decode_graphs(
                quantized_tokens=quantized_tokens,
                original_graphs=graphs,
                mode="both"  # Both edge and object prediction
            )
            
            if decoder_results is not None and "total_loss" in decoder_results:
                decoder_loss = decoder_results["total_loss"]
                logging.debug(f"Decoder loss: {decoder_loss.item()}")
        
        # If RVQ_loss is not None, add it to the model's loss for training
        if return_dict:
            out["RVQ_loss"] = RVQ_loss
            out["RVQ_indices"] = indices
            out["decoder_loss"] = decoder_loss
        else:
            total_loss = out[0] if RVQ_loss is None else out[0] + RVQ_loss
            if decoder_loss is not None:
                total_loss = total_loss + decoder_loss if total_loss is not None else decoder_loss
            out = (total_loss,) + out[1:]
            
        return out
        
    
AutoConfig.register(KG_LMConfig.model_type, KG_LMConfig)
AutoModel.register(KG_LMConfig, KG_LM)
