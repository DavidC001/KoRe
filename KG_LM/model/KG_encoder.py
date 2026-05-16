"""
pytorch model that when received a Batch of graphs,
returns for each the residual vector quantized representsion for each graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GraphNorm
import logging

from vector_quantize_pytorch import VectorQuantize

# graph pooling
from torch_geometric.nn import global_mean_pool

from torch_geometric.data import Batch

import torch
import torch.nn as nn
import torch.nn.functional as F

from KG_LM.model.qformer import QFormerPool
from torch_geometric.utils import to_dense_batch

class DirectionalVQ(nn.Module):
    """
    Multi-step 'directional' VQ built from base VectorQuantize:
    - use_cosine_sim=True => codes + inputs are L2-normalized internally
    - per step: pick code, STE token, subtract PROJECTION only (OMP-like)
    - returns tokens [B, Q, D], indices [B, Q], and total loss
    """
    def __init__(
        self,
        dim: int,
        num_quantizers: int = 3,
        codebook_size: int = 512,
        *,
        shared_codebook: bool = True,
        beta_commit: float = 0.25,
        dead_codebook_threshold: float = 0.5,
        vq_kwargs: dict | None = None
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.beta_commit = beta_commit

        vq_kwargs = dict(vq_kwargs or {})
        # we use our own commit; disable internal one
        vq_kwargs.setdefault("commitment_weight", 0.0)
        vq_kwargs.setdefault("use_cosine_sim", True)
        vq_kwargs.setdefault("threshold_ema_dead_code", dead_codebook_threshold)
        vq_kwargs.setdefault("dim", dim)
        vq_kwargs.setdefault("codebook_size", codebook_size)

        if shared_codebook:
            self.vq = VectorQuantize(**vq_kwargs)
            self.vqs = None
        else:
            self.vq = None
            self.vqs = nn.ModuleList([
                VectorQuantize(**vq_kwargs)
                for _ in range(num_quantizers)
            ])

    def _get_vq(self, i: int) -> VectorQuantize:
        return self.vq if self.vq is not None else self.vqs[i]

    def forward(self, x: torch.Tensor):
        """
        x: [B, D]
        returns:
          tokens:  [B, Q, D]  (STE outputs at each depth)
          indices: [B, Q]     (code indices per depth)
          loss:    scalar     (cosine commit + residual leftover)
        """
        assert x.ndim == 2, "expecting [B, D] after graph pooling"

        # quantizer expects float32 internally
        residual = x.float()

        tokens = []
        indices = []
        total_loss = residual.new_tensor(0.)

        for i in range(self.num_quantizers):
            vq = self._get_vq(i)
            vq.float()

            # quantize the current residual (VectorQuantize normalizes internally for cosine)
            quantized, embed_ind, _ = vq(residual)   # quantized ~ unit-norm (cosine codebook)
            indices.append(embed_ind)                # [B,]

            # cosine-based commitment (directional), numerically safe
            norm_res = F.normalize(residual, dim=-1)
            norm_code = F.normalize(quantized.detach(), dim=-1)
            cos_sim = (norm_res * norm_code).sum(-1).clamp(-1.0, 1.0)     # [B]
            commit_loss = (1.0 - cos_sim).mean()
            total_loss = total_loss + self.beta_commit * commit_loss

            # STE token to feed downstream (gradient to encoder through residual)
            token = residual + (quantized - residual).detach()
            tokens.append(token)  # [B, D]

            # OMP-style residual update: subtract only the projection onto the chosen code direction
            alpha = (residual * norm_code).sum(-1, keepdim=True)          # [B, 1]
            residual = residual - alpha * norm_code                       # stays orthogonal to code

        # penalize any leftover residual energy
        total_loss = total_loss + residual.pow(2).mean()

        tokens = torch.stack(tokens, dim=1)            # [B, Q, D]
        indices = torch.stack(indices, dim=1)          # [B, Q]
        return tokens, indices, total_loss


class KGEncoder(nn.Module):
    def __init__(
        self, 
        node_embedding_dim, edge_embedding_dim, final_embedding_dim, std_tokens,
        dropout=0.2, num_heads=1, gat_layers=3, graph_pooling=True, q_former=False,
        num_quantizers=3, codebook_size=512, codebook_dim=0, shared_codebook=False,
        beta_commit=0.25, dead_codebook_threshold=0.5, bounding_tokens=False
    ):
        """
        Initializes the KGEncoder model.
        
        Args:
            node_embedding_dim (int): Dimension of the node embeddings.
            edge_embedding_dim (int): Dimension of the edge embeddings.
            final_embedding_dim (int): Dimension of the final output embeddings.
            std_tokens (float): Standard deviation of the token embeddings.
            dropout (float): Dropout rate for the model.
            num_heads (int): Number of attention heads in GATv2Conv.
            gat_layers (int): Number of GATv2Conv layers.
            graph_pooling (bool): Whether to apply global mean pooling to the graph representations.
            q_former (bool): Whether to use a Q-Former architecture to pool graph representations.
            num_quantizers (int): Number of quantizers for residual vector quantization.
            codebook_size (int): Size of the codebook for vector quantization.
            codebook_dim (int): Dimension of the downsampled codebook. If 0, no downsampling is applied.
            shared_codebook (bool): Whether to use a shared codebook across quantizers.
            beta_commit (float): Commitment loss weight for the vector quantization.
            dead_codebook_threshold (float): Threshold for dead codebook entries.
            bounding_tokens (bool): Whether to use special bounding tokens around the KG embeddings in the input sequence.
        """
        super(KGEncoder, self).__init__()
        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        
        self.convs = nn.ModuleList([
            TransformerConv(
                in_channels=node_embedding_dim,
                out_channels=node_embedding_dim // num_heads,
                heads=num_heads,
                edge_dim=edge_embedding_dim,
                dropout=dropout,
                concat=True
            )
            for _ in range(gat_layers)
        ])
        self.norms = nn.ModuleList([GraphNorm(node_embedding_dim) for _ in range(gat_layers)])

        self.num_heads = num_heads
        
        self.dropout = nn.Dropout(dropout)
        self.num_quantizers = num_quantizers

        self.codebook_dim = codebook_dim
        if codebook_dim > 0:
            self.downsample = nn.Linear(node_embedding_dim, codebook_dim)
            self.downsample_norm = nn.LayerNorm(codebook_dim)
            self.upsample   = nn.Linear(codebook_dim, node_embedding_dim)
            self.upsample_norm = nn.LayerNorm(node_embedding_dim)

        vq_dim = node_embedding_dim if codebook_dim == 0 else codebook_dim
        self.dir_vq = DirectionalVQ(
            dim=vq_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            shared_codebook=shared_codebook,
            beta_commit=beta_commit,
            dead_codebook_threshold=dead_codebook_threshold,
            vq_kwargs=dict(
                # optional arguments:
                # orthogonal_reg_weight=1e-4,
                # codebook_diversity_loss_weight=1e-4,
                # kmeans_init=True,
                # ...
            )
        )
        
        self.kg_bias = nn.Parameter(torch.zeros(final_embedding_dim))
        self.text_embs_std = std_tokens

        # output projection layer with spectral normalization
        self.skip_to_final = nn.Linear(node_embedding_dim, final_embedding_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(node_embedding_dim, 4 * node_embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * node_embedding_dim, final_embedding_dim),
        )
        self.output_norm = nn.LayerNorm(final_embedding_dim)
        
        self.graph_pooling = graph_pooling
        self.q_former = q_former
        
        assert not (self.graph_pooling and self.q_former), "Cannot use both graph_pooling and q_former simultaneously."
        if self.q_former:
            logging.info("KGEncoder: using Q-Former for graph pooling")
            self.q_former_pool = QFormerPool(channels=node_embedding_dim, num_heads=num_heads, dropout=dropout)
        
        self.bounding_tokens = bounding_tokens
        if self.bounding_tokens:
            logging.info("KGEncoder: using bounding tokens around KG embeddings")
            self.start_token = nn.Parameter(torch.randn(1, 1, final_embedding_dim) * self.text_embs_std)
            self.end_token = nn.Parameter(torch.randn(1, 1, final_embedding_dim) * self.text_embs_std)
        else:
            logging.info("KGEncoder: NOT using bounding tokens around KG embeddings")
        
    def forward(self, graphs: Batch):
        """
        Forward pass for the KGEncoder.
        
        Args:
            graphs (Batch): A batch of graphs from PyTorch Geometric.
        
        Returns:
            torch.Tensor: The quantized representations of the graphs.
        """
        logging.debug("Forward into KG_encoder")
        x, edge_index, edge_attr = graphs.x, graphs.edge_index, graphs.edge_attr
        
        logging.debug("Got graph data")

        # Apply graph convolution layers
        for i, conv in enumerate(self.convs):
            device = self.convs[i].lin_query.weight.device
            dtype = self.convs[i].lin_query.weight.dtype
            x = x.to(device, dtype=dtype)
            edge_attr = edge_attr.to(device, dtype=dtype)
            edge_index = edge_index.to(device)

            x_in = x
            x = conv(x, edge_index, edge_attr)
            x = self.norms[i](x + x_in, graphs.batch.to(x.device))

        if self.graph_pooling:
            x = global_mean_pool(x, graphs.batch.to(x.device))
        elif self.q_former:
            x_padded, mask = to_dense_batch(x, graphs.batch)
            x = self.q_former_pool(x_padded, mask)  # (batch_size, out_channels)
        else:
            # Find the first occurrence of each batch index
            num_graphs = graphs.batch.max().item() + 1
            
            # Find first node index for each graph in the batch (GPU-optimized, no loops)
            first_indices = torch.zeros(num_graphs, dtype=torch.long, device=graphs.batch.device)
            mask = graphs.batch[:-1] != graphs.batch[1:] # Find where the batch index changes
            first_indices[1:] = torch.where(mask)[0] + 1 # Shift indices by 1 to account for the first node in each graph
            
            x = x[first_indices]
        
        logging.debug(f"Graph embeddings shape after Graph Conv: {x.shape}") 

        if self.codebook_dim > 0:
            x = self.downsample(x)
            x = self.downsample_norm(x)
        
        # quantize directionally
        tokens, indices, loss = self.dir_vq(x)     # tokens: [B, Q, vq_dim]

        dtype = self.kg_bias.dtype
        tokens = tokens.to(dtype=dtype)
        
        if self.codebook_dim > 0:
            tokens = self.upsample(tokens)
            tokens = self.upsample_norm(tokens)
        
        tokens = self.dropout(tokens)                             # [B, Q, node_D]
        proj = self.output_projection(tokens)                     # [B, Q, final_D]
        skip = self.skip_to_final(tokens)                         # [B, Q, final_D]
        tokens = self.output_norm(proj + skip)                    # residual in final space

        # token standardization
        g_mu = tokens.mean(dim=(-2,-1), keepdim=True)
        g_std = tokens.std(dim=(-2,-1), keepdim=True).clamp_min(1e-6)
        tokens = (tokens - g_mu) / g_std
        # match text emb std
        tokens = tokens * self.text_embs_std + self.kg_bias
        
        # add bounding tokens if needed
        if self.bounding_tokens:
            B, Q, D = tokens.shape
            start_tokens = self.start_token.expand(B, -1, -1)  # [B, 1, D]
            end_tokens = self.end_token.expand(B, -1, -1)      # [B, 1, D]
            tokens = torch.cat([start_tokens, tokens, end_tokens], dim=1)  # [B, Q+2, D]

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"Output shape after adapter and normalization: {x.shape}")
            logging.debug(f"Quantized output shape: {tokens.shape}, Indices shape: {indices.shape}, Loss: {loss.item()}")
        
        # Return the quantized representations
        return tokens, indices, loss