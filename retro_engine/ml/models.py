"""Physics-Informed Multi-Task Graph Transformer (PI-GT).
A novel neural architecture integrating 3D spatial geometry, Gasteiger electrostatic charges,
and multi-task prediction for chemical retrosynthesis.
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from .physics_encoder import MoleculeGraphData


class SpatialGraphTransformerLayer(nn.Module):
    """Transformer self-attention layer modulated by 3D spatial distances and bond embeddings."""

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, edge_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # 3D Distance RBF projection
        self.spatial_proj = nn.Sequential(
            nn.Linear(1, num_heads),
            nn.SiLU(),
            nn.Linear(num_heads, num_heads)
        )

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, dist_matrix: torch.Tensor) -> torch.Tensor:
        """
        h: [N_atoms, hidden_dim]
        dist_matrix: [N_atoms, N_atoms]
        """
        num_atoms = h.size(0)
        h_norm = self.norm1(h)

        # Multi-Head Projections: [N, num_heads, head_dim]
        Q = self.q_proj(h_norm).view(num_atoms, self.num_heads, self.head_dim).transpose(0, 1)
        K = self.k_proj(h_norm).view(num_atoms, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.v_proj(h_norm).view(num_atoms, self.num_heads, self.head_dim).transpose(0, 1)

        # Scaled Dot-Product: [num_heads, N, N]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 3D Spatial Distance Bias: [N, N, 1] -> [N, N, num_heads] -> [num_heads, N, N]
        dist_expanded = dist_matrix.unsqueeze(-1)
        spatial_bias = self.spatial_proj(dist_expanded).permute(2, 0, 1)
        scores = scores + spatial_bias

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Aggregate context: [num_heads, N, head_dim] -> [N, hidden_dim]
        context = torch.matmul(attn_weights, V)
        context = context.transpose(0, 1).contiguous().view(num_atoms, self.hidden_dim)
        
        # Residual + FFN
        h = h + self.out_proj(context)
        h = h + self.ffn(self.norm2(h))
        return h


class PhysicsInformedGraphTransformer(nn.Module):
    """Deep Multi-Task Neural Network for Retrosynthesis, Kinetic Barriers, and Solvent Polarity."""

    def __init__(
        self,
        node_in_dim: int = 64,
        edge_in_dim: int = 16,
        global_in_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        num_classes: int = 30,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.node_embed = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.layers = nn.ModuleList([
            SpatialGraphTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                edge_dim=edge_in_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Global readout pooling (Attention Readout)
        self.pool_gate = nn.Linear(hidden_dim, 1)
        self.global_embed = nn.Linear(global_in_dim, hidden_dim)

        # Fused Latent Representation: hidden_dim (from atoms) + hidden_dim (from globals) = 256
        fused_dim = hidden_dim * 2

        # -------------------------------------------------------------
        # MULTI-TASK PREDICTION HEADS
        # -------------------------------------------------------------
        # 1. Retrosynthetic Disconnection Policy Head
        self.policy_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        # 2. Kinetic Activation Barrier Head (Estimated ΔG‡ in kJ/mol, e.g. 40 to 140 kJ/mol)
        self.activation_barrier_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 3. Optimal Solvent Dielectric Constant (ε) Head (e.g. 2.0 to 80.0)
        self.solvent_dielectric_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, graph: MoleculeGraphData) -> Dict[str, torch.Tensor]:
        """
        Forward pass taking physics-augmented molecular graph.
        Returns:
            policy_logits: [1, num_classes] (Logits for top bond disconnections)
            predicted_barrier: [1, 1] (Estimated activation energy ΔG‡ in kJ/mol)
            predicted_dielectric: [1, 1] (Estimated optimal solvent dielectric constant ε)
            latent_embedding: [1, fused_dim] (Continuous vector representation)
        """
        # Node Embedding: [N, hidden_dim]
        h = self.node_embed(graph.node_features)

        # Stacked Spatial Graph Transformer Layers
        for layer in self.layers:
            h = layer(h, graph.spatial_distances)

        # Attention Readout Pooling: [1, hidden_dim]
        gate_weights = F.softmax(self.pool_gate(h), dim=0)
        h_graph = torch.sum(gate_weights * h, dim=0, keepdim=True)

        # Global features embedding: [1, hidden_dim]
        h_glob = self.global_embed(graph.global_features.unsqueeze(0))

        # Fused Latent Representation [1, fused_dim]
        z = torch.cat([h_graph, h_glob], dim=-1)

        # Multi-Task Predictions
        policy_logits = self.policy_head(z)
        barrier = F.relu(self.activation_barrier_head(z)) + 40.0  # Baseline ~40 kJ/mol
        dielectric = F.softplus(self.solvent_dielectric_head(z)) + 2.0  # Baseline ~2.0 ε

        return {
            "policy_logits": policy_logits,
            "predicted_barrier_kj": barrier,
            "predicted_dielectric": dielectric,
            "latent_embedding": z,
        }
