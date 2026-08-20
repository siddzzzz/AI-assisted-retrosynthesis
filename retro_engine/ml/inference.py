"""Neural Retrosynthesis Policy Inference Engine.
Applies the trained Physics-Informed Graph Transformer to rank candidate retrosynthetic disconnections.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn.functional as F

from .physics_encoder import MolecularPhysicsFeatureExtractor, MoleculeGraphData
from .models import PhysicsInformedGraphTransformer
from ..chem.reaction_rules import ReactionRuleLibrary


class NeuralRetrosynthesisPolicy:
    """Evaluates molecules using the Physics-Informed Multi-Task Graph Transformer."""

    def __init__(self, model_weights_path: Optional[str] = None, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.rule_lib = ReactionRuleLibrary()
        self.rules = self.rule_lib.get_all_rules()
        self.num_classes = len(self.rules)

        self.model = PhysicsInformedGraphTransformer(
            num_classes=self.num_classes,
            hidden_dim=128,
            num_layers=3,
            num_heads=4,
        ).to(self.device)

        if model_weights_path and os.path.exists(model_weights_path):
            try:
                ckpt = torch.load(model_weights_path, map_location=self.device)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["model_state_dict"])
                else:
                    self.model.load_state_dict(ckpt)
                print(f"Loaded trained Physics-Informed Graph Transformer from {model_weights_path}")
            except Exception as e:
                print(f"Warning: Could not load checkpoint ({e}), using initialized neural weights.")

        self.model.eval()

    def predict_disconnections(
        self, smiles: str, top_k: int = 5
    ) -> Dict[str, Any]:
        """Predict top-k retrosynthetic transformations, kinetic barrier, and solvent dielectric."""
        graph = MolecularPhysicsFeatureExtractor.extract_features(smiles)
        if graph is None:
            return {
                "success": False,
                "error": "Failed to extract physics features",
                "top_predictions": [],
                "predicted_barrier_kj": 65.0,
                "predicted_dielectric": 25.0,
            }

        # Move graph tensors to device
        graph.node_features = graph.node_features.to(self.device)
        graph.edge_index = graph.edge_index.to(self.device)
        graph.edge_features = graph.edge_features.to(self.device)
        graph.spatial_distances = graph.spatial_distances.to(self.device)
        graph.global_features = graph.global_features.to(self.device)

        with torch.no_grad():
            outputs = self.model(graph)
            logits = outputs["policy_logits"]
            probs = F.softmax(logits, dim=-1).squeeze(0)
            barrier_kj = float(outputs["predicted_barrier_kj"].item())
            dielectric = float(outputs["predicted_dielectric"].item())

        # Extract top-k rules
        top_probs, top_indices = torch.topk(probs, k=min(top_k, self.num_classes))
        
        predictions = []
        for p, idx in zip(top_probs.tolist(), top_indices.tolist()):
            if idx < len(self.rules):
                rule = self.rules[idx]
                predictions.append({
                    "rule_id": rule.rule_id,
                    "reaction_name": rule.name,
                    "reaction_class": rule.reaction_class,
                    "neural_probability": round(p, 4),
                    "confidence_pct": round(p * 100.0, 1),
                    "explanation": rule.explanation,
                })

        return {
            "success": True,
            "smiles": smiles,
            "top_predictions": predictions,
            "predicted_barrier_kj": round(barrier_kj, 1),
            "predicted_dielectric": round(dielectric, 1),
            "device": str(self.device),
        }
