"""Unit tests for the Physics-Informed Graph Transformer (PI-GT) ML architecture."""

import pytest
import torch
from retro_engine.ml.physics_encoder import MolecularPhysicsFeatureExtractor
from retro_engine.ml.models import PhysicsInformedGraphTransformer
from retro_engine.ml.inference import NeuralRetrosynthesisPolicy


def test_molecular_physics_feature_extractor():
    smi = "CC(=O)Nc1ccc(O)cc1"  # Paracetamol
    graph = MolecularPhysicsFeatureExtractor.extract_features(smi)
    
    assert graph is not None
    assert graph.node_features.dim() == 2
    assert graph.node_features.size(1) == 64  # 64-dim physics node features
    assert graph.edge_features.size(1) == 16  # 16-dim bond features
    assert graph.spatial_distances.size(0) == graph.num_atoms
    assert graph.spatial_distances.size(1) == graph.num_atoms
    assert graph.global_features.size(0) == 16


def test_physics_informed_graph_transformer_forward():
    smi = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    graph = MolecularPhysicsFeatureExtractor.extract_features(smi)
    assert graph is not None

    model = PhysicsInformedGraphTransformer(
        num_classes=25,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
    )
    model.eval()

    with torch.no_grad():
        outputs = model(graph)
        assert "policy_logits" in outputs
        assert "predicted_barrier_kj" in outputs
        assert "predicted_dielectric" in outputs
        assert "latent_embedding" in outputs
        
        assert outputs["policy_logits"].size(1) == 25
        assert outputs["predicted_barrier_kj"].size() == (1, 1)
        assert outputs["predicted_dielectric"].size() == (1, 1)


def test_neural_policy_inference():
    policy = NeuralRetrosynthesisPolicy(device="cpu")
    pred = policy.predict_disconnections("CC(=O)Nc1ccc(O)cc1", top_k=3)
    
    assert pred["success"] is True
    assert len(pred["top_predictions"]) == 3
    assert "predicted_barrier_kj" in pred
    assert "predicted_dielectric" in pred
    assert pred["predicted_barrier_kj"] > 0
