"""Tests for Multi-Step Retrosynthesis Search and Route Optimization."""

import pytest
from retro_engine.search.mcts import RetrosynthesisSearchTree
from retro_engine.pipeline import RetrosynthesisEngine


def test_search_tree_paracetamol():
    engine = RetrosynthesisEngine()
    paracetamol_smi = "CC(=O)Nc1ccc(O)cc1"
    
    result = engine.plan_synthesis(paracetamol_smi, max_depth=5, max_routes=4)
    assert result["success"] is True
    assert result["total_plans_found"] > 0
    
    top_plan = result["plans"][0]
    assert top_plan["is_solved"] is True
    assert len(top_plan["steps"]) >= 1
    assert len(top_plan["starting_materials"]) >= 1


def test_search_tree_aspirin():
    engine = RetrosynthesisEngine()
    aspirin_smi = "CC(=O)Oc1ccccc1C(=O)O"
    
    result = engine.plan_synthesis(aspirin_smi, max_depth=4, max_routes=4)
    assert result["success"] is True
    assert result["total_plans_found"] > 0
    
    top_plan = result["plans"][0]
    assert top_plan["is_solved"] is True


def test_generate_sop_protocol():
    engine = RetrosynthesisEngine()
    result = engine.plan_synthesis("CC(=O)Nc1ccc(O)cc1")
    assert result["total_plans_found"] > 0
    
    sop = engine.generate_laboratory_sop(result["plans"][0])
    assert "# Laboratory Standard Operating Procedure" in sop
    assert "Required Raw Material Feedstocks" in sop
    assert "Step-by-Step Reaction Protocol" in sop
