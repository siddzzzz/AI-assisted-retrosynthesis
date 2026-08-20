"""Tests for System 2 Critic, Protecting Groups, Solvents, and Green Metrics."""

import pytest
from retro_engine.chem.reaction_rules import ReactionRuleLibrary
from retro_engine.critic.chemoselectivity import ChemoselectivityAnalyzer
from retro_engine.critic.protecting_groups import ProtectingGroupPlanner
from retro_engine.critic.solvent_matrix import SolventConditionPredictor
from retro_engine.critic.green_metrics import GreenChemistryEvaluator


def test_chemoselectivity_grignard_alcohol_clash():
    lib = ReactionRuleLibrary()
    grignard_rule = next(r for r in lib.get_all_rules() if r.rule_id == "RXN_GRIGNARD_ADDITION")

    # Reactants with free alcohol: Hydroxyaldehyde + Grignard
    reactants = ["O=Cc1ccccc1O", "C[MgBr]"]
    product = "CC(O)c1ccccc1O"
    conflict = ChemoselectivityAnalyzer.evaluate_step(reactants, product, grignard_rule)
    
    assert conflict.has_conflict is True
    assert "alcohol" in conflict.conflicting_groups or "phenol" in conflict.conflicting_groups
    assert conflict.severity == "critical"
    assert conflict.needs_protection is True
    assert conflict.suggested_protecting_group == "tbs"


def test_protecting_group_strategy():
    report = ProtectingGroupPlanner.generate_strategy_report("alcohol", "tbs")
    assert report["needs_protection"] is True
    assert "TBS" in report["protecting_group"]
    assert "TBS-Cl" in report["installation"]["reagents"]
    assert "TBAF" in report["deprotection"]["reagents"]


def test_solvent_condition_predictor():
    lib = ReactionRuleLibrary()
    sonogashira = next(r for r in lib.get_all_rules() if r.rule_id == "RXN_SONOGASHIRA_01")
    conditions = SolventConditionPredictor.predict_conditions(sonogashira)
    
    assert "strictly_anhydrous" in sonogashira.moisture_sensitivity
    assert "Inert" in conditions.atmosphere
    assert conditions.aqueous_compatibility_score == 0.0


def test_green_metrics_route_eval():
    steps = [
        {
            "step_yield": 0.90,
            "atom_economy": 88.0,
            "reactants_smiles": ["Nc1ccc(O)cc1", "CC(=O)O"],
            "product_smiles": "CC(=O)Nc1ccc(O)cc1",
            "reagents": ["HATU", "DIPEA"],
            "solvents": ["DMF"],
        }
    ]
    metrics = GreenChemistryEvaluator.evaluate_route(steps)
    assert metrics.total_steps == 1
    assert metrics.cumulative_yield == 90.0
    assert metrics.green_chemistry_score > 50.0
