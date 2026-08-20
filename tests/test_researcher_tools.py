"""Unit tests for researcher-centric modules: 3D Sterics, Stoichiometry, Precedent, and Purification."""

import pytest
from retro_engine.critic.steric_3d import Steric3DAnalyzer
from retro_engine.critic.stoichiometry import ReactionStoichiometryCalculator
from retro_engine.critic.precedent_matcher import PrecedentMatcher
from retro_engine.critic.byproduct_predictor import ByproductPurificationPredictor


def test_steric_3d_analyzer():
    # Paracetamol molecule
    paracetamol_smi = "CC(=O)Nc1ccc(O)cc1"
    res = Steric3DAnalyzer.evaluate_molecule_sterics(paracetamol_smi, "C-N Coupling")
    assert "steric_burial_score" in res
    assert "accessibility" in res
    assert 0.0 <= res["steric_burial_score"] <= 100.0


def test_stoichiometry_calculator():
    # Amide coupling to synthesize 2.0g of Paracetamol
    product = "CC(=O)Nc1ccc(O)cc1"
    reactants = ["Nc1ccc(O)cc1", "CC(=O)O"]
    reagents = ["HATU", "DIPEA"]
    solvents = ["DMF"]
    
    table_data = ReactionStoichiometryCalculator.calculate_step_stoichiometry(
        product_smiles=product,
        reactants_smiles=reactants,
        reagents=reagents,
        solvents=solvents,
        target_product_mass_g=2.0,
        step_yield=0.90,
    )
    
    assert table_data["target_scale_g"] == 2.0
    assert table_data["target_product_mmol"] > 0
    assert len(table_data["stoichiometry_table"]) >= 2
    
    # Check limiting reactant (first reactant) has eq = 1.00
    limiting = table_data["stoichiometry_table"][0]
    assert limiting["equivalents"] == 1.00
    assert limiting["mass_g"] > 0.0


def test_precedent_matcher():
    # Query an aromatic amide
    product = "CC(=O)Nc1ccc(O)cc1"
    match = PrecedentMatcher.find_nearest_precedent(product, "C-N Coupling")
    assert "has_precedent" in match
    assert match["has_precedent"] is True
    assert "patent_id" in match
    assert match["similarity_pct"] > 0.0


def test_byproduct_purification_predictor():
    product = "CC(=O)Nc1ccc(O)cc1"
    purif = ByproductPurificationPredictor.forecast_purification("C-N Coupling", product)
    assert "recommended_eluent" in purif
    assert "tlc_visualization" in purif
    assert len(purif["expected_side_products"]) > 0
