"""Tests for reaction rules library and single-step retrosynthetic disconnections."""

import pytest
from rdkit import Chem
from retro_engine.chem.reaction_rules import ReactionRuleLibrary, apply_retro_rule
from retro_engine.chem.mol_utils import canonicalize_smiles


def test_reaction_rule_library_init():
    lib = ReactionRuleLibrary()
    rules = lib.get_all_rules()
    assert len(rules) >= 10
    rule_ids = [r.rule_id for r in rules]
    assert "RXN_AMIDE_01" in rule_ids
    assert "RXN_SUZUKI_01" in rule_ids
    assert "RXN_NITRO_REDUCTION" in rule_ids


def test_amide_disconnection():
    lib = ReactionRuleLibrary()
    rule = next(r for r in lib.get_all_rules() if r.rule_id == "RXN_AMIDE_01")
    
    # Paracetamol: CC(=O)Nc1ccc(O)cc1
    mol = Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1")
    disconnections = apply_retro_rule(mol, rule)
    assert len(disconnections) > 0
    
    precursor_smis, matched_rule = disconnections[0]
    can_precursors = [canonicalize_smiles(s) for s in precursor_smis]
    assert canonicalize_smiles("Nc1ccc(O)cc1") in can_precursors
    assert canonicalize_smiles("CC(=O)O") in can_precursors


def test_suzuki_disconnection():
    lib = ReactionRuleLibrary()
    rule = next(r for r in lib.get_all_rules() if r.rule_id == "RXN_SUZUKI_01")
    
    # 4-Methylbiphenyl: Cc1ccc(-c2ccccc2)cc1
    mol = Chem.MolFromSmiles("Cc1ccc(-c2ccccc2)cc1")
    disconnections = apply_retro_rule(mol, rule)
    assert len(disconnections) > 0
