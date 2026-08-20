"""Tests for chemoinformatics utilities, molecule validation, properties, and SVG rendering."""

import pytest
from retro_engine.chem.mol_utils import (
    canonicalize_smiles,
    get_mol_from_smiles,
    calculate_mol_properties,
    render_mol_svg,
    identify_functional_groups,
    calculate_atom_economy,
    smiles_to_inchikey,
)


def test_canonicalize_smiles():
    # Paracetamol in alternate SMILES representations
    smi1 = "CC(=O)Nc1ccc(O)cc1"
    smi2 = "Oc1ccc(NC(=O)C)cc1"
    can1 = canonicalize_smiles(smi1)
    can2 = canonicalize_smiles(smi2)
    assert can1 is not None
    assert can1 == can2


def test_calculate_mol_properties():
    # Aspirin: CC(=O)Oc1ccccc1C(=O)O
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    props = calculate_mol_properties(aspirin)
    assert props["valid"] is True
    assert props["formula"] == "C9H8O4"
    assert 179.0 < props["molecular_weight"] < 181.0
    assert "carboxylic_acid" in props["functional_groups"]
    assert "ester" in props["functional_groups"]


def test_identify_functional_groups():
    # Compound with amine and alcohol
    smi = "NCC(O)c1ccccc1"
    groups = identify_functional_groups(smi)
    assert "primary_amine" in groups
    assert "alcohol" in groups


def test_calculate_atom_economy():
    # Phenol + Acetic anhydride -> Paracetamol (or phenyl acetate)
    reactants = ["Nc1ccc(O)cc1", "CC(=O)OC(=O)C"]
    product = "CC(=O)Nc1ccc(O)cc1"
    ae = calculate_atom_economy(reactants, product)
    assert 65.0 < ae < 75.0  # Byproduct is acetic acid (MW 60)


def test_render_mol_svg():
    smi = "CC(=O)Nc1ccc(O)cc1"
    svg = render_mol_svg(smi, width=200, height=150, dark_mode=True)
    assert "<svg" in svg
    assert "</svg>" in svg
