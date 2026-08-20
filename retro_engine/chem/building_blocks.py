"""Fundamental chemical building blocks and commodity raw materials registry."""

import json
import os
from typing import Dict, Optional, Set, Any, List
from .mol_utils import canonicalize_smiles, smiles_to_inchikey, get_mol_from_smiles
from rdkit.Chem import Descriptors


class BuildingBlockCatalog:
    """Registry of basic chemical feedstocks and commodity starting materials."""

    def __init__(self, data_file: Optional[str] = None):
        self.catalog_by_inchikey: Dict[str, Dict[str, Any]] = {}
        self.catalog_by_smiles: Dict[str, Dict[str, Any]] = {}
        
        if data_file and os.path.exists(data_file):
            self.load_from_json(data_file)
        else:
            self._load_core_stock()

    def add_compound(
        self,
        smiles: str,
        name: str,
        category: str = "Commodity Chemical",
    ):
        """Add a verified basic chemical starting material to the catalog."""
        can_smi = canonicalize_smiles(smiles)
        if not can_smi:
            return

        inchikey = smiles_to_inchikey(can_smi)
        entry = {
            "smiles": can_smi,
            "name": name,
            "category": category,
            "is_fundamental_feedstock": True,
        }

        self.catalog_by_smiles[can_smi] = entry
        if inchikey:
            self.catalog_by_inchikey[inchikey] = entry

    def is_fundamental_starting_material(self, smiles: str) -> bool:
        """Check if a compound is a fundamental petrochemical or commodity chemical feedstock."""
        can_smi = canonicalize_smiles(smiles)
        if not can_smi:
            return False

        # 1. Direct match in commodity feedstocks
        if can_smi in self.catalog_by_smiles:
            return True

        inchikey = smiles_to_inchikey(can_smi)
        if inchikey and inchikey in self.catalog_by_inchikey:
            return True

        # 2. Simple Molecule Heuristic: Small commodity molecules (<= 6 heavy atoms)
        mol = get_mol_from_smiles(can_smi)
        if mol is not None:
            num_heavy = mol.GetNumHeavyAtoms()
            if num_heavy <= 4:
                return True

        return False

    def get_compound_info(self, smiles: str) -> Optional[Dict[str, Any]]:
        """Retrieve feedstock metadata for a compound."""
        can_smi = canonicalize_smiles(smiles)
        if not can_smi:
            return None

        if can_smi in self.catalog_by_smiles:
            return self.catalog_by_smiles[can_smi]

        inchikey = smiles_to_inchikey(can_smi)
        if inchikey and inchikey in self.catalog_by_inchikey:
            return self.catalog_by_inchikey[inchikey]

        if self.is_fundamental_starting_material(can_smi):
            return {
                "smiles": can_smi,
                "name": "Commodity Feedstock Chemical",
                "category": "Raw Material",
                "is_fundamental_feedstock": True,
            }

        return None

    def _load_core_stock(self):
        """Pre-populate fundamental organic and petrochemical starting feedstocks."""
        core_feedstocks = [
            # Basic Aromatic Feedstocks
            ("c1ccccc1", "Benzene", "Aromatic Feedstock"),
            ("Cc1ccccc1", "Toluene", "Aromatic Feedstock"),
            ("Oc1ccccc1", "Phenol", "Commodity Phenol"),
            ("Nc1ccccc1", "Aniline", "Aromatic Amine Feedstock"),
            ("O=[N+]([O-])c1ccccc1", "Nitrobenzene", "Nitroarene Feedstock"),
            ("Brc1ccccc1", "Bromobenzene", "Halobenzene Feedstock"),
            ("Clc1ccccc1", "Chlorobenzene", "Halobenzene Feedstock"),
            ("CC(C)Cc1ccccc1", "Isobutylbenzene", "Alkylbenzene Feedstock"),
            ("c1ccncc1", "Pyridine", "Heteroaromatic Feedstock"),

            # Aliphatic & Carbonyl Feedstocks
            ("CC(=O)O", "Acetic acid", "Carboxylic Acid Feedstock"),
            ("CC(=O)OC(=O)C", "Acetic anhydride", "Acylating Reagent"),
            ("CC(=O)Cl", "Acetyl chloride", "Acylating Reagent"),
            ("CC(=O)C", "Acetone", "Ketone Feedstock"),
            ("O=Cc1ccccc1", "Benzaldehyde", "Aldehyde Feedstock"),
            ("O=C(O)c1ccccc1", "Benzoic acid", "Carboxylic Acid Feedstock"),
            ("CCO", "Ethanol", "Alcohol Feedstock"),
            ("CO", "Methanol", "Alcohol Feedstock"),
            ("CC(C)O", "Isopropanol", "Alcohol Feedstock"),
            ("C#C", "Acetylene", "Alkyne Feedstock"),
            ("C=C", "Ethylene", "Alkene Feedstock"),
            ("CCN", "Ethylamine", "Alkylamine Feedstock"),
            ("CN", "Methylamine", "Alkylamine Feedstock"),
            ("CCN(CC)CC", "Triethylamine", "Base Reagent"),
            ("ClCC", "Chloroethane", "Alkyl Halide Feedstock"),
            ("BrCC", "Bromoethane", "Alkyl Halide Feedstock"),
            ("CCI", "Iodomethane", "Alkylating Agent"),
            ("C(C)(C)(C)Cl", "tert-Butyl chloride", "Alkyl Halide Feedstock"),
            ("CC(C)CC(C)Cl", "Isobutyl chloride", "Alkyl Halide Feedstock"),

            # Simple Boronic Acids & Reagents
            ("OB(O)c1ccccc1", "Phenylboronic acid", "Organoboron Building Block"),
            ("OB(O)c1ccc(C)cc1", "4-Tolylboronic acid", "Organoboron Building Block"),
            ("OB(O)c1cccnc1", "3-Pyridinylboronic acid", "Heteroaryl Boronic Acid"),
        ]

        for smi, name, cat in core_feedstocks:
            self.add_compound(smiles=smi, name=name, category=cat)


_DEFAULT_CATALOG: Optional[BuildingBlockCatalog] = None

def get_default_catalog(data_dir: Optional[str] = None) -> BuildingBlockCatalog:
    """Retrieve the singleton default building block catalog."""
    global _DEFAULT_CATALOG
    if _DEFAULT_CATALOG is None:
        json_path = None
        if data_dir:
            json_path = os.path.join(data_dir, "building_blocks.json")
        _DEFAULT_CATALOG = BuildingBlockCatalog(data_file=json_path)
    return _DEFAULT_CATALOG
