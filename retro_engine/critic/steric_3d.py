"""3D Conformer Generation and Steric Accessibility / Crowding Analyzer.
Uses RDKit 3D MMFF94 force field and ETKDG to assess physical steric hindrance.
"""

from typing import Dict, List, Optional, Any, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
import numpy as np


class Steric3DAnalyzer:
    """Evaluates 3D molecular conformation and steric hindrance around reaction centers."""

    @classmethod
    def generate_3d_conformer(cls, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Generate an energy-minimized 3D conformer using ETKDG and MMFF94."""
        if mol is None:
            return None
        
        mol_3d = Chem.AddHs(Chem.Mol(mol))
        try:
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            res = AllChem.EmbedMolecule(mol_3d, params)
            if res != 0:
                # Fallback to standard embed
                res = AllChem.EmbedMolecule(mol_3d)
            
            if res == 0:
                # Optimize geometry using MMFF94 force field
                AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=200)
                return mol_3d
        except Exception:
            pass

        return None

    @classmethod
    def calculate_atom_steric_burial(
        cls, mol_3d: Chem.Mol, atom_idx: int, radius_angstrom: float = 3.5
    ) -> float:
        """Calculate the percentage of steric crowding / neighbor density within a sphere of radius R."""
        if mol_3d is None or atom_idx >= mol_3d.GetNumAtoms():
            return 0.0

        try:
            conf = mol_3d.GetConformer()
            target_pos = np.array(conf.GetAtomPosition(atom_idx))
            
            neighbor_count = 0
            heavy_neighbor_count = 0
            
            for i in range(mol_3d.GetNumAtoms()):
                if i == atom_idx:
                    continue
                pos = np.array(conf.GetAtomPosition(i))
                dist = np.linalg.norm(target_pos - pos)
                if dist <= radius_angstrom:
                    neighbor_count += 1
                    if mol_3d.GetAtomWithIdx(i).GetAtomicNum() > 1:
                        heavy_neighbor_count += 1

            # Empirical burial score: 0% (completely open) to 100% (highly crowded/buried)
            # Typically 6-8 heavy neighbors within 3.5A is dense crowding
            burial_score = min(100.0, (heavy_neighbor_count / 7.0) * 100.0)
            return round(burial_score, 1)
        except Exception:
            return 0.0

    @classmethod
    def evaluate_molecule_sterics(
        cls, smiles: str, reaction_class: str = ""
    ) -> Dict[str, Any]:
        """Compute holistic 3D steric assessment and benchtop kinetics recommendations."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "has_3d": False,
                "steric_burial_score": 25.0,
                "accessibility": "Normal",
                "bench_advice": "Standard reaction kinetics expected.",
            }

        mol_3d = cls.generate_3d_conformer(mol)
        if mol_3d is None:
            # Fallback 2D topological estimation
            num_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
            num_heavy = mol.GetNumHeavyAtoms()
            estimated_score = min(85.0, max(15.0, num_heavy * 2.5 + num_rot * 2.0))
            return {
                "has_3d": False,
                "steric_burial_score": round(estimated_score, 1),
                "accessibility": "Moderately Accessible",
                "bench_advice": "Standard thermal reaction conditions suitable.",
            }

        # Calculate max burial score across heteroatoms / reactive centers
        burial_scores = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in [7, 8, 9, 15, 16, 17, 35, 53] or atom.GetIsAromatic():
                score = cls.calculate_atom_steric_burial(mol_3d, atom.GetIdx())
                burial_scores.append(score)

        avg_burial = np.mean(burial_scores) if burial_scores else 30.0
        max_burial = max(burial_scores) if burial_scores else 30.0

        if max_burial > 70.0:
            accessibility = "Sterically Crowded / Hindered"
            advice = "Sterically encumbered reaction center detected. Consider elevated temperature, microwave irradiation, or activating catalysts."
        elif max_burial > 45.0:
            accessibility = "Moderately Accessible"
            advice = "Moderate steric accessibility. Standard laboratory temperature and agitation recommended."
        else:
            accessibility = "High Kinetic Accessibility"
            advice = "Unencumbered open reaction center. Clean, rapid reaction kinetics expected at room temperature."

        return {
            "has_3d": True,
            "steric_burial_score": round(max_burial, 1),
            "average_burial": round(avg_burial, 1),
            "accessibility": accessibility,
            "bench_advice": advice,
        }
