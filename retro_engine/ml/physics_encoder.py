"""Physics-Informed Molecular Feature Extractor.
Extracts Gasteiger partial atomic charges, 3D Euclidean spatial coordinates,
electronegativity vectors, and bond order matrices for Graph Transformers.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
from ..chem.mol_utils import get_mol_from_smiles


@dataclass
class MoleculeGraphData:
    """Represents a physics-augmented molecular graph for the neural network."""
    node_features: torch.Tensor       # Shape: [N_atoms, 64]
    edge_index: torch.Tensor          # Shape: [2, N_edges]
    edge_features: torch.Tensor       # Shape: [N_edges, 16]
    spatial_distances: torch.Tensor   # Shape: [N_atoms, N_atoms] (3D Euclidean matrix)
    global_features: torch.Tensor     # Shape: [16] (MW, LogP, TPSA, CSP3, Atom count)
    num_atoms: int
    smiles: str


class MolecularPhysicsFeatureExtractor:
    """Transforms molecules into physics-grounded node, edge, and spatial representations."""

    ATOM_TYPES = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  # H, B, C, N, O, F, Si, P, S, Cl, Br, I
    HYBRIDIZATIONS = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]

    # Pauling Electronegativity scale
    ELECTRONEGATIVITY = {
        1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
        14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66
    }

    @classmethod
    def extract_features(cls, smiles: str) -> Optional[MoleculeGraphData]:
        """Convert a SMILES string into a physics-augmented molecular graph."""
        mol = get_mol_from_smiles(smiles)
        if mol is None:
            return None

        # Add explicit hydrogens for accurate electrostatic partial charges
        mol_h = Chem.AddHs(mol)
        num_atoms = mol_h.GetNumAtoms()
        if num_atoms == 0:
            return None

        # 1. Compute Gasteiger Partial Atomic Charges
        try:
            AllChem.ComputeGasteigerCharges(mol_h)
        except Exception:
            pass

        # 2. Compute 3D Coordinates (ETKDG)
        conf_3d = None
        try:
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            if AllChem.EmbedMolecule(mol_h, params) == 0:
                conf_3d = mol_h.GetConformer()
        except Exception:
            pass

        # 3. Construct Node Features Matrix [N_atoms, 64]
        node_feats = []
        positions = []

        for atom in mol_h.GetAtoms():
            vec = []
            
            # One-hot atom type (12 dims)
            atomic_num = atom.GetAtomicNum()
            vec.extend([1.0 if atomic_num == t else 0.0 for t in cls.ATOM_TYPES])
            
            # Electronegativity (1 dim)
            en = cls.ELECTRONEGATIVITY.get(atomic_num, 2.5)
            vec.append(en / 4.0)

            # Gasteiger Partial Charge (1 dim, clipped [-1.5, 1.5])
            try:
                charge = float(atom.GetProp('_GasteigerCharge'))
                if np.isnan(charge) or np.isinf(charge):
                    charge = 0.0
            except Exception:
                charge = 0.0
            vec.append(max(-1.5, min(1.5, charge)))

            # Formal Charge & Degree & Valence (3 dims)
            vec.append(float(atom.GetFormalCharge()) / 3.0)
            vec.append(float(atom.GetTotalDegree()) / 6.0)
            vec.append(float(atom.GetTotalValence()) / 8.0)

            # Aromaticity & Ring membership (2 dims)
            vec.append(1.0 if atom.GetIsAromatic() else 0.0)
            vec.append(1.0 if atom.IsInRing() else 0.0)

            # Hybridization One-Hot (5 dims)
            hyb = atom.GetHybridization()
            vec.extend([1.0 if hyb == h else 0.0 for h in cls.HYBRIDIZATIONS])

            # Implicit H count & Mass (2 dims)
            vec.append(float(atom.GetTotalNumHs()) / 4.0)
            vec.append(float(atom.GetMass()) / 150.0)

            # Pad remaining to reach 64 dimensions
            pad_len = 64 - len(vec)
            if pad_len > 0:
                vec.extend([0.0] * pad_len)
            else:
                vec = vec[:64]

            node_feats.append(vec)

            # 3D spatial position
            if conf_3d is not None:
                pos = conf_3d.GetAtomPosition(atom.GetIdx())
                positions.append([pos.x, pos.y, pos.z])
            else:
                positions.append([0.0, 0.0, 0.0])

        node_tensor = torch.tensor(node_feats, dtype=torch.float32)

        # 4. Construct 3D Spatial Distance Matrix [N, N]
        pos_arr = np.array(positions)
        dist_matrix = np.zeros((num_atoms, num_atoms), dtype=np.float32)
        if conf_3d is not None:
            for i in range(num_atoms):
                for j in range(num_atoms):
                    dist = np.linalg.norm(pos_arr[i] - pos_arr[j])
                    dist_matrix[i, j] = dist
        dist_tensor = torch.tensor(dist_matrix, dtype=torch.float32)

        # 5. Construct Covalent Edges & Edge Features [N_edges, 16]
        src_list, dst_list, edge_feats = [], [], []
        for bond in mol_h.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            b_type = bond.GetBondType()

            # 16-dim bond representation
            b_vec = [
                1.0 if b_type == Chem.rdchem.BondType.SINGLE else 0.0,
                1.0 if b_type == Chem.rdchem.BondType.DOUBLE else 0.0,
                1.0 if b_type == Chem.rdchem.BondType.TRIPLE else 0.0,
                1.0 if b_type == Chem.rdchem.BondType.AROMATIC else 0.0,
                1.0 if bond.GetIsConjugated() else 0.0,
                1.0 if bond.IsInRing() else 0.0,
            ]
            # Spatial bond length
            b_dist = dist_matrix[u, v] if conf_3d is not None else 1.5
            b_vec.append(b_dist / 5.0)
            
            # Pad to 16 dims
            b_vec.extend([0.0] * (16 - len(b_vec)))

            # Bidirectional graph edges
            src_list.extend([u, v])
            dst_list.extend([v, u])
            edge_feats.extend([b_vec, b_vec])

        if not src_list:
            edge_index = torch.zeros((2, 1), dtype=torch.long)
            edge_tensor = torch.zeros((1, 16), dtype=torch.float32)
        else:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_tensor = torch.tensor(edge_feats, dtype=torch.float32)

        # 6. Global Molecular Descriptors [16 dims]
        mw = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        rot_b = rdMolDescriptors.CalcNumRotatableBonds(mol)
        csp3 = Descriptors.FractionCSP3(mol)
        
        glob_vec = [
            mw / 500.0,
            (logp + 3.0) / 10.0,
            tpsa / 200.0,
            float(rot_b) / 15.0,
            csp3,
            float(num_atoms) / 50.0,
        ]
        glob_vec.extend([0.0] * (16 - len(glob_vec)))
        global_tensor = torch.tensor(glob_vec, dtype=torch.float32)

        return MoleculeGraphData(
            node_features=node_tensor,
            edge_index=edge_index,
            edge_features=edge_tensor,
            spatial_distances=dist_tensor,
            global_features=global_tensor,
            num_atoms=num_atoms,
            smiles=smiles,
        )
