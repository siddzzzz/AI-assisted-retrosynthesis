"""Molecular representation, canonicalization, property computation, and 2D rendering."""

from typing import Dict, List, Optional, Tuple, Any
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, rdMolDescriptors, rdchem
from rdkit.Chem.Draw import rdMolDraw2D
import re


# Common functional group SMARTS patterns for detection and chemoselectivity analysis
FUNCTIONAL_GROUP_PATTERNS: Dict[str, str] = {
    "primary_amine": "[NX3;H2;!$(NC=O)]",
    "secondary_amine": "[NX3;H1;!$(NC=O)]",
    "tertiary_amine": "[NX3;H0;!$(NC=O)]",
    "aromatic_amine": "[c][NX3;H2,H1]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "ester": "[CX3](=O)[OX2H0][#6]",
    "amide": "[CX3](=O)[NX3]",
    "alcohol": "[OX2H;!$(OC=O);!$(Oc)]",
    "phenol": "[c][OX2H]",
    "aldehyde": "[CX3H1](=O)",
    "ketone": "[#6][CX3](=O)[#6]",
    "ether": "[OD2]([#6])[#6]",
    "alkyl_halide": "[CX4][F,Cl,Br,I]",
    "aryl_halide": "[c][F,Cl,Br,I]",
    "aryl_boronic_acid": "[c][BX3]([OX2H])[OX2H]",
    "aryl_boronate_ester": "[c][BX3]([OX2])[OX2]",
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
    "terminal_alkyne": "[CX2]#[CX2H1]",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "nitrile": "[CX2]#[NX1]",
    "thiol": "[#16X2H]",
    "thioether": "[#16X2H0]",
    "azide": "[$(N=[N+]=[N-]),$([N-][N+]#[N])]",
    "epoxide": "[OX2r3]1[CX3r3][CX3r3]1",
    "isocyanate": "[NX2]=[CX2]=[OX1]",
    "sulfonyl_chloride": "[SX4](=O)(=O)Cl",
    "acid_chloride": "[CX3](=O)Cl",
    "protecting_boc": "[CX3](=O)OC(C)(C)C",
    "protecting_acetal": "[CX4]([OX2][#6])([OX2][#6])",
    "protecting_tbs": "[Si](C)(C)C(C)(C)C",
}

# Pre-compile SMARTS patterns for fast matching
COMPILED_PATTERNS: Dict[str, Chem.Mol] = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in FUNCTIONAL_GROUP_PATTERNS.items()
    if Chem.MolFromSmarts(smarts) is not None
}


def get_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Parse a SMILES string into an RDKit Mol object, sanitizing it."""
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is not None:
            Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def canonicalize_smiles(smiles: str, remove_stereo: bool = False) -> Optional[str]:
    """Return the canonical SMILES string for a molecule.
    
    Args:
        smiles: Input SMILES string.
        remove_stereo: If True, strips stereochemical markers for broad catalog matching.
    """
    mol = get_mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        if remove_stereo:
            Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def smiles_to_inchikey(smiles: str) -> Optional[str]:
    """Convert a SMILES string to its standard InChIKey for robust catalog lookup."""
    mol = get_mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def calculate_mol_properties(smiles: str) -> Dict[str, Any]:
    """Compute key physicochemical and drug-like properties for a molecule."""
    mol = get_mol_from_smiles(smiles)
    if mol is None:
        return {
            "valid": False,
            "smiles": smiles,
            "error": "Invalid chemical structure or SMILES syntax",
        }

    canonical_smi = Chem.MolToSmiles(mol, canonical=True)
    mw = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    formula = rdMolDescriptors.CalcMolFormula(mol)
    fraction_csp3 = Descriptors.FractionCSP3(mol)

    # Functional groups present
    groups = identify_functional_groups(mol)

    return {
        "valid": True,
        "smiles": canonical_smi,
        "formula": formula,
        "molecular_weight": round(mw, 3),
        "logp": round(logp, 2),
        "tpsa": round(tpsa, 1),
        "hbd": hbd,
        "hba": hba,
        "rotatable_bonds": rot_bonds,
        "heavy_atoms": heavy_atoms,
        "fraction_csp3": round(fraction_csp3, 3),
        "functional_groups": groups,
    }


def identify_functional_groups(mol_or_smiles: Any) -> List[str]:
    """Identify functional groups present in a molecule using SMARTS matching."""
    if isinstance(mol_or_smiles, str):
        mol = get_mol_from_smiles(mol_or_smiles)
    else:
        mol = mol_or_smiles

    if mol is None:
        return []

    detected = []
    for name, pattern in COMPILED_PATTERNS.items():
        if mol.HasSubstructMatch(pattern):
            detected.append(name)
    return detected


def calculate_atom_economy(reactants_smi: List[str], product_smi: str) -> float:
    """Calculate the atom economy percentage of a reaction step.
    
    Atom Economy = (Molecular Weight of Desired Product / Sum of MW of All Reactants) * 100
    """
    prod_mol = get_mol_from_smiles(product_smi)
    if prod_mol is None:
        return 0.0
    prod_mw = Descriptors.ExactMolWt(prod_mol)

    reactants_mw = 0.0
    for r_smi in reactants_smi:
        r_mol = get_mol_from_smiles(r_smi)
        if r_mol is not None:
            reactants_mw += Descriptors.ExactMolWt(r_mol)

    if reactants_mw <= 0:
        return 0.0

    return min(100.0, round((prod_mw / reactants_mw) * 100.0, 1))


def render_mol_svg(
    smiles: str,
    width: int = 280,
    height: int = 180,
    highlight_atoms: Optional[List[int]] = None,
    dark_mode: bool = False,
) -> str:
    """Generate a clean, high-resolution SVG string for 2D molecular display with high-contrast bonds."""
    mol = get_mol_from_smiles(smiles)
    if mol is None:
        return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#64748b" font-family="sans-serif" font-size="12">Invalid Structure</text></svg>'

    try:
        # Prepare 2D coordinates
        mol_to_draw = Chem.Mol(mol)
        AllChem.Compute2DCoords(mol_to_draw)

        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = drawer.drawOptions()
        opts.clearBackground = True
        opts.bondLineWidth = 2.4
        opts.fixedFontSize = 14
        opts.minFontSize = 11
        opts.padding = 0.12
        opts.additionalAtomLabelPadding = 0.08

        # Crisp High-Contrast Light Chemistry Palette
        opts.setBackgroundColour((1.0, 1.0, 1.0, 1.0))
        opts.symbolColour = (0.05, 0.05, 0.05)

        if highlight_atoms:
            drawer.DrawMolecule(mol_to_draw, highlightAtoms=highlight_atoms)
        else:
            drawer.DrawMolecule(mol_to_draw)

        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg
    except Exception as e:
        return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#ef4444" font-family="sans-serif" font-size="12">Render Error</text></svg>'
