"""Automated Reaction Stoichiometry and Bench Batch Protocol Calculator.
Computes exact millimoles, masses, liquid volumes, and equivalents for any target scale.
"""

from typing import List, Dict, Any, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors
from ..chem.mol_utils import get_mol_from_smiles


# Common laboratory reagent densities and molecular weights
REAGENT_PROPERTIES: Dict[str, Dict[str, float]] = {
    "triethylamine": {"mw": 101.19, "density": 0.726},
    "et3n": {"mw": 101.19, "density": 0.726},
    "diisopropylethylamine": {"mw": 129.24, "density": 0.742},
    "dipea": {"mw": 129.24, "density": 0.742},
    "pyridine": {"mw": 79.10, "density": 0.982},
    "dmap": {"mw": 122.17, "density": 1.000},
    "acetic anhydride": {"mw": 102.09, "density": 1.082},
    "ac2o": {"mw": 102.09, "density": 1.082},
    "acetyl chloride": {"mw": 78.50, "density": 1.104},
    "thionyl chloride": {"mw": 118.97, "density": 1.638},
    "socl2": {"mw": 118.97, "density": 1.638},
    "boc2o": {"mw": 218.25, "density": 0.950},
    "tbs-cl": {"mw": 150.72, "density": 0.890},
    "tbaf": {"mw": 261.46, "density": 0.900},
    "hatu": {"mw": 380.23, "density": 1.000},
    "edc·hcl": {"mw": 191.70, "density": 1.000},
    "hobt": {"mw": 135.12, "density": 1.000},
    "k2co3": {"mw": 138.21, "density": 2.430},
    "cs2co3": {"mw": 325.82, "density": 4.070},
    "pd(pph3)4": {"mw": 1155.56, "density": 1.000},
    "pd(dppf)cl2": {"mw": 732.11, "density": 1.000},
    "pd(oac)2": {"mw": 224.51, "density": 1.000},
    "nabh4": {"mw": 37.83, "density": 1.070},
    "nabh(oac)3": {"mw": 211.94, "density": 1.000},
    "hno3": {"mw": 63.01, "density": 1.410},
    "h2so4": {"mw": 98.08, "density": 1.840},
    "br2": {"mw": 159.81, "density": 3.102},
    "nano2": {"mw": 69.00, "density": 2.168},
    "cubr": {"mw": 143.45, "density": 4.980},
    "nacn": {"mw": 49.01, "density": 1.600},
}


class ReactionStoichiometryCalculator:
    """Calculates wet-lab stoichiometry tables for bench execution at any desired target scale."""

    @classmethod
    def calculate_step_stoichiometry(
        cls,
        product_smiles: str,
        reactants_smiles: List[str],
        reagents: List[str],
        solvents: List[str],
        target_product_mass_g: float = 1.0,
        step_yield: float = 0.85,
    ) -> Dict[str, Any]:
        """Compute stoichiometric table for a single reaction step."""
        prod_mol = get_mol_from_smiles(product_smiles)
        prod_mw = Descriptors.ExactMolWt(prod_mol) if prod_mol else 200.0
        
        # Calculate required mmol of desired product
        target_product_mmol = (target_product_mass_g / prod_mw) * 1000.0
        
        # Account for step yield to find required theoretical starting mmol
        effective_yield = max(0.20, min(1.0, step_yield))
        required_input_mmol = target_product_mmol / effective_yield

        entries = []

        # 1. Main Reactants
        for idx, r_smi in enumerate(reactants_smiles):
            r_mol = get_mol_from_smiles(r_smi)
            r_mw = Descriptors.ExactMolWt(r_mol) if r_mol else 150.0
            
            # Limiting reagent (1.00 eq), second reactant (1.10 - 1.20 eq)
            eq = 1.00 if idx == 0 else 1.15
            mmol = required_input_mmol * eq
            mass_mg = mmol * r_mw
            mass_g = mass_mg / 1000.0

            entries.append({
                "role": "Limiting Reactant" if idx == 0 else "Reactant",
                "name": f"Reactant {idx + 1}",
                "smiles": r_smi,
                "mw": round(r_mw, 2),
                "equivalents": eq,
                "mmol": round(mmol, 2),
                "mass_g": round(mass_g, 3),
                "mass_mg": round(mass_mg, 1),
                "volume_ml": None,
            })

        # 2. Catalysts & Reagents
        for reg in reagents:
            reg_clean = reg.lower().strip()
            # Match against known reagent database
            matched = None
            for key, val in REAGENT_PROPERTIES.items():
                if key in reg_clean:
                    matched = (key, val)
                    break

            if matched:
                key, props = matched
                mw = props["mw"]
                density = props.get("density", 1.0)
                
                # Assign sensible chemical equivalents
                if "pd" in key or "cat" in reg_clean or "cu" in key:
                    eq = 0.03  # 3 mol% catalyst
                    role = "Catalyst"
                elif "base" in reg_clean or "k2co3" in key or "cs2co3" in key or "et3n" in key or "dipea" in key:
                    eq = 2.00
                    role = "Base"
                elif "hno3" in key or "h2so4" in key:
                    eq = 1.50
                    role = "Acid Reagent"
                else:
                    eq = 1.20
                    role = "Coupling Reagent"

                mmol = required_input_mmol * eq
                mass_mg = mmol * mw
                mass_g = mass_mg / 1000.0
                vol_ml = round(mass_g / density, 2) if density > 0 and density < 2.0 else None

                entries.append({
                    "role": role,
                    "name": reg,
                    "smiles": "",
                    "mw": round(mw, 2),
                    "equivalents": eq,
                    "mmol": round(mmol, 2),
                    "mass_g": round(mass_g, 3),
                    "mass_mg": round(mass_mg, 1),
                    "volume_ml": vol_ml,
                })

        # 3. Solvent Volume (Standard 0.20 M reaction concentration)
        primary_solvent = solvents[0] if solvents else "DMF"
        target_molarity = 0.20  # mol/L
        solvent_vol_ml = round((required_input_mmol / 1000.0) / target_molarity * 1000.0, 1)
        solvent_vol_ml = max(5.0, solvent_vol_ml)  # Minimum 5 mL for lab vessel

        return {
            "target_scale_g": target_product_mass_g,
            "target_product_mmol": round(target_product_mmol, 2),
            "expected_yield_pct": round(effective_yield * 100, 1),
            "primary_solvent": primary_solvent,
            "recommended_solvent_volume_ml": solvent_vol_ml,
            "concentration_molar": target_molarity,
            "stoichiometry_table": entries,
        }
