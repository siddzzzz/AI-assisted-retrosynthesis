"""Green Chemistry metrics, environmental evaluation, and multi-objective Pareto scoring."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..chem.mol_utils import calculate_atom_economy, get_mol_from_smiles
from rdkit.Chem import Descriptors


@dataclass
class RouteMetrics:
    """Comprehensive sustainability, safety, and chemical efficiency scores."""
    total_steps: int
    cumulative_yield: float  # Percentage (e.g., 68.5%)
    average_atom_economy: float  # Percentage (e.g., 85.2%)
    estimated_pmi: float  # Process Mass Intensity (lower is greener, typical pharma is 20-100)
    estimated_e_factor: float  # kg waste / kg product
    green_chemistry_score: float  # 0 to 100 (higher is greener)
    hazard_score: float  # 0 (completely safe) to 100 (high hazard)
    hazard_warnings: List[str]
    pareto_tags: List[str]  # e.g., ["Shortest Route", "Greenest Pathway", "Highest Yield"]


class GreenChemistryEvaluator:
    """Evaluates multi-step chemical synthesis routes across environmental, safety, and chemical metrics."""

    HAZARDOUS_PATTERNS = [
        ("N=[N+]=[N-]", "Azide functionality detected — potential explosive / runaway thermal hazard if heated."),
        ("OO", "Peroxide or hydroperoxide linkage detected — potential shock-sensitive explosive."),
        ("[Sn]", "Organotin species involved — high aquatic and mammalian neurotoxicity."),
        ("[Hg]", "Mercury catalyst/reagent involved — severe toxicological and environmental bioaccumulation hazard."),
        ("ClC(Cl)Cl", "Chloroform / Carbon tetrachloride solvent — toxic / carcinogenic solvent hazard."),
    ]

    @classmethod
    def evaluate_step_hazards(cls, smiles_list: List[str], reagents: List[str]) -> List[str]:
        """Scan molecular structures and reagents for safety hazard red flags."""
        warnings = []
        for smi in smiles_list:
            for pattern, warn in cls.HAZARDOUS_PATTERNS:
                if pattern in smi and warn not in warnings:
                    warnings.append(warn)

        reagent_str = " ".join(reagents)
        if "n-BuLi" in reagent_str or "tert-BuLi" in reagent_str:
            warnings.append("Pyrophoric organolithium reagent: violently ignites upon contact with air.")
        if "AlCl3" in reagent_str:
            warnings.append("Lewis acid generates copious corrosive HCl gas upon contact with ambient moisture.")
        if "SOCl2" in reagent_str or "PCl5" in reagent_str:
            warnings.append("Toxic, pungent chlorinating agent; generates SO2/HCl gas byproduct.")

        return warnings

    @classmethod
    def evaluate_route(
        cls,
        steps: List[Dict[str, Any]],
    ) -> RouteMetrics:
        """Compute holistic green chemistry and multi-objective metrics for a complete synthesis route."""
        total_steps = len(steps)
        if total_steps == 0:
            return RouteMetrics(
                total_steps=0,
                cumulative_yield=100.0,
                average_atom_economy=100.0,
                estimated_pmi=1.0,
                estimated_e_factor=0.0,
                green_chemistry_score=100.0,
                hazard_score=0.0,
                hazard_warnings=[],
                pareto_tags=["Direct Feedstock"],
            )

        # 1. Cumulative yield & Atom Economy
        cum_yield = 1.0
        atom_economies = []
        all_warnings = []
        has_hazardous_solvent = False
        uses_aqueous_or_green = False

        for step in steps:
            step_yield = step.get("step_yield", step.get("yield", 0.85))
            cum_yield *= step_yield

            ae = step.get("atom_economy", 85.0)
            atom_economies.append(ae)

            # Hazard checking
            reactants = step.get("reactants_smiles", step.get("reactants", []))
            product = step.get("product_smiles", step.get("product", ""))
            reagents = step.get("reagents", [])
            step_warnings = cls.evaluate_step_hazards(reactants + [product], reagents)
            all_warnings.extend(step_warnings)

            # Solvents
            solvents = step.get("solvents", [])
            if any(s in ["DCM", "DMF", "1,2-DCE", "Nitrobenzene"] for s in solvents):
                has_hazardous_solvent = True
            if any(s in ["Water", "EtOH", "EtOAc", "2-MeTHF"] for s in solvents):
                uses_aqueous_or_green = True

        all_warnings = list(set(all_warnings))
        avg_ae = sum(atom_economies) / len(atom_economies) if atom_economies else 80.0

        # Process Mass Intensity (PMI) estimate
        estimated_pmi = round(max(5.0, total_steps * 16.0 * (100.0 / max(cum_yield * 100, 1.0))), 1)
        estimated_e_factor = round(max(0.0, estimated_pmi - 1.0), 1)

        # Hazard Score calculation (0 to 100)
        base_hazard = len(all_warnings) * 20.0
        if has_hazardous_solvent:
            base_hazard += 15.0
        hazard_score = min(100.0, round(base_hazard, 1))

        # Green Chemistry Score (0 to 100)
        green_score = (
            (avg_ae * 0.35)
            + ((cum_yield * 100.0) * 0.30)
            + (25.0 if uses_aqueous_or_green else 10.0)
            - (hazard_score * 0.20)
        )
        green_score = max(0.0, min(100.0, round(green_score, 1)))

        # Pareto Classification tags
        tags = []
        if total_steps <= 2:
            tags.append("Direct Pathway")
        elif total_steps >= 3:
            tags.append(f"{total_steps}-Step Multi-Stage Synthesis")

        if green_score >= 70.0:
            tags.append("Greenest Pathway")
        if cum_yield >= 0.70:
            tags.append("High Yield Route")

        if not tags:
            tags.append("Standard Synthetic Pathway")

        return RouteMetrics(
            total_steps=total_steps,
            cumulative_yield=round(cum_yield * 100.0, 1),
            average_atom_economy=round(avg_ae, 1),
            estimated_pmi=estimated_pmi,
            estimated_e_factor=estimated_e_factor,
            green_chemistry_score=green_score,
            hazard_score=hazard_score,
            hazard_warnings=all_warnings,
            pareto_tags=tags,
        )
