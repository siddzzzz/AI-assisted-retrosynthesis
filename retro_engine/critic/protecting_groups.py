"""Automated Protecting Group (PG) Strategy Planner and Subtree Generator."""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from ..chem.mol_utils import canonicalize_smiles, get_mol_from_smiles
from rdkit import Chem


@dataclass
class ProtectionStep:
    """Represents an integrated protection-deprotection sequence for a functional group."""
    protecting_group_name: str  # "Boc", "TBS", "Acetal", "Benzyl Ester"
    target_functional_group: str  # "primary_amine", "alcohol", "ketone", "carboxylic_acid"
    protection_reagents: str
    protection_solvents: str
    protection_yield: float
    deprotection_reagents: str
    deprotection_solvents: str
    deprotection_yield: float
    explanation: str


class ProtectingGroupPlanner:
    """Manages strategic protection and deprotection transformations to resolve chemoselectivity conflicts."""

    PG_CATALOG: Dict[str, ProtectionStep] = {
        "boc": ProtectionStep(
            protecting_group_name="tert-Butyloxycarbonyl (Boc)",
            target_functional_group="primary_amine / secondary_amine",
            protection_reagents="Boc2O (1.1 eq), Et3N (1.5 eq)",
            protection_solvents="DCM or THF / H2O (biphasic)",
            protection_yield=0.96,
            deprotection_reagents="TFA / DCM (1:4, v/v) or 4M HCl in 1,4-dioxane",
            deprotection_solvents="DCM or 1,4-Dioxane",
            deprotection_yield=0.95,
            explanation="Protects basic/nucleophilic amine as carbamate; highly stable to bases and organometallics, cleaved cleanly with acid.",
        ),
        "tbs": ProtectionStep(
            protecting_group_name="tert-Butyldimethylsilyl (TBS / TBDMS)",
            target_functional_group="alcohol / phenol",
            protection_reagents="TBS-Cl (1.2 eq), Imidazole (2.0 eq)",
            protection_solvents="Anhydrous DMF or DCM",
            protection_yield=0.94,
            deprotection_reagents="TBAF (1.0M in THF, 1.2 eq) or HF·pyridine",
            deprotection_solvents="THF",
            deprotection_yield=0.96,
            explanation="Silicon-based protecting group for alcohols/phenols; robust to nucleophiles and mild acids, removed quantitatively with fluoride.",
        ),
        "acetal": ProtectionStep(
            protecting_group_name="1,3-Dioxolane (Acetal / Ketal)",
            target_functional_group="aldehyde / ketone",
            protection_reagents="Ethylene glycol, p-TsOH·H2O (cat.), Dean-Stark trap",
            protection_solvents="Toluene or Benzene (reflux)",
            protection_yield=0.92,
            deprotection_reagents="Aqueous 2M HCl or PPTS (cat.) in acetone/H2O",
            deprotection_solvents="Acetone / H2O",
            deprotection_yield=0.94,
            explanation="Converts reactive carbonyl electrophile into inert cyclic acetal/ketal, preventing hydride or nucleophilic attack.",
        ),
        "ester": ProtectionStep(
            protecting_group_name="Methyl / Ethyl Ester",
            target_functional_group="carboxylic_acid",
            protection_reagents="MeOH / EtOH, SOCl2 (cat.) or H2SO4 (cat.)",
            protection_solvents="MeOH or EtOH (reflux)",
            protection_yield=0.95,
            deprotection_reagents="LiOH·H2O (2.0 eq) in THF/MeOH/H2O (3:1:1)",
            deprotection_solvents="THF / MeOH / H2O",
            deprotection_yield=0.96,
            explanation="Masks acidic carboxylic acid proton to prevent reagent quenching; saponified with mild aqueous base.",
        ),
    }

    @classmethod
    def get_protection_plan(cls, pg_key: str) -> Optional[ProtectionStep]:
        """Retrieve the standard operating protocol for a given protecting group."""
        return cls.PG_CATALOG.get(pg_key.lower())

    @classmethod
    def generate_strategy_report(
        cls,
        conflicting_group: str,
        suggested_pg: str,
    ) -> Dict[str, Any]:
        """Construct a comprehensive report on protecting group implementation."""
        pg_info = cls.get_protection_plan(suggested_pg)
        if not pg_info:
            return {
                "needs_protection": False,
                "strategy_summary": "No standard protecting group assigned.",
            }

        total_pg_yield = round(pg_info.protection_yield * pg_info.deprotection_yield, 3)

        return {
            "needs_protection": True,
            "protecting_group": pg_info.protecting_group_name,
            "target_group": conflicting_group,
            "installation": {
                "reagents": pg_info.protection_reagents,
                "solvent": pg_info.protection_solvents,
                "yield": f"{int(pg_info.protection_yield * 100)}%",
            },
            "deprotection": {
                "reagents": pg_info.deprotection_reagents,
                "solvent": pg_info.deprotection_solvents,
                "yield": f"{int(pg_info.deprotection_yield * 100)}%",
            },
            "cumulative_pg_yield": f"{int(total_pg_yield * 100)}%",
            "extra_steps": 2,
            "rationale": pg_info.explanation,
        }
