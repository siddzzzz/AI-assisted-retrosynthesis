"""Chemoselectivity and Functional Group Interference Analysis."""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from ..chem.mol_utils import identify_functional_groups, get_mol_from_smiles
from ..chem.reaction_rules import RetroReactionRule


@dataclass
class ConflictReport:
    """Detailed assessment of chemoselectivity and competing functional group clashes."""
    has_conflict: bool
    conflicting_groups: List[str]
    severity: str  # "none", "minor", "moderate", "critical"
    needs_protection: bool
    suggested_protecting_group: Optional[str]  # e.g. "boc", "tbs", "acetal", "benzyl_ester"
    explanation: str
    confidence_penalty: float  # Value from 0.0 to 0.70 deducted from rule confidence


class ChemoselectivityAnalyzer:
    """Evaluates whether competing functional groups will interfere with the intended transformation."""

    # Explicit cross-reactivity clashes and severity
    REAGENT_CLASH_RULES: Dict[str, Dict[str, Dict[str, Any]]] = {
        "Organometallic C-C Addition": {
            "alcohol": {
                "severity": "critical",
                "protection": "tbs",
                "penalty": 0.50,
                "msg": "Free hydroxyl (-OH) group possesses an acidic proton (pKa ~16) that will quantitatively quench the organometallic reagent (Grignard/organolithium), preventing addition.",
            },
            "carboxylic_acid": {
                "severity": "critical",
                "protection": "ester",
                "penalty": 0.60,
                "msg": "Carboxylic acid proton (pKa ~4-5) will immediately deprotonate and quench the basic carbon nucleophile.",
            },
            "primary_amine": {
                "severity": "critical",
                "protection": "boc",
                "penalty": 0.45,
                "msg": "Amine N-H protons (pKa ~35-38 with organolithium) or coordinating lone pair will compete with nucleophilic attack on the target carbonyl.",
            },
            "phenol": {
                "severity": "critical",
                "protection": "tbs",
                "penalty": 0.55,
                "msg": "Phenolic proton (pKa ~10) is strongly acidic and will destroy organometallic reagents.",
            },
        },
        "Acylation": {
            "primary_amine": {
                "severity": "moderate",
                "protection": "boc",
                "penalty": 0.30,
                "msg": "Amine is significantly more nucleophilic than alcohol and will selectively undergo N-acylation instead of the desired O-acylation.",
            },
            "secondary_amine": {
                "severity": "moderate",
                "protection": "boc",
                "penalty": 0.25,
                "msg": "Secondary amine will outcompete hydroxyl group for electrophilic acyl transfer.",
            },
        },
        "C-N Coupling": {
            "acid_chloride": {
                "severity": "critical",
                "protection": "carboxylic_acid",
                "penalty": 0.40,
                "msg": "Uncontrolled multi-acylation or polymer formation can occur if multiple electrophilic centers are unmasked.",
            },
        },
        "Transition Metal C-N Coupling": {
            "carboxylic_acid": {
                "severity": "critical",
                "protection": "ester",
                "penalty": 0.35,
                "msg": "Carboxylic acid poisons strong alkoxide/carbonate bases used in Buchwald-Hartwig aminations.",
            },
            "alcohol": {
                "severity": "moderate",
                "protection": "tbs",
                "penalty": 0.20,
                "msg": "Alcohol can undergo competitive palladium-catalyzed C-O Ullmann/Buchwald etherification.",
            },
        },
        "Electrophilic Aromatic Substitution": {
            "primary_amine": {
                "severity": "critical",
                "protection": "boc",
                "penalty": 0.40,
                "msg": "Strong Lewis acids (AlCl3/FeCl3) form strong unreactive complexes with basic amines, deactivating the ring and halting Friedel-Crafts reaction.",
            },
        },
    }

    @classmethod
    def evaluate_step(
        cls,
        reactants_smi: List[str],
        product_smi: str,
        rule: RetroReactionRule,
    ) -> ConflictReport:
        """Analyze if any reactant possesses functional groups incompatible with the reaction class."""
        reaction_class = rule.reaction_class
        all_detected_groups: List[str] = []

        for r_smi in reactants_smi:
            groups = identify_functional_groups(r_smi)
            all_detected_groups.extend(groups)

        # Check against the rule's explicit incompatible groups
        conflicts = []
        highest_severity = "none"
        highest_penalty = 0.0
        suggested_pg = None
        explanations = []

        # 1. Rule-level incompatibilities
        for group in all_detected_groups:
            if group in rule.incompatible_groups:
                conflicts.append(group)
                if group in rule.protection_needed_for:
                    suggested_pg = rule.protection_needed_for[group]

        # 2. Reagent class interference matrix
        class_rules = cls.REAGENT_CLASH_RULES.get(reaction_class, {})
        for group in all_detected_groups:
            if group in class_rules:
                entry = class_rules[group]
                conflicts.append(group)
                explanations.append(entry["msg"])
                if entry["penalty"] > highest_penalty:
                    highest_penalty = entry["penalty"]
                    highest_severity = entry["severity"]
                    if not suggested_pg and entry.get("protection"):
                        suggested_pg = entry["protection"]

        conflicts = list(set(conflicts))

        if not conflicts:
            return ConflictReport(
                has_conflict=False,
                conflicting_groups=[],
                severity="none",
                needs_protection=False,
                suggested_protecting_group=None,
                explanation="No functional group clashes detected. The reaction pathway is clean and chemoselective.",
                confidence_penalty=0.0,
            )

        if not explanations:
            explanations.append(
                f"Competing functional groups ({', '.join(conflicts)}) detected which may lead to side-reactions."
            )

        return ConflictReport(
            has_conflict=True,
            conflicting_groups=conflicts,
            severity=highest_severity if highest_severity != "none" else "moderate",
            needs_protection=suggested_pg is not None,
            suggested_protecting_group=suggested_pg,
            explanation="; ".join(explanations),
            confidence_penalty=min(0.60, highest_penalty if highest_penalty > 0 else 0.20),
        )
