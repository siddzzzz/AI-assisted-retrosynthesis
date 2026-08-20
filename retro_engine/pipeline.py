"""High-level Orchestration API for the AI-Assisted Retrosynthesis Engine."""

from typing import List, Dict, Optional, Any
import json
import time

from .chem.mol_utils import (
    canonicalize_smiles,
    calculate_mol_properties,
    render_mol_svg,
    get_mol_from_smiles,
)
from .chem.reaction_rules import ReactionRuleLibrary
from .chem.building_blocks import BuildingBlockCatalog, get_default_catalog
from .search.mcts import RetrosynthesisSearchTree, SynthesisPlan


# Curated benchmark molecules with common names and SMILES
BENCHMARK_PRESETS = [
    {
        "name": "Paracetamol (Acetaminophen)",
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "category": "Analgesic / Antipyretic",
        "description": "Multi-step route: Phenol -> Nitrophenol -> p-Aminophenol -> Paracetamol.",
    },
    {
        "name": "Aspirin (Acetylsalicylic acid)",
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "category": "NSAID Analgesic",
        "description": "Multi-step route: Phenol -> Salicylic acid (Kolbe-Schmitt) -> Aspirin.",
    },
    {
        "name": "Ibuprofen",
        "smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        "category": "NSAID Anti-inflammatory",
        "description": "Multi-step route: Isobutylbenzene -> Acylation -> Nitrile -> Ibuprofen.",
    },
    {
        "name": "Lidocaine",
        "smiles": "CCN(CC)CC(=O)Nc1c(C)cccc1C",
        "category": "Local Anesthetic",
        "description": "Multi-step amide coupling and nucleophilic amination route.",
    },
    {
        "name": "Phenylephrine",
        "smiles": "CNC[C@H](O)c1cccc(O)c1",
        "category": "Decongestant",
        "description": "Multi-step sequence with ketone reduction and reductive amination.",
    },
]


class RetrosynthesisEngine:
    """Complete, unified AI-assisted retrosynthetic synthesis planning pipeline."""

    def __init__(self, data_dir: Optional[str] = None):
        self.catalog = get_default_catalog(data_dir=data_dir)
        self.rule_lib = ReactionRuleLibrary()
        self.search_tree = RetrosynthesisSearchTree(
            rule_library=self.rule_lib,
            catalog=self.catalog,
        )

    def analyze_target(self, smiles: str) -> Dict[str, Any]:
        """Validate SMILES, compute properties, and render high-contrast 2D chemical structure."""
        props = calculate_mol_properties(smiles)
        if not props.get("valid"):
            return props

        canonical_smi = props["smiles"]
        svg = render_mol_svg(canonical_smi, width=300, height=200, dark_mode=False)
        props["svg_light"] = svg
        props["svg_dark"] = svg
        return props

    def plan_synthesis(
        self,
        target_smiles: str,
        max_depth: int = 6,
        max_routes: int = 6,
        time_limit_sec: float = 12.0,
    ) -> Dict[str, Any]:
        """Execute multi-step retrosynthetic route search and return scored pathways."""
        target_analysis = self.analyze_target(target_smiles)
        if not target_analysis.get("valid"):
            return {
                "success": False,
                "error": target_analysis.get("error", "Invalid molecule"),
                "target": None,
                "plans": [],
            }

        canonical_smi = target_analysis["smiles"]
        plans: List[SynthesisPlan] = self.search_tree.plan_synthesis(
            target_smiles=canonical_smi,
            max_depth=max_depth,
            max_routes=max_routes,
            time_limit_sec=time_limit_sec,
        )

        serialized_plans = [p.to_dict() for p in plans]

        # Add SVG drawings for all intermediates and starting materials in each plan
        for plan_dict in serialized_plans:
            for step in plan_dict["steps"]:
                step["product_svg"] = render_mol_svg(step["product_smiles"], width=220, height=140, dark_mode=False)
                step["reactants_svgs"] = [
                    render_mol_svg(r_smi, width=180, height=120, dark_mode=False)
                    for r_smi in step["reactants_smiles"]
                ]
            for sm in plan_dict["starting_materials"]:
                sm["svg"] = render_mol_svg(sm["smiles"], width=180, height=120, dark_mode=False)

        return {
            "success": True,
            "target": target_analysis,
            "total_plans_found": len(serialized_plans),
            "plans": serialized_plans,
        }

    def generate_laboratory_sop(self, plan_dict: Dict[str, Any]) -> str:
        """Generate a complete Standard Operating Procedure (SOP) laboratory protocol in Markdown."""
        lines = []
        target_smi = plan_dict.get("target_smiles", "Target Molecule")
        metrics = plan_dict.get("metrics", {})
        steps = plan_dict.get("steps", [])

        lines.append(f"# Laboratory Standard Operating Procedure (SOP)")
        lines.append(f"**Target Compound (SMILES):** `{target_smi}`  ")
        lines.append(f"**Total Steps:** {metrics.get('total_steps', 0)} | **Overall Predicted Yield:** {metrics.get('cumulative_yield', 0)}% | **Green Score:** {metrics.get('green_chemistry_score', 0)}/100  ")
        lines.append(f"**Estimated PMI:** {metrics.get('estimated_pmi', 0)} | **Average Atom Economy:** {metrics.get('average_atom_economy', 0)}%  ")
        lines.append("\n---\n")

        # Starting Materials Table
        lines.append("## 1. Required Raw Material Feedstocks\n")
        lines.append("| Compound Name | SMILES | Category |")
        lines.append("|---|---|---|")
        for sm in plan_dict.get("starting_materials", []):
            lines.append(f"| {sm.get('name', 'Feedstock Material')} | `{sm.get('smiles', '')}` | {sm.get('category', 'Commodity Chemical')} |")

        lines.append("\n---\n")

        # Step by step execution
        lines.append("## 2. Step-by-Step Reaction Protocol\n")
        for step in steps:
            s_num = step["step_number"]
            r_name = step["reaction_name"]
            prod = step["product_smiles"]
            reacts = ", ".join(f"`{r}`" for r in step["reactants_smiles"])
            reagents = ", ".join(step["reagents"])
            solvents = ", ".join(step["solvents"])

            lines.append(f"### Step {s_num}: {r_name}")
            lines.append(f"- **Reaction Class:** {step['reaction_class']}")
            lines.append(f"- **Reactants:** {reacts}")
            lines.append(f"- **Target Product for this step:** `{prod}`")
            lines.append(f"- **Reagents & Catalysts:** {reagents}")
            lines.append(f"- **Reaction Medium & Solvents:** {solvents}")
            lines.append(f"- **Temperature:** {step['temperature']}")
            lines.append(f"- **Atmosphere & Moisture Requirements:** {step['moisture_category']}")
            lines.append(f"- **Expected Step Yield:** {int(step['step_yield'] * 100)}% (Atom Economy: {step['atom_economy']}%)")
            lines.append(f"- **Workup & Purification Protocol:** {step['workup_protocol']}")

            if step.get("protection_plan"):
                pg = step["protection_plan"]
                lines.append(f"\n> **🛡️ Protecting Group Strategy ({pg['protecting_group']}):**")
                lines.append(f"> - Install: {pg['installation']['reagents']} in {pg['installation']['solvent']}")
                lines.append(f"> - Deprotect: {pg['deprotection']['reagents']} in {pg['deprotection']['solvent']}")

            if step.get("is_cascade") and step.get("cascade_note"):
                lines.append(f"\n> **⚡ One-Pot Telescoping Note:** {step['cascade_note']}")

            lines.append("\n")

        # Safety and hazards
        lines.append("## 3. Process Safety & Hazard Assessment\n")
        hazards = metrics.get("hazard_warnings", [])
        if hazards:
            for h in hazards:
                lines.append(f"- ⚠️ **Hazard Warning:** {h}")
        else:
            lines.append("- ✅ Standard laboratory hazards. Standard PPE and chemical fume hood required.")

        return "\n".join(lines)
