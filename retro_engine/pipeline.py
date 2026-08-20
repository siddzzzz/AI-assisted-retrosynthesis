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
from .critic.stoichiometry import ReactionStoichiometryCalculator
from .ml.inference import NeuralRetrosynthesisPolicy


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
        self.neural_policy = NeuralRetrosynthesisPolicy()

    def analyze_target(self, smiles: str) -> Dict[str, Any]:
        """Validate SMILES, compute properties, neural physics predictions, and render 2D chemical structure."""
        props = calculate_mol_properties(smiles)
        if not props.get("valid"):
            return props

        canonical_smi = props["smiles"]
        svg = render_mol_svg(canonical_smi, width=300, height=200, dark_mode=False)
        props["svg_light"] = svg
        props["svg_dark"] = svg

        # Neural Physics-Informed Predictions
        neural_preds = self.neural_policy.predict_disconnections(canonical_smi, top_k=3)
        props["neural_predictions"] = neural_preds

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

    def rescale_plan_stoichiometry(self, plan_dict: Dict[str, Any], target_scale_g: float = 1.0) -> Dict[str, Any]:
        """Recalculate batch stoichiometry tables across all steps for a new target scale."""
        updated_steps = []
        for step in plan_dict.get("steps", []):
            step_copy = dict(step)
            stoich = ReactionStoichiometryCalculator.calculate_step_stoichiometry(
                product_smiles=step["product_smiles"],
                reactants_smiles=step["reactants_smiles"],
                reagents=step["reagents"],
                solvents=step["solvents"],
                target_product_mass_g=target_scale_g,
                step_yield=step.get("step_yield", 0.85),
            )
            step_copy["stoichiometry"] = stoich
            updated_steps.append(step_copy)

        res = dict(plan_dict)
        res["steps"] = updated_steps
        return res

    def generate_laboratory_sop(self, plan_dict: Dict[str, Any], target_scale_g: float = 1.0) -> str:
        """Generate an Electronic Lab Notebook (ELN) Standard Operating Procedure with stoichiometry tables."""
        lines = []
        target_smi = plan_dict.get("target_smiles", "Target Molecule")
        metrics = plan_dict.get("metrics", {})
        steps = plan_dict.get("steps", [])

        lines.append(f"# Electronic Lab Notebook (ELN) Bench Protocol")
        lines.append(f"**Target Compound (SMILES):** `{target_smi}`  ")
        lines.append(f"**Target Batch Scale:** {target_scale_g} g | **Total Steps:** {metrics.get('total_steps', 0)} | **Overall Predicted Yield:** {metrics.get('cumulative_yield', 0)}%  ")
        lines.append(f"**Green Score:** {metrics.get('green_chemistry_score', 0)}/100 | **Estimated PMI:** {metrics.get('estimated_pmi', 0)} | **Avg. Atom Economy:** {metrics.get('average_atom_economy', 0)}%  ")
        lines.append("\n---\n")

        # 1. Starting Materials
        lines.append("## 1. Required Raw Material Feedstocks\n")
        lines.append("| Compound Name | SMILES | Category |")
        lines.append("|---|---|---|")
        for sm in plan_dict.get("starting_materials", []):
            lines.append(f"| {sm.get('name', 'Feedstock Material')} | `{sm.get('smiles', '')}` | {sm.get('category', 'Commodity Chemical')} |")

        lines.append("\n---\n")

        # 2. Step by Step execution with Stoichiometry Tables
        lines.append("## 2. Step-by-Step Reaction Protocols & Stoichiometry\n")
        for step in steps:
            s_num = step["step_number"]
            r_name = step["reaction_name"]
            prod = step["product_smiles"]
            reagents = ", ".join(step["reagents"])
            solvents = ", ".join(step["solvents"])
            sterics = step.get("sterics", {})
            precedent = step.get("precedent", {})
            purif = step.get("purification", {})
            stoich = step.get("stoichiometry", {})

            lines.append(f"### Step {s_num}: {r_name}")
            lines.append(f"- **Reaction Class:** {step['reaction_class']}")
            lines.append(f"- **Target Product for this step:** `{prod}`")
            lines.append(f"- **Operational Conditions:** {step['temperature']} in {solvents}")
            lines.append(f"- **Atmosphere & Moisture:** {step['moisture_category']}")
            lines.append(f"- **Expected Step Yield:** {int(step['step_yield'] * 100)}% (Atom Economy: {step['atom_economy']}%)")
            
            # Literature Precedent Citation
            if precedent.get("citation"):
                lines.append(f"- 📚 **Literature Precedent:** {precedent['citation']}")

            # 3D Sterics
            if sterics.get("accessibility"):
                lines.append(f"- 🧊 **3D Steric Assessment:** {sterics['accessibility']} (Steric Burial Score: {sterics.get('steric_burial_score', 0)}%) — *{sterics.get('bench_advice', '')}*")

            # Stoichiometry Table
            lines.append("\n#### Stoichiometric Quantities (Target: " + str(target_scale_g) + " g Product):")
            lines.append("| Component | Role | MW (g/mol) | Eq. | mmol | Mass (g / mg) | Vol (mL) |")
            lines.append("|---|---|---|---|---|---|---|")
            
            for item in stoich.get("stoichiometry_table", []):
                mass_disp = f"{item['mass_g']} g" if item['mass_g'] >= 1.0 else f"{item['mass_mg']} mg"
                vol_disp = f"{item['volume_ml']} mL" if item.get('volume_ml') else "—"
                name_disp = item['name'] if item['name'] else item['smiles']
                lines.append(f"| {name_disp} | {item['role']} | {item['mw']} | {item['equivalents']} | {item['mmol']} | {mass_disp} | {vol_disp} |")

            lines.append(f"\n- **Solvent Volume:** {stoich.get('recommended_solvent_volume_ml', 20.0)} mL ({stoich.get('primary_solvent', 'Solvent')}) for ~{stoich.get('concentration_molar', 0.2)} M reaction concentration.")

            # Workup & Flash Chromatography
            lines.append(f"\n- **Workup Procedure:** {step['workup_protocol']}")
            if purif.get("recommended_eluent"):
                lines.append(f"- **Flash Chromatography:** Silica Gel column with eluent `{purif['recommended_eluent']}` ({purif.get('tlc_visualization', 'UV 254 nm')}).")
                if purif.get("expected_side_products"):
                    lines.append(f"- **Watch for Byproducts:** {', '.join(purif['expected_side_products'])}")

            if step.get("protection_plan"):
                pg = step["protection_plan"]
                lines.append(f"\n> **🛡️ Protecting Group Sequence ({pg['protecting_group']}):**")
                lines.append(f"> - Protection: {pg['installation']['reagents']} in {pg['installation']['solvent']} ({pg['installation']['yield']})")
                lines.append(f"> - Deprotection: {pg['deprotection']['reagents']} in {pg['deprotection']['solvent']} ({pg['deprotection']['yield']})")

            if step.get("is_cascade") and step.get("cascade_note"):
                lines.append(f"\n> **⚡ One-Pot Telescoping Candidate:** {step['cascade_note']}")

            lines.append("\n")

        # 3. Safety & Hazards
        lines.append("## 3. Laboratory Safety & Hazard Assessment\n")
        hazards = metrics.get("hazard_warnings", [])
        if hazards:
            for h in hazards:
                lines.append(f"- ⚠️ **Hazard Alert:** {h}")
        else:
            lines.append("- ✅ Standard organic chemistry hazards. Standard PPE and certified fume hood required.")

        return "\n".join(lines)
