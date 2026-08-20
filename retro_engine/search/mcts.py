"""Multi-Objective Retrosynthesis Tree Search (MCTS / Best-First Heuristic Search).
Performs backward retrosynthetic disconnections down to fundamental commodity feedstocks.
"""

from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
import heapq
import time
from rdkit import Chem

from ..chem.mol_utils import canonicalize_smiles, calculate_atom_economy, get_mol_from_smiles
from ..chem.reaction_rules import ReactionRuleLibrary, RetroReactionRule, apply_retro_rule
from ..chem.building_blocks import BuildingBlockCatalog, get_default_catalog
from ..critic.chemoselectivity import ChemoselectivityAnalyzer, ConflictReport
from ..critic.protecting_groups import ProtectingGroupPlanner
from ..critic.solvent_matrix import SolventConditionPredictor, ReactionConditions
from ..critic.green_metrics import GreenChemistryEvaluator, RouteMetrics
from .cascade_detector import CascadeReactionDetector


@dataclass
class ReactionStepNode:
    """Represents an instantiated synthetic reaction step within a route."""
    step_number: int
    rule_id: str
    reaction_name: str
    reaction_class: str
    product_smiles: str
    reactants_smiles: List[str]
    reagents: List[str]
    solvents: List[str]
    temperature: str
    moisture_category: str
    workup_protocol: str
    step_yield: float
    atom_economy: float
    confidence_score: float
    conflict_report: Dict[str, Any]
    protection_plan: Optional[Dict[str, Any]]
    explanation: str
    is_cascade: bool = False
    cascade_note: Optional[str] = None


@dataclass
class SynthesisPlan:
    """A complete, validated multi-step synthetic pathway from starting materials to target."""
    plan_id: str
    target_smiles: str
    steps: List[ReactionStepNode]
    starting_materials: List[Dict[str, Any]]
    metrics: RouteMetrics
    is_solved: bool
    execution_time_sec: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize plan for API responses and UI rendering."""
        return {
            "plan_id": self.plan_id,
            "target_smiles": self.target_smiles,
            "is_solved": self.is_solved,
            "execution_time_sec": round(self.execution_time_sec, 3),
            "metrics": {
                "total_steps": self.metrics.total_steps,
                "cumulative_yield": self.metrics.cumulative_yield,
                "average_atom_economy": self.metrics.average_atom_economy,
                "estimated_pmi": self.metrics.estimated_pmi,
                "estimated_e_factor": self.metrics.estimated_e_factor,
                "green_chemistry_score": self.metrics.green_chemistry_score,
                "hazard_score": self.metrics.hazard_score,
                "hazard_warnings": self.metrics.hazard_warnings,
                "pareto_tags": self.metrics.pareto_tags,
            },
            "starting_materials": self.starting_materials,
            "steps": [
                {
                    "step_number": s.step_number,
                    "rule_id": s.rule_id,
                    "reaction_name": s.reaction_name,
                    "reaction_class": s.reaction_class,
                    "product_smiles": s.product_smiles,
                    "reactants_smiles": s.reactants_smiles,
                    "reagents": s.reagents,
                    "solvents": s.solvents,
                    "temperature": s.temperature,
                    "moisture_category": s.moisture_category,
                    "workup_protocol": s.workup_protocol,
                    "step_yield": round(s.step_yield, 2),
                    "atom_economy": round(s.atom_economy, 1),
                    "confidence_score": round(s.confidence_score, 2),
                    "conflict_report": s.conflict_report,
                    "protection_plan": s.protection_plan,
                    "explanation": s.explanation,
                    "is_cascade": s.is_cascade,
                    "cascade_note": s.cascade_note,
                }
                for s in self.steps
            ],
        }


class RetrosynthesisSearchTree:
    """Multi-Objective Retrosynthetic Route Planner utilizing Best-First Tree Search and System 2 Critic."""

    def __init__(
        self,
        rule_library: Optional[ReactionRuleLibrary] = None,
        catalog: Optional[BuildingBlockCatalog] = None,
    ):
        self.rule_library = rule_library or ReactionRuleLibrary()
        self.catalog = catalog or get_default_catalog()

    def plan_synthesis(
        self,
        target_smiles: str,
        max_depth: int = 5,
        max_routes: int = 6,
        time_limit_sec: float = 12.0,
    ) -> List[SynthesisPlan]:
        """Search for valid multi-step synthetic routes to basic chemical starting materials."""
        start_time = time.time()
        canonical_target = canonicalize_smiles(target_smiles)
        if not canonical_target:
            return []

        # Recursive Best-First Route Search without short-circuiting the target
        raw_routes: List[List[Dict[str, Any]]] = []
        rules = self.rule_library.get_all_rules()

        def search_mol(
            current_smi: str,
            current_depth: int,
            visited_in_branch: Set[str],
        ) -> List[List[Dict[str, Any]]]:
            """Recursively find all valid sub-pathways for current_smi."""
            # Terminating leaf check (only if not at the root depth)
            if current_depth > 1 and self.catalog.is_fundamental_starting_material(current_smi):
                return [[]]

            if current_depth > max_depth or (time.time() - start_time) > time_limit_sec:
                # If reached max depth and it's small, accept as terminating node
                mol = get_mol_from_smiles(current_smi)
                if current_depth > 1 and mol and mol.GetNumHeavyAtoms() <= 8:
                    return [[]]
                return []

            if current_smi in visited_in_branch:
                # Cycle prevention
                return []

            mol = get_mol_from_smiles(current_smi)
            if mol is None:
                return []

            sub_pathways: List[List[Dict[str, Any]]] = []

            for rule in rules:
                disconnections = apply_retro_rule(mol, rule)
                for reactants_smi, matched_rule in disconnections:
                    # Evaluate System 2 Critic
                    conflict = ChemoselectivityAnalyzer.evaluate_step(
                        reactants_smi, current_smi, matched_rule
                    )
                    conditions = SolventConditionPredictor.predict_conditions(matched_rule)
                    ae = calculate_atom_economy(reactants_smi, current_smi)

                    # Protection group handling
                    protection_info = None
                    effective_yield = matched_rule.typical_yield
                    if conflict.needs_protection and conflict.suggested_protecting_group:
                        protection_info = ProtectingGroupPlanner.generate_strategy_report(
                            conflicting_group=", ".join(conflict.conflicting_groups),
                            suggested_pg=conflict.suggested_protecting_group,
                        )
                        effective_yield *= 0.90

                    base_confidence = max(0.20, 1.0 - conflict.confidence_penalty)

                    step_dict = {
                        "rule_id": matched_rule.rule_id,
                        "reaction_name": matched_rule.name,
                        "reaction_class": matched_rule.reaction_class,
                        "product_smiles": current_smi,
                        "reactants_smiles": reactants_smi,
                        "reagents": conditions.reagents,
                        "solvents": [conditions.primary_solvent] + conditions.co_solvents,
                        "temperature": conditions.temperature_range,
                        "moisture_category": conditions.moisture_category,
                        "workup_protocol": conditions.workup_protocol,
                        "step_yield": effective_yield,
                        "atom_economy": ae,
                        "confidence_score": base_confidence,
                        "conflict_report": {
                            "has_conflict": conflict.has_conflict,
                            "severity": conflict.severity,
                            "explanation": conflict.explanation,
                        },
                        "protection_plan": protection_info,
                        "explanation": matched_rule.explanation,
                        "is_cascade_compatible": matched_rule.is_cascade_compatible,
                    }

                    # Recurse on all reactants
                    all_reactants_solved = True
                    branch_visited = visited_in_branch | {current_smi}
                    reactant_solutions: List[List[List[Dict[str, Any]]]] = []

                    for r_smi in reactants_smi:
                        r_solutions = search_mol(r_smi, current_depth + 1, branch_visited)
                        if not r_solutions:
                            all_reactants_solved = False
                            break
                        reactant_solutions.append(r_solutions)

                    if all_reactants_solved:
                        # Combine sub-solutions across all reactants
                        combined_sub_steps = self._cartesian_combine_steps(reactant_solutions)
                        for prev_steps in combined_sub_steps:
                            full_route = prev_steps + [step_dict]
                            sub_pathways.append(full_route)

            return sub_pathways

        raw_routes = search_mol(canonical_target, current_depth=1, visited_in_branch=set())
        
        # Deduplicate and instantiate completed plans
        dedup_plans: List[SynthesisPlan] = []
        seen_signatures: Set[str] = set()

        for idx, route_steps in enumerate(raw_routes):
            if not route_steps:
                continue

            # Build signature from reaction names & products
            sig = " -> ".join(f"{s['reaction_name']}:{s['product_smiles']}" for s in route_steps)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)

            # Analyze for cascade/one-pot optimizations
            cascade_detector = CascadeReactionDetector()
            annotated_steps = cascade_detector.detect_cascades(route_steps)

            # Build ReactionStepNode instances
            step_nodes = []
            for s_num, s_data in enumerate(annotated_steps, start=1):
                node = ReactionStepNode(
                    step_number=s_num,
                    rule_id=s_data["rule_id"],
                    reaction_name=s_data["reaction_name"],
                    reaction_class=s_data["reaction_class"],
                    product_smiles=s_data["product_smiles"],
                    reactants_smiles=s_data["reactants_smiles"],
                    reagents=s_data["reagents"],
                    solvents=s_data["solvents"],
                    temperature=s_data["temperature"],
                    moisture_category=s_data["moisture_category"],
                    workup_protocol=s_data["workup_protocol"],
                    step_yield=s_data["step_yield"],
                    atom_economy=s_data["atom_economy"],
                    confidence_score=s_data["confidence_score"],
                    conflict_report=s_data["conflict_report"],
                    protection_plan=s_data["protection_plan"],
                    explanation=s_data["explanation"],
                    is_cascade=s_data.get("is_cascade", False),
                    cascade_note=s_data.get("cascade_note", None),
                )
                step_nodes.append(node)

            # Identify leaf starting materials
            all_products = {s.product_smiles for s in step_nodes}
            leaf_reactants = set()
            for s in step_nodes:
                for r in s.reactants_smiles:
                    if r not in all_products:
                        leaf_reactants.add(r)

            starting_mats = []
            for r_smi in leaf_reactants:
                info = self.catalog.get_compound_info(r_smi)
                if info:
                    starting_mats.append(info)
                else:
                    starting_mats.append({
                        "smiles": r_smi,
                        "name": "Basic Raw Material",
                        "category": "Commodity Feedstock",
                    })

            # Compute route metrics
            metrics = GreenChemistryEvaluator.evaluate_route(
                steps=[s.__dict__ for s in step_nodes],
            )

            plan = SynthesisPlan(
                plan_id=f"ROUTE_{len(dedup_plans) + 1:02d}",
                target_smiles=canonical_target,
                steps=step_nodes,
                starting_materials=starting_mats,
                metrics=metrics,
                is_solved=True,
                execution_time_sec=time.time() - start_time,
            )
            dedup_plans.append(plan)

        # Sort plans: prioritize diverse multi-step pathways with high green score & cumulative yield
        dedup_plans.sort(
            key=lambda p: (
                p.metrics.total_steps * 5.0  # Encourage rich complete multi-step routes
                + p.metrics.green_chemistry_score * 0.4
                + p.metrics.cumulative_yield * 0.3
            ),
            reverse=True,
        )

        return dedup_plans[:max_routes]

    def _cartesian_combine_steps(
        self, reactant_solutions: List[List[List[Dict[str, Any]]]]
    ) -> List[List[Dict[str, Any]]]:
        """Combine step lists from independent reactant branches."""
        if not reactant_solutions:
            return [[]]

        combined = [[]]
        for r_sol_list in reactant_solutions:
            new_combined = []
            for base in combined:
                for sol in r_sol_list:
                    merged = list(base)
                    for step in sol:
                        if step not in merged:
                            merged.append(step)
                    new_combined.append(merged)
            combined = new_combined
        return combined
