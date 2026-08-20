"""One-pot tandem and cascade reaction opportunity detector."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class CascadeGroup:
    """Group of adjacent synthetic steps that can be run sequentially in one pot."""
    group_id: str
    step_indices: List[int]
    common_solvent: str
    rationale: str
    time_and_waste_savings: str


class CascadeReactionDetector:
    """Identifies consecutive steps that can be telescoped into a one-pot cascade reaction."""

    COMPATIBLE_PAIRS = [
        ("RXN_NITRO_REDUCTION", "RXN_AMIDE_01", "In situ hydrogenation of nitro group followed immediately by acyl donor addition without isolating unstable aniline intermediate."),
        ("RXN_NITRO_REDUCTION", "RXN_PHENOL_ACYLATION", "Direct reduction followed by acetylation in the same reaction vessel (Telescoped Paracetamol protocol)."),
        ("RXN_SNAR_AROMATIC_AMINE", "RXN_AMIDE_01", "Sequential nucleophilic aromatic substitution and acylation in polar aprotic medium."),
        ("RXN_SUZUKI_01", "RXN_AMIDE_01", "One-pot cross-coupling and subsequent coupling in compatible ethereal/aqueous mixture."),
    ]

    def detect_cascades(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Annotate steps with cascade and one-pot operational instructions where applicable."""
        annotated = [dict(s) for s in steps]
        if len(annotated) < 2:
            return annotated

        for i in range(len(annotated) - 1):
            s1 = annotated[i]
            s2 = annotated[i + 1]

            rule1 = s1.get("rule_id", "")
            rule2 = s2.get("rule_id", "")

            # Check for known compatible pairs or common green solvents
            matched_pair = None
            for r_a, r_b, explanation in self.COMPATIBLE_PAIRS:
                if (rule1 == r_a and rule2 == r_b) or (
                    s1.get("is_cascade_compatible") and s2.get("is_cascade_compatible")
                ):
                    matched_pair = explanation
                    break

            if matched_pair:
                s1["is_cascade"] = True
                s1["cascade_note"] = (
                    f"One-Pot Telescoping Candidate with Step {i + 2}: {matched_pair}"
                )
                s2["is_cascade"] = True
                s2["cascade_note"] = (
                    f"Telescoped from Step {i + 1}: Reaction can proceed in the same pot without intermediate column chromatography."
                )

        return annotated
