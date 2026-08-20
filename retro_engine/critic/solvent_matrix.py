"""Physical chemistry reaction medium, moisture sensitivity, and solvent condition predictor."""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from ..chem.reaction_rules import RetroReactionRule


@dataclass
class ReactionConditions:
    """Detailed physical and chemical operational conditions for a reaction step."""
    reagents: List[str]
    primary_solvent: str
    co_solvents: List[str]
    green_alternative_solvent: Optional[str]
    temperature_range: str
    atmosphere: str  # "Air / Open flask", "Inert N2 / Ar atmosphere", "Pressure vessel"
    moisture_category: str  # "Aqueous / Water-tolerant", "Moisture-sensitive", "Strictly Anhydrous"
    workup_protocol: str
    safety_notes: List[str]
    aqueous_compatibility_score: float  # 0.0 (destroyed by water) to 1.0 (requires/tolerates water)


class SolventConditionPredictor:
    """Predicts realistic wet-lab reaction conditions, solvents, and safety protocols."""

    # Solvent database with environmental & green replacement ratings
    SOLVENT_DATABASE = {
        "Water": {"green": True, "bp": 100, "toxicity": "None", "notes": "Universal green solvent"},
        "EtOH": {"green": True, "bp": 78, "toxicity": "Low", "notes": "Preferred protic green solvent"},
        "i-PrOH": {"green": True, "bp": 82, "toxicity": "Low", "notes": "Safe secondary alcohol solvent"},
        "EtOAc": {"green": True, "bp": 77, "toxicity": "Low", "notes": "Preferred ester green solvent"},
        "2-MeTHF": {"green": True, "bp": 80, "toxicity": "Low", "notes": "Bio-derived green replacement for THF/DCM"},
        "CPME": {"green": True, "bp": 106, "toxicity": "Low", "notes": "Green ether replacement for toxic ethers"},
        "DMF": {"green": False, "bp": 153, "toxicity": "Reprotoxic", "notes": "Polar aprotic; candidate for replacement with Cyrene/EtOAc"},
        "DCM": {"green": False, "bp": 40, "toxicity": "Suspected Carcinogen", "notes": "Chlorinated; replace with EtOAc or 2-MeTHF if possible"},
        "THF": {"green": False, "bp": 66, "toxicity": "Moderate", "notes": "Peroxide former; replace with 2-MeTHF"},
        "Toluene": {"green": False, "bp": 110, "toxicity": "Moderate", "notes": "Non-polar aromatic solvent"},
    }

    @classmethod
    def predict_conditions(cls, rule: RetroReactionRule) -> ReactionConditions:
        """Derive detailed laboratory conditions from the reaction rule and physical constraints."""
        reagents = list(rule.recommended_reagents)
        solvents = list(rule.recommended_solvents)
        primary_solv = solvents[0] if solvents else "DMF"
        co_solvs = solvents[1:] if len(solvents) > 1 else []

        # Determine green alternative
        green_alt = None
        if "DCM" in primary_solv or "DCM" in co_solvs:
            green_alt = "EtOAc or 2-MeTHF (Green alternative to DCM)"
        elif "DMF" in primary_solv or "DMF" in co_solvs:
            green_alt = "Cyrene or Propylene Carbonate (Green alternative to DMF)"
        elif "THF" in primary_solv:
            green_alt = "2-Methyltetrahydrofuran (2-MeTHF)"

        # Atmosphere & Moisture classification
        moist_cat = rule.moisture_sensitivity
        if moist_cat == "strictly_anhydrous":
            atmosphere = "Strictly Inert Atmosphere (Dry N2 or Ar gas; flame-dried glassware)"
            moisture_label = "Strictly Anhydrous (Water destroys reagent)"
            aqueous_score = 0.0
            workup = "Careful quench at 0°C with saturated aqueous NH4Cl; extract with EtOAc; dry over anhydrous Na2SO4."
            safety = [
                "Pyrophoric or moisture-reactive organometallic species involved.",
                "Ensure all syringes, needles, and solvents are anhydrous and degassed.",
            ]
        elif moist_cat == "moisture_sensitive":
            atmosphere = "Nitrogen balloon / Anhydrous conditions recommended"
            moisture_label = "Moisture-sensitive (Standard dry glassware)"
            aqueous_score = 0.3
            workup = "Wash organic layer with saturated aqueous NaHCO3 and brine; concentrate in vacuo."
            safety = ["Reagents generate acidic fumes (e.g. HCl) upon hydrolysis; perform in a fume hood."]
        elif moist_cat == "aqueous" or "H2O" in primary_solv or any("H2O" in s for s in co_solvs):
            atmosphere = "Ambient air / Open flask"
            moisture_label = "Aqueous / Water-tolerant"
            aqueous_score = 1.0
            workup = "Acidify/Basify aqueous layer; extract target into organic phase or filter precipitate directly."
            safety = ["Non-flammable aqueous medium; high process safety."]
        else:
            atmosphere = "Ambient air or Argon atmosphere"
            moisture_label = "Tolerant of ambient moisture"
            aqueous_score = 0.8
            workup = "Standard aqueous-organic partitioning (EtOAc/Water); wash with brine; column chromatography."
            safety = ["Standard laboratory PPE (gloves, safety goggles, chemical fume hood)."]

        return ReactionConditions(
            reagents=reagents,
            primary_solvent=primary_solv,
            co_solvents=co_solvs,
            green_alternative_solvent=green_alt,
            temperature_range=rule.temperature,
            atmosphere=atmosphere,
            moisture_category=moisture_label,
            workup_protocol=workup,
            safety_notes=safety,
            aqueous_compatibility_score=aqueous_score,
        )
