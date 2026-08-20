"""Side-Product, Byproduct, and Flash Chromatography Purification Recommender."""

from typing import Dict, List, Any, Optional
from ..chem.mol_utils import identify_functional_groups, get_mol_from_smiles
from rdkit.Chem import Descriptors


class ByproductPurificationPredictor:
    """Predicts likely side-products and recommends chromatography separation protocols."""

    CLASS_PURIFICATION_RULES: Dict[str, Dict[str, Any]] = {
        "C-N Coupling": {
            "side_products": ["Trace unreacted carboxylic acid", "Coupling reagent urea byproduct (e.g. EDC-urea or DCU)", "Hydrolyzed ester intermediate"],
            "mobile_phase": "DCM / MeOH (95:5) + 0.1% Et3N or EtOAc / Hexanes (1:1)",
            "stationary_phase": "Silica Gel (40-63 µm)",
            "visualization": "UV 254 nm & Ninhydrin stain for primary/secondary amines",
            "separation_difficulty": "Easy to Moderate (Acid/base aqueous wash removes bulk urea)",
        },
        "Transition Metal C-C Coupling": {
            "side_products": ["Homocoupling biaryl dimer (Ar-Ar)", "Proto-deboronation byproduct (Ar-H)", "Unreacted aryl halide"],
            "mobile_phase": "Hexanes / EtOAc (9:1 to 4:1 gradient)",
            "stationary_phase": "Silica Gel (40-63 µm) with Celite filtration pad",
            "visualization": "UV 254 nm & Phosphomolybdic acid (PMA) stain",
            "separation_difficulty": "Moderate (Requires Celite pad to filter insoluble palladium black)",
        },
        "Electrophilic Aromatic Substitution": {
            "side_products": ["Ortho/Para regioisomeric mixture", "Di-substituted over-reaction byproducts", "Acidic fumes byproduct"],
            "mobile_phase": "Petroleum Ether / EtOAc (8:2) or Recrystallization from EtOH",
            "stationary_phase": "Silica Gel or direct Fractional Recrystallization",
            "visualization": "UV 254 nm & KMnO4 dip",
            "separation_difficulty": "Moderate to High (Regioisomers have close Rf values; recrystallization preferred)",
        },
        "Acylation": {
            "side_products": ["Acetic acid / Carboxylic acid byproduct", "Minor di-acylated species if secondary nucleophiles present"],
            "mobile_phase": "EtOAc / Hexanes (3:7) or Aqueous precipitation",
            "stationary_phase": "Silica Gel or Buchner suction filtration of precipitate",
            "visualization": "UV 254 nm & Iodine chamber",
            "separation_difficulty": "Easy (Product often crystallizes directly upon cooling/water addition)",
        },
        "Reduction": {
            "side_products": ["Partially reduced nitroso / hydroxylamine intermediates", "Trace oxidation byproducts upon air exposure"],
            "mobile_phase": "DCM / MeOH (9:1) or EtOAc / MeOH (95:5)",
            "stationary_phase": "Silica Gel pad (filtration through Celite)",
            "visualization": "UV 254 nm & Ninhydrin / Dragendorff stain",
            "separation_difficulty": "Easy (Filter through Celite to remove spent Pd/C catalyst)",
        },
    }

    @classmethod
    def forecast_purification(cls, reaction_class: str, product_smiles: str) -> Dict[str, Any]:
        """Generate side-product forecasts and bench purification instructions."""
        matched = cls.CLASS_PURIFICATION_RULES.get(
            reaction_class,
            {
                "side_products": ["Minor baseline degradation impurities", "Unreacted starting materials"],
                "mobile_phase": "Hexanes / EtOAc (7:3) or DCM / MeOH (98:2)",
                "stationary_phase": "Silica Gel (40-63 µm)",
                "visualization": "UV 254 nm & KMnO4 stain",
                "separation_difficulty": "Standard Flash Chromatography",
            }
        )

        mol = get_mol_from_smiles(product_smiles)
        logp = Descriptors.MolLogP(mol) if mol else 2.0
        
        # Refine mobile phase polarity based on product LogP
        mobile_phase = matched["mobile_phase"]
        if logp < 0.5:
            mobile_phase = "DCM / MeOH (9:1) + 0.1% AcOH (High Polarity System)"
        elif logp > 3.5:
            mobile_phase = "Hexanes / EtOAc (95:5) (Low Polarity Hydrocarbon System)"

        return {
            "expected_side_products": matched["side_products"],
            "recommended_eluent": mobile_phase,
            "stationary_phase": matched["stationary_phase"],
            "tlc_visualization": matched["visualization"],
            "separation_difficulty": matched["separation_difficulty"],
        }
