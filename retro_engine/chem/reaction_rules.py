"""Curated organic reaction rules, SMARTS transformations, and retrosynthetic disconnection templates.
Enriched with foundational petrochemical & functionalization stages for deep multi-step planning.
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from .mol_utils import canonicalize_smiles, get_mol_from_smiles


@dataclass
class RetroReactionRule:
    """Represents a single retrosynthetic transformation rule with physical & laboratory context."""
    rule_id: str
    name: str
    reaction_class: str
    retro_smarts: str  # Product >> Reactant1.Reactant2
    recommended_reagents: List[str]
    recommended_solvents: List[str]
    temperature: str
    moisture_sensitivity: str  # "tolerant", "moisture_sensitive", "strictly_anhydrous", "aqueous"
    typical_yield: float
    atom_economy: float
    incompatible_groups: List[str]
    protection_needed_for: Dict[str, str] = field(default_factory=dict)
    explanation: str = ""
    is_cascade_compatible: bool = False
    
    _compiled_rxn: Optional[Any] = field(default=None, repr=False)

    def get_reaction(self) -> Optional[rdChemReactions.ChemicalReaction]:
        """Lazy-compile and return the RDKit ChemicalReaction object."""
        if self._compiled_rxn is None:
            try:
                rxn = rdChemReactions.ReactionFromSmarts(self.retro_smarts)
                if rxn is not None:
                    rxn.Initialize()
                    self._compiled_rxn = rxn
            except Exception:
                self._compiled_rxn = None
        return self._compiled_rxn


class ReactionRuleLibrary:
    """Library of validated organic synthesis and retrosynthesis disconnection templates."""

    def __init__(self):
        self.rules: List[RetroReactionRule] = []
        self._load_default_rules()

    def _load_default_rules(self):
        """Populate the library with diverse, multi-stage organic chemical transformations."""
        
        # -------------------------------------------------------------
        # 1. AMIDE, PEPTIDE & SULFONAMIDE FORMATIONS
        # -------------------------------------------------------------
        self.add_rule(RetroReactionRule(
            rule_id="RXN_AMIDE_01",
            name="Amide Bond Disconnection (Carboxylic Acid + Amine)",
            reaction_class="C-N Coupling",
            retro_smarts="[C:1](=[O:2])[N:3]>>[C:1](=[O:2])O.[N:3]",
            recommended_reagents=["HATU, DIPEA", "EDC·HCl, HOBt, Et3N", "CDI", "PyBOP"],
            recommended_solvents=["DMF", "DCM", "THF"],
            temperature="0°C to rt (20-25°C)",
            moisture_sensitivity="tolerant",
            typical_yield=0.90,
            atom_economy=82.0,
            incompatible_groups=["acid_chloride"],
            protection_needed_for={"primary_amine": "boc", "secondary_amine": "boc"},
            explanation="Activation of carboxylic acid with peptide coupling agents followed by nucleophilic attack of primary/secondary amine.",
            is_cascade_compatible=True,
        ))

        self.add_rule(RetroReactionRule(
            rule_id="RXN_AMIDE_ACID_CHLORIDE",
            name="Schotten-Baumann Amide Formation (Acid Chloride + Amine)",
            reaction_class="C-N Coupling",
            retro_smarts="[C:1](=[O:2])[N:3]>>[C:1](=[O:2])Cl.[N:3]",
            recommended_reagents=["Et3N or Pyridine", "DMAP (cat.)", "K2CO3 (aq)"],
            recommended_solvents=["DCM", "THF", "Toluene / H2O biphasic"],
            temperature="0°C to rt",
            moisture_sensitivity="moisture_sensitive",
            typical_yield=0.94,
            atom_economy=85.0,
            incompatible_groups=["alcohol", "phenol", "thiol"],
            protection_needed_for={"alcohol": "tbs"},
            explanation="Rapid nucleophilic acyl substitution on an acid chloride with tertiary amine base as HCl scavenger.",
            is_cascade_compatible=True,
        ))

        self.add_rule(RetroReactionRule(
            rule_id="RXN_ACYL_CHLORIDE_FORMATION",
            name="Acyl Chloride Preparation from Carboxylic Acid",
            reaction_class="Functional Group Interconversion",
            retro_smarts="[C:1](=[O:2])Cl>>[C:1](=[O:2])O",
            recommended_reagents=["Thionyl chloride (SOCl2, 1.2 eq)", "Oxalyl chloride (COCl)2, DMF (cat.)"],
            recommended_solvents=["DCM", "Toluene", "Neat"],
            temperature="rt to 60°C",
            moisture_sensitivity="strictly_anhydrous",
            typical_yield=0.95,
            atom_economy=78.0,
            incompatible_groups=["primary_amine", "alcohol"],
            explanation="Conversion of carboxylic acid to highly reactive acid chloride using SOCl2 or oxalyl chloride.",
        ))

        self.add_rule(RetroReactionRule(
            rule_id="RXN_SULFONAMIDE_01",
            name="Sulfonamide Formation (Sulfonyl Chloride + Amine)",
            reaction_class="S-N Coupling",
            retro_smarts="[S:1](=[O:2])(=[O:3])[N:4]>>[S:1](=[O:2])(=[O:3])Cl.[N:4]",
            recommended_reagents=["Pyridine", "Et3N, DMAP (cat.)", "DIPEA"],
            recommended_solvents=["DCM", "MeCN", "Pyridine"],
            temperature="0°C to rt",
            moisture_sensitivity="moisture_sensitive",
            typical_yield=0.91,
            atom_economy=84.0,
            incompatible_groups=["alcohol"],
            protection_needed_for={"alcohol": "tbs"},
            explanation="Nucleophilic attack of amine onto sulfonyl chloride; key step in antibacterial and celecoxib-type sulfonamide drugs.",
        ))

        # -------------------------------------------------------------
        # 2. ESTERIFICATION & PHENOL ACYLATION
        # -------------------------------------------------------------
        self.add_rule(RetroReactionRule(
            rule_id="RXN_ESTER_01",
            name="Ester Disconnection (Carboxylic Acid + Alcohol)",
            reaction_class="Acylation",
            retro_smarts="[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])O.[C:4][O:3]",
            recommended_reagents=["EDC·HCl, DMAP", "DCC, DMAP (Steglich)", "H2SO4 (cat., Fischer)"],
            recommended_solvents=["DCM", "THF", "Toluene (reflux)"],
            temperature="rt to 80°C",
            moisture_sensitivity="tolerant",
            typical_yield=0.88,
            atom_economy=88.0,
            incompatible_groups=["primary_amine", "secondary_amine"],
            protection_needed_for={"primary_amine": "boc"},
            explanation="Coupling of carboxylic acid with an alcohol via carbodiimide activation or acid catalysis.",
        ))

        self.add_rule(RetroReactionRule(
            rule_id="RXN_PHENOL_ACYLATION",
            name="Phenol O-Acylation / Esterification (e.g. Aspirin Synthesis)",
            reaction_class="Acylation",
            retro_smarts="[c:1][O:2][C:3](=[O:4])[C:5]>>[c:1][O:2].[C:5][C:3](=[O:4])O",
            recommended_reagents=["Acetic anhydride (Ac2O), H3PO4 or H2SO4 (cat.)", "Acetyl chloride, Pyridine"],
            recommended_solvents=["EtOAc", "DCM", "Neat (with catalytic acid)"],
            temperature="50°C to 80°C",
            moisture_sensitivity="tolerant",
            typical_yield=0.95,
            atom_economy=86.0,
            incompatible_groups=["primary_amine"],
            explanation="Acid-catalyzed O-acylation of phenolic hydroxyl group by acetic anhydride or acetyl chloride.",
            is_cascade_compatible=True,
        ))

        # -------------------------------------------------------------
        # 3. NOBEL PRIZE C-C CROSS-COUPLINGS
        # -------------------------------------------------------------
        self.add_rule(RetroReactionRule(
            rule_id="RXN_SUZUKI_01",
            name="Suzuki-Miyaura Cross-Coupling (Aryl Halide + Boronic Acid)",
            reaction_class="Transition Metal C-C Coupling",
            retro_smarts="[c:1]-[c:2]>>[c:1]Br.[c:2]B(O)O",
            recommended_reagents=["Pd(PPh3)4 or Pd(dppf)Cl2 (2-5 mol%)", "K2CO3 or Cs2CO3 (2.0 eq)"],
            recommended_solvents=["1,4-Dioxane / H2O (4:1)", "DMF / H2O", "Toluene / EtOH / H2O"],
            temperature="80°C to 100°C",
            moisture_sensitivity="tolerant",
            typical_yield=0.89,
            atom_economy=78.0,
            incompatible_groups=["acid_chloride", "alkyl_halide"],
            explanation="Palladium-catalyzed cross-coupling of an aryl halide with an organoboronic acid in aqueous-organic media.",
            is_cascade_compatible=True,
        ))

        self.add_rule(RetroReactionRule(
            rule_id="RXN_SONOGASHIRA_01",
            name="Sonogashira Cross-Coupling (Aryl Halide + Terminal Alkyne)",
            reaction_class="Transition Metal C-C Coupling",
            retro_smarts="[c:1]-[C:2]#[C:3]>>[c:1]Br.[C:2]#[C:3]",
            recommended_reagents=["Pd(PPh3)2Cl2 (2 mol%)", "CuI (4 mol%)", "Et3N or DIPEA"],
            recommended_solvents=["THF", "DMF", "MeCN"],
            temperature="rt to 60°C (under N2)",
            moisture_sensitivity="strictly_anhydrous",
            typical_yield=0.85,
            atom_economy=89.0,
            incompatible_groups=[],
            explanation="Pd/Cu co-catalyzed cross-coupling of terminal alkynes with aryl halides.",
        ))

        self.add_rule(RetroReactionRule(
            rule_id="RXN_HECK_01",
            name="Heck Olefin Cross-Coupling (Aryl Halide + Alkene)",
            reaction_class="Transition Metal C-C Coupling",
            retro_smarts="[c:1]-[C:2]=[C:3]>>[c:1]Br.[C:2]=[C:3]",
            recommended_reagents=["Pd(OAc)2, P(o-tolyl)3", "Et3N or K2CO3"],
            recommended_solvents=["DMF", "MeCN", "NMP"],
            temperature="100°C to 120°C",
            moisture_sensitivity="moisture_sensitive",
            typical_yield=0.82,
            atom_economy=85.0,
            incompatible_groups=["acid_chloride"],
            explanation="Palladium-catalyzed coupling of aryl halides with terminal or substituted alkenes.",
        ))

        self.add_rule(RetroReactionRule(
            rule_id="RXN_BUCHWALD_HARTWIG",
            name="Buchwald-Hartwig C-N Amination (Aryl Halide + Amine)",
            reaction_class="Transition Metal C-N Coupling",
            retro_smarts="[c:1]-[N:2]>>[c:1]Br.[N:2]",
            recommended_reagents=["Pd2(dba)3, BINAP or BrettPhos", "NaOtBu or Cs2CO3"],
            recommended_solvents=["Toluene", "1,4-Dioxane", "t-BuOH"],
            temperature="80°C to 110°C (under N2)",
            moisture_sensitivity="strictly_anhydrous",
            typical_yield=0.84,
            atom_economy=86.0,
            incompatible_groups=["carboxylic_acid", "alcohol"],
            protection_needed_for={"alcohol": "tbs", "carboxylic_acid": "ester"},
            explanation="Palladium-catalyzed coupling of aryl halides with amines using sterically bulky phosphine ligands.",
        ))

        # -------------------------------------------------------------
        # 4. REDUCTIONS & REDUCTIVE AMINATIONS
        # -------------------------------------------------------------
        self.add_rule(RetroReactionRule(
            rule_id="RXN_NITRO_REDUCTION",
            name="Nitro Reduction to Primary Aromatic Amine",
            reaction_class="Reduction",
            retro_smarts="[c:1][NX3;H2]>>[c:1][N+](=O)[O-]",
            recommended_reagents=["H2 (1-3 atm), 10% Pd/C", "Fe powder, NH4Cl (aq)", "SnCl2·2H2O in EtOH", "Zn, NH4COOH"],
            recommended_solvents=["MeOH", "EtOH / H2O", "EtOAc"],
            temperature="rt to 70°C",
            moisture_sensitivity="tolerant",
            typical_yield=0.95,
            atom_economy=92.0,
            incompatible_groups=[],
            explanation="Catalytic hydrogenation or metal-mediated reduction of aromatic nitro group to aniline.",
            is_cascade_compatible=True,
        ))

        self.add_rule(RetroReactionRule(
            rule_id="RXN_REDUCTIVE_AMINATION",
            name="Reductive Amination (Carbonyl + Amine)",
            reaction_class="Reductive Coupling",
            retro_smarts="[C:1]-[N:2]>>[C:1]=O.[N:2]",
            recommended_reagents=["NaBH(OAc)3, AcOH (cat.)", "NaBH3CN", "H2 / Pd-C"],
            recommended_solvents=["1,2-DCE", "MeOH", "THF", "DCM"],
            temperature="rt to 40°C",
            moisture_sensitivity="tolerant",
            typical_yield=0.87,
            atom_economy=91.0,
            incompatible_groups=["acid_chloride"],
            explanation="In situ condensation of carbonyl and amine followed by selective hydride reduction.",
            is_cascade_compatible=True,
        ))

        self.add_rule(RetroReactionRule(
            rule_id="RXN_KETONE_REDUCTION",
            name="Ketone / Aldehyde Reduction to Alcohol",
            reaction_class="Reduction",
            retro_smarts="[C:1][C:2]([OH])[C:3]>>[C:1][C:2](=O)[C:3]",
            recommended_reagents=["NaBH4 (1.5 eq)", "LiAlH4 in dry THF", "CBS oxazaborolidine catalyst"],
            recommended_solvents=["MeOH or EtOH", "THF"],
            temperature="0°C to rt",
            moisture_sensitivity="tolerant",
            typical_yield=0.96,
            atom_economy=98.0,
            incompatible_groups=[],
            explanation="Hydride reduction of ketone or aldehyde to alcohol with near 100% atom efficiency.",
        ))

        # -------------------------------------------------------------
        # 5. ELECTROPHILIC AROMATIC SUBSTITUTIONS & BASIC FEEDSTOCKS
        # -------------------------------------------------------------
        self.add_rule(RetroReactionRule(
            rule_id="RXN_AROMATIC_NITRATION",
            name="Electrophilic Aromatic Nitration (Arene -> Nitroarene)",
            reaction_class="Electrophilic Aromatic Substitution",
            retro_smarts="[c:1][N+](=O)[O-]>>[c:1]",
            recommended_reagents=["HNO3 (65%) / H2SO4 (conc., 1:1 mixture, nitrating acid)"],
            recommended_solvents=["H2SO4 (solvent/catalyst)", "DCM", "Neat"],
            temperature="0°C to 45°C",
            moisture_sensitivity="tolerant",
            typical_yield=0.92,
            atom_economy=82.0,
            incompatible_groups=["primary_amine"],
            explanation="Generation of nitronium ion (NO2+) followed by electrophilic aromatic substitution on benzene/phenol derivatives.",
            is_cascade_compatible=True,
        ))

        self.add_rule(RetroReactionRule(
            rule_id="RXN_AROMATIC_BROMINATION",
            name="Electrophilic Aromatic Bromination (Arene -> Bromoarene)",
            reaction_class="Electrophilic Aromatic Substitution",
            retro_smarts="[c:1]Br>>[c:1]",
            recommended_reagents=["Br2 (1.0 eq), FeBr3 or AlCl3 (cat.)", "NBS (N-Bromosuccinimide) in MeCN"],
            recommended_solvents=["DCM", "MeCN", "AcOH"],
            temperature="0°C to rt",
            moisture_sensitivity="moisture_sensitive",
            typical_yield=0.91,
            atom_economy=75.0,
            incompatible_groups=[],
            explanation="Halogenation of aromatic rings to introduce bromine coupling handles for cross-couplings.",
        ))

        self.add_rule(RetroReactionRule(
            rule_id="RXN_FRIEDEL_CRAFTS_ACYLATION",
            name="Friedel-Crafts Acylation (Arene + Acid Chloride -> Ketone)",
            reaction_class="Electrophilic Aromatic Substitution",
            retro_smarts="[c:1][C:2](=[O:3])[#6:4]>>[c:1].[#6:4][C:2](=[O:3])Cl",
            recommended_reagents=["AlCl3 (1.2 eq)", "FeCl3", "TfOH"],
            recommended_solvents=["DCM", "1,2-DCE", "Neat"],
            temperature="0°C to 40°C",
            moisture_sensitivity="strictly_anhydrous",
            typical_yield=0.88,
            atom_economy=82.0,
            incompatible_groups=["primary_amine"],
            protection_needed_for={"primary_amine": "boc"},
            explanation="Lewis acid-catalyzed electrophilic aromatic substitution installing acyl ketone group (e.g. 4-isobutylacetophenone in Ibuprofen route).",
        ))

        self.add_rule(RetroReactionRule(
            rule_id="RXN_FRIEDEL_CRAFTS_ALKYLATION",
            name="Friedel-Crafts Alkylation (Arene + Alkyl Halide)",
            reaction_class="Electrophilic Aromatic Substitution",
            retro_smarts="[c:1][C:2]([#6:3])[#6:4]>>[c:1].[C:2]([#6:3])([#6:4])Cl",
            recommended_reagents=["AlCl3 (cat., 0.1 eq)", "BF3·OEt2", "Zeolite beta"],
            recommended_solvents=["DCM", "Hexane", "Neat arene"],
            temperature="0°C to 50°C",
            moisture_sensitivity="strictly_anhydrous",
            typical_yield=0.85,
            atom_economy=80.0,
            incompatible_groups=["primary_amine"],
            explanation="Alkylation of aromatic ring with alkyl halide using Lewis acid catalyst.",
        ))

        self.add_rule(RetroReactionRule(
            rule_id="RXN_KOLBE_SCHMITT",
            name="Kolbe-Schmitt Carboxylation (Phenol -> Salicylic Acid)",
            reaction_class="Carboxylation",
            retro_smarts="[c:1]([O:2])[C:3](=[O:4])O>>[c:1][O:2]",
            recommended_reagents=["CO2 (100 atm), NaOH / KOH", "K2CO3 in DMF (microwave-assisted)"],
            recommended_solvents=["Aqueous / Autoclave", "DMF"],
            temperature="125°C to 150°C (pressure)",
            moisture_sensitivity="tolerant",
            typical_yield=0.88,
            atom_economy=95.0,
            incompatible_groups=[],
            explanation="Industrial carboxylation of sodium phenolate with carbon dioxide to synthesize salicylic acid (Aspirin precursor).",
            is_cascade_compatible=True,
        ))

        # -------------------------------------------------------------
        # 6. DIAZOTIZATION & SANDMEYER TRANSFORMATIONS
        # -------------------------------------------------------------
        self.add_rule(RetroReactionRule(
            rule_id="RXN_SANDMEYER_HALOGENATION",
            name="Sandmeyer Reaction (Aromatic Amine -> Aryl Bromide)",
            reaction_class="Diazotization & Halogenation",
            retro_smarts="[c:1]Br>>[c:1][NX3;H2]",
            recommended_reagents=["NaNO2 (1.1 eq), HBr (48% aq), CuBr (1.0 eq)"],
            recommended_solvents=["H2O / HBr (aqueous)"],
            temperature="0°C to 60°C",
            moisture_sensitivity="aqueous",
            typical_yield=0.86,
            atom_economy=72.0,
            incompatible_groups=[],
            explanation="Conversion of aniline to diazonium salt followed by copper-catalyzed displacement with bromide.",
        ))

        self.add_rule(RetroReactionRule(
            rule_id="RXN_DIAZONIUM_HYDROLYSIS",
            name="Diazonium Hydrolysis to Phenol (Aniline -> Phenol)",
            reaction_class="Hydrolysis",
            retro_smarts="[c:1][OH]>>[c:1][NX3;H2]",
            recommended_reagents=["NaNO2, H2SO4 (aq), then heat in aqueous Cu2O/Cu(NO3)2"],
            recommended_solvents=["H2O (aqueous)"],
            temperature="0°C to 100°C",
            moisture_sensitivity="aqueous",
            typical_yield=0.84,
            atom_economy=80.0,
            incompatible_groups=[],
            explanation="Diazotization of primary aromatic amine followed by hot aqueous hydrolysis to afford the corresponding phenol.",
        ))

        # -------------------------------------------------------------
        # 7. NUCLEOPHILIC ADDITIONS, ETHERS & CYANATION
        # -------------------------------------------------------------
        self.add_rule(RetroReactionRule(
            rule_id="RXN_WILLIAMSON_ETHER",
            name="Williamson Ether Synthesis (Alkyl Halide + Alcohol/Phenol)",
            reaction_class="Nucleophilic Substitution",
            retro_smarts="[#6:1][O:2][C:3]>>[#6:1][O:2].[C:3]Br",
            recommended_reagents=["K2CO3, KI (cat.)", "Cs2CO3", "NaH (60% in oil)"],
            recommended_solvents=["DMF", "MeCN", "Acetone (reflux)"],
            temperature="50°C to 80°C",
            moisture_sensitivity="moisture_sensitive",
            typical_yield=0.89,
            atom_economy=80.0,
            incompatible_groups=["acid_chloride"],
            explanation="Nucleophilic substitution of an alkoxide/phenoxide on an alkyl halide.",
        ))

        self.add_rule(RetroReactionRule(
            rule_id="RXN_SN2_CYANATION",
            name="Aliphatic Cyanation (Alkyl Halide -> Nitrile)",
            reaction_class="Nucleophilic Substitution",
            retro_smarts="[#6:1][C:2]#N>>[#6:1]Br",
            recommended_reagents=["NaCN or KCN (1.2 eq), NaI (cat.)", "TBAB (phase transfer cat.)"],
            recommended_solvents=["DMSO", "DMF", "EtOH / H2O"],
            temperature="60°C to 90°C",
            moisture_sensitivity="tolerant",
            typical_yield=0.91,
            atom_economy=85.0,
            incompatible_groups=["aldehyde", "ketone"],
            explanation="Nucleophilic substitution of alkyl halide with cyanide ion to extend carbon chain by one unit.",
        ))

        self.add_rule(RetroReactionRule(
            rule_id="RXN_NITRILE_HYDROLYSIS",
            name="Nitrile Hydrolysis to Carboxylic Acid",
            reaction_class="Hydrolysis",
            retro_smarts="[#6:1][C:2](=[O:3])O>>[#6:1][C:2]#N",
            recommended_reagents=["NaOH (6M aq), EtOH (reflux)", "H2SO4 / H2O (reflux)"],
            recommended_solvents=["H2O / EtOH"],
            temperature="100°C to 120°C",
            moisture_sensitivity="aqueous",
            typical_yield=0.93,
            atom_economy=95.0,
            incompatible_groups=["ester", "acid_chloride"],
            explanation="Aqueous alkaline or acidic hydrolysis of nitrile intermediate to yield carboxylic acid.",
            is_cascade_compatible=True,
        ))

        self.add_rule(RetroReactionRule(
            rule_id="RXN_GRIGNARD_ADDITION",
            name="Grignard Addition to Aldehyde/Ketone (Alcohol Synthesis)",
            reaction_class="Organometallic C-C Addition",
            retro_smarts="[#6:1][C:2]([OH])([#6:3])>>[#6:1][C:2]=O.[#6:3]Br",
            recommended_reagents=["R-MgBr in THF / Et2O", "LaCl3·2LiCl (for selective addition)"],
            recommended_solvents=["Anhydrous THF", "Anhydrous Diethyl ether"],
            temperature="-78°C to 0°C (under N2)",
            moisture_sensitivity="strictly_anhydrous",
            typical_yield=0.84,
            atom_economy=80.0,
            incompatible_groups=["alcohol", "carboxylic_acid", "primary_amine", "secondary_amine", "thiol"],
            protection_needed_for={"alcohol": "tbs", "carboxylic_acid": "ester", "primary_amine": "boc"},
            explanation="Nucleophilic attack of organomagnesium halide onto carbonyl carbon.",
        ))

    def add_rule(self, rule: RetroReactionRule):
        """Add a reaction rule to the library."""
        self.rules.append(rule)

    def get_rules_by_class(self, reaction_class: str) -> List[RetroReactionRule]:
        """Filter rules by category/class."""
        return [r for r in self.rules if r.reaction_class.lower() == reaction_class.lower()]

    def get_all_rules(self) -> List[RetroReactionRule]:
        """Return the full list of reaction rules."""
        return self.rules


def apply_retro_rule(mol: Chem.Mol, rule: RetroReactionRule) -> List[Tuple[List[str], RetroReactionRule]]:
    """Apply a retrosynthetic rule to a target molecule.
    
    Returns a list of candidate precursor sets: [([precursor_smi_1, precursor_smi_2], rule), ...]
    """
    rxn = rule.get_reaction()
    if rxn is None:
        return []

    try:
        reactants_tuples = rxn.RunReactants((mol,))
    except Exception:
        return []

    results: List[Tuple[List[str], RetroReactionRule]] = []
    seen_sets = set()

    for precursor_set in reactants_tuples:
        precursor_smis = []
        valid_set = True

        for p_mol in precursor_set:
            try:
                Chem.SanitizeMol(p_mol)
                smi = Chem.MolToSmiles(p_mol, canonical=True)
                if not smi or smi == "":
                    valid_set = False
                    break
                precursor_smis.append(smi)
            except Exception:
                valid_set = False
                break

        if valid_set and precursor_smis:
            # Sort precursor smiles for deduplication
            sorted_tuple = tuple(sorted(precursor_smis))
            if sorted_tuple not in seen_sets:
                seen_sets.add(sorted_tuple)
                results.append((precursor_smis, rule))

    return results
