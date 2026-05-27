import os
import json
import logging
from typing import List, Dict, Any
from backend.schemas import AgentStepResult

logger = logging.getLogger("drug_checker.agents.retrieval")

# Define helper to resolve paths dynamically
def get_db_path(filename: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "..", "db", filename)

def load_json_db(filename: str) -> Dict[str, Any]:
    path = get_db_path(filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading local DB {filename} at {path}: {e}")
        return {}

def run(drugs: List[str], conditions: List[str], target_illness: str = None, medical_history: str = None, allergies: str = None) -> AgentStepResult:
    """Executes the Retrieval Agent (Agent 2) performing parallel lookups in 3 databases."""
    logs = ["Initializing Retrieval Agent..."]
    
    # 0. Incorporate medical history conditions into active checked conditions
    active_conditions = list(conditions)
    if medical_history:
        logs.append(f"Parsing medical history context: '{medical_history}'")
        # Extract known condition keywords
        known_keywords = {
            "asthma": "asthma",
            "reactive airway": "asthma",
            "copd": "asthma",
            "bronchospasm": "asthma",
            "bronchitis": "asthma",
            "renal impairment": "renal impairment",
            "kidney disease": "renal impairment",
            "kidney failure": "renal impairment",
            "renal failure": "renal impairment",
            "renal insufficiency": "renal impairment",
            "ckd": "renal impairment",
            "gfr": "renal impairment",
            "egfr": "renal impairment",
            "hypertension": "hypertension",
            "high blood pressure": "hypertension",
            "high bp": "hypertension",
            "bp": "hypertension",
            "peptic ulcer": "peptic ulcer disease",
            "ulcer": "peptic ulcer disease",
            "stomach ulcer": "peptic ulcer disease",
            "gastric ulcer": "peptic ulcer disease",
            "cirrhosis": "liver impairment",
            "liver": "liver impairment",
            "liver impairment": "liver impairment",
            "liver failure": "liver impairment",
            "hepatic impairment": "liver impairment",
            "hepatic insufficiency": "liver impairment",
            "hepatitis": "liver impairment",
            "infection": "active infection"
        }
        history_lower = medical_history.lower()
        for kw, norm in known_keywords.items():
            if kw in history_lower and norm not in active_conditions:
                active_conditions.append(norm)
                logs.append(f"Extracted '{norm}' from medical history profile.")
                
    conditions = active_conditions
    
    # 1. FDA Interactions Lookup
    logs.append("Phase 2.1: Querying FDA Drug-Drug Interaction Database...")
    fda_db = load_json_db("fda_interactions.json")
    interactions_found = []
    
    # Check all pairs of drugs
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            d1, d2 = drugs[i], drugs[j]
            found_match = False
            for inter in fda_db.get("interactions", []):
                id1, id2 = inter["drug1"].lower(), inter["drug2"].lower()
                # Check both combinations
                if (d1 == id1 and d2 == id2) or (d1 == id2 and d2 == id1):
                    interactions_found.append(inter)
                    logs.append(f"[FDA MATCH] Found interaction between '{d1}' and '{d2}': Severity {inter['severity']}")
                    found_match = True
            
            if not found_match:
                logs.append(f"No direct FDA interaction record found for pair: {d1} + {d2}")
                
    # 2. Disease Contraindications Lookup
    logs.append("Phase 2.2: Querying Disease Contraindications Database...")
    disease_db = load_json_db("disease_contraindications.json")
    contraindications_found = []
    
    for drug in drugs:
        for condition in conditions:
            found_match = False
            for contra in disease_db.get("contraindications", []):
                cdrug = contra["drug"].lower()
                cdisease = contra["disease"].lower()
                # Use sub-string search for flexible condition matches
                if drug == cdrug and (condition in cdisease or cdisease in condition):
                    contraindications_found.append(contra)
                    logs.append(f"[DISEASE MATCH] Found contraindication for drug '{drug}' under condition '{condition}': Severity {contra['severity']}")
                    found_match = True
            if not found_match:
                logs.append(f"No direct contraindication record found for drug '{drug}' and condition '{condition}'")
                
    # 2.5. Allergy Conflicts Check
    if allergies and drugs:
        logs.append("Phase 2.2b: Checking for drug allergy conflicts with active profile...")
        allergy_terms = [a.strip().lower() for a in allergies.split(",") if a.strip()]
        for drug in drugs:
            for allergy in allergy_terms:
                if allergy and (allergy in drug or drug in allergy):
                    record = {
                        "drug": drug,
                        "disease": f"Allergy to {allergy}",
                        "severity": "CRITICAL",
                        "mechanism": "Immunological hypersensitivity response.",
                        "effects": "Anaphylaxis, severe skin rash (Steven-Johnson Syndrome), hives, respiratory distress, or other allergic responses.",
                        "clinical_management": "IMMEDIATELY DISCONTINUE and avoid all exposure. Do not co-administer.",
                        "alternatives": "Avoid all drugs in this therapeutic or chemical class. Consult an allergist."
                    }
                    contraindications_found.append(record)
                    logs.append(f"[ALLERGY ALERT] Drug '{drug}' matches patient-reported allergy: '{allergy}'!")
                
    # 3. Metabolism CYP450 DB Lookup
    logs.append("Phase 2.3: Querying CYP450 Metabolism Pathway Database...")
    metabolism_db = load_json_db("metabolism_db.json")
    metabolism_interactions_found = []
    
    # Find matching metabolism features
    # A pharmacokinetic interaction occurs if drug A is a substrate of an enzyme, 
    # and drug B is an inhibitor or inducer of that same enzyme.
    for enzyme in metabolism_db.get("enzymes", []):
        enzyme_name = enzyme["name"]
        substrates = [s.lower() for s in enzyme.get("substrates", [])]
        inhibitors_strong = [i.lower() for i in enzyme.get("inhibitors", {}).get("strong", [])]
        inhibitors_mod = [i.lower() for i in enzyme.get("inhibitors", {}).get("moderate", [])]
        inducers = [ind.lower() for ind in enzyme.get("inducers", [])]
        
        # Check if our drug list contains substrates and modulators of this enzyme
        active_substrates = [d for d in drugs if d in substrates]
        active_inhibitors_strong = [d for d in drugs if d in inhibitors_strong]
        active_inhibitors_mod = [d for d in drugs if d in inhibitors_mod]
        active_inducers = [d for d in drugs if d in inducers]
        
        # Report interactions
        if active_substrates:
            if active_inhibitors_strong:
                for sub in active_substrates:
                    for inh in active_inhibitors_strong:
                        if sub != inh:
                            record = {
                                "enzyme": enzyme_name,
                                "substrate": sub,
                                "modulator": inh,
                                "modulator_type": "strong inhibitor",
                                "severity": "CRITICAL",
                                "details": f"Pharmacokinetic. '{inh}' is a strong inhibitor of {enzyme_name}, which metabolizes '{sub}'. Co-administration increases '{sub}' serum concentrations and heightens risk of serious toxicities."
                            }
                            metabolism_interactions_found.append(record)
                            logs.append(f"[METABOLISM MATCH] CYP450 pathway conflict! {inh} (strong inhibitor) blocks metabolism of substrate {sub} via {enzyme_name}.")
                            
            if active_inhibitors_mod:
                for sub in active_substrates:
                    for inh in active_inhibitors_mod:
                        if sub != inh:
                            record = {
                                "enzyme": enzyme_name,
                                "substrate": sub,
                                "modulator": inh,
                                "modulator_type": "moderate inhibitor",
                                "severity": "WARNING",
                                "details": f"Pharmacokinetic. '{inh}' is a moderate inhibitor of {enzyme_name}, which metabolizes '{sub}'. Co-administration may increase '{sub}' serum concentrations, requiring close monitoring."
                            }
                            metabolism_interactions_found.append(record)
                            logs.append(f"[METABOLISM MATCH] CYP450 pathway warning. {inh} (moderate inhibitor) slows metabolism of substrate {sub} via {enzyme_name}.")
                            
            if active_inducers:
                for sub in active_substrates:
                    for ind in active_inducers:
                        if sub != ind:
                            record = {
                                "enzyme": enzyme_name,
                                "substrate": sub,
                                "modulator": ind,
                                "modulator_type": "inducer",
                                "severity": "WARNING",
                                "details": f"Pharmacokinetic. '{ind}' is an inducer of {enzyme_name}, which metabolizes '{sub}'. Co-administration accelerates '{sub}' clearance, reducing its clinical effectiveness."
                            }
                            metabolism_interactions_found.append(record)
                            logs.append(f"[METABOLISM MATCH] CYP450 pathway alert. {ind} (inducer) speeds up metabolism of substrate {sub} via {enzyme_name}.")

    output_data = {
        "fda_interactions": interactions_found,
        "disease_contraindications": contraindications_found,
        "metabolism_interactions": metabolism_interactions_found
    }
    
    logs.append(f"Retrieval summary: Found {len(interactions_found)} FDA, {len(contraindications_found)} Disease, and {len(metabolism_interactions_found)} Metabolism conflicts.")
    
    return AgentStepResult(
        agent_name="Retrieval Agent",
        description="Performs parallel database queries to fetch interaction clinical context.",
        input_data={"drugs": drugs, "conditions": conditions},
        output_data=output_data,
        logs=logs
    )
