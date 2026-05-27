import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field
from backend.schemas import AgentStepResult
from backend.agents.llm import call_llm, is_ai_active

logger = logging.getLogger("drug_checker.agents.input")

# List of known drugs/conditions for robust simulation matching
KNOWN_DRUGS = [
    "aspirin", "ibuprofen", "warfarin", "sildenafil", "lisinopril", "propranolol", 
    "metoprolol", "metformin", "clarithromycin", "nitroglycerin", "spironolactone", 
    "acetaminophen", "paracetamol", "pseudoephedrine", "prednisone", "fluoxetine", 
    "phenelzine", "atorvastatin", "simvastatin", "lovastatin", "amlodipine", 
    "grapefruit juice", "erythromycin", "verapamil", "diltiazem", "fluconazole"
]

BRAND_TO_GENERIC = {
    "tylenol": "acetaminophen",
    "paracetamol": "acetaminophen",
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "coumadin": "warfarin",
    "viagra": "sildenafil",
    "zestril": "lisinopril",
    "prinivil": "lisinopril",
    "inderal": "propranolol",
    "lopressor": "metoprolol",
    "toprol-xl": "metoprolol",
    "toprol xl": "metoprolol",
    "glucophage": "metformin",
    "biaxin": "clarithromycin",
    "nitrostat": "nitroglycerin",
    "nitro": "nitroglycerin",
    "aldactone": "spironolactone",
    "sudafed": "pseudoephedrine",
    "prozac": "fluoxetine",
    "lipitor": "atorvastatin",
    "zocor": "simvastatin",
    "norvasc": "amlodipine",
    "cardizem": "diltiazem",
    "tiazac": "diltiazem",
    "calan": "verapamil",
    "verelan": "verapamil",
    "diflucan": "fluconazole",
    "e-mycin": "erythromycin",
    "ery-tab": "erythromycin",
    "eryc": "erythromycin"
}

KNOWN_CONDITIONS = {
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

class InputAgentSchema(BaseModel):
    drugs: List[str] = Field(description="List of drug names extracted from the query, in lowercase.")
    conditions: List[str] = Field(description="List of medical conditions extracted from both the query and the history, normalized in lowercase.")
    intent: str = Field(description="The user's primary goal, e.g. checking drug interactions, safety, dosage or general information.")
    normalized_allergies: Optional[List[str]] = Field(default=None, description="List of drug names the patient is allergic to, extracted from allergies and normalized to generic names in lowercase.")

def run_simulation(query: str, medical_history: str = None, allergies: str = None) -> Tuple[List[str], List[str], str, str, List[str]]:
    """Rule-based parser for query text, returning (drugs, conditions, normalized_allergies, intent, logs)."""
    logs = ["Running Input Agent in Simulation Mode..."]
    normalized_query = query.lower()
    
    # Extract drugs (checking generic names first, then brand names)
    extracted_drugs = []
    for drug in KNOWN_DRUGS:
        if drug in normalized_query and drug not in extracted_drugs:
            extracted_drugs.append(drug)
            logs.append(f"Detected drug matching keyword: '{drug}'")
            
    for brand, generic in BRAND_TO_GENERIC.items():
        if brand in normalized_query and generic not in extracted_drugs:
            extracted_drugs.append(generic)
            logs.append(f"Detected brand drug '{brand}' -> normalized to generic '{generic}'")
            
    # Extract conditions from both query and medical history
    extracted_conditions = []
    combined_context = normalized_query
    if medical_history:
        combined_context += " " + medical_history.lower()
        
    for term, norm_cond in KNOWN_CONDITIONS.items():
        if term in combined_context and norm_cond not in extracted_conditions:
            extracted_conditions.append(norm_cond)
            logs.append(f"Detected condition matching keyword '{term}' -> normalized to '{norm_cond}'")
            
    # Normalize allergies
    normalized_algs_list = []
    if allergies:
        alg_terms = [a.strip().lower() for a in allergies.split(",") if a.strip()]
        for alg in alg_terms:
            norm_alg = BRAND_TO_GENERIC.get(alg, alg)
            normalized_algs_list.append(norm_alg)
            if norm_alg != alg:
                logs.append(f"Normalized allergy drug '{alg}' -> generic '{norm_alg}'")
                
    normalized_algs = ", ".join(normalized_algs_list) if normalized_algs_list else allergies

    # Infer intent
    intent = "interaction_check"
    if "dose" in normalized_query or "dosage" in normalized_query:
        intent = "dosage_check"
        logs.append("Inferred intent: Dosage Check based on query keywords.")
    elif "safe" in normalized_query or "safety" in normalized_query:
        intent = "safety_evaluation"
        logs.append("Inferred intent: Safety Evaluation based on query keywords.")
    else:
        logs.append("Inferred intent: Drug Interaction Check (Default).")
        
    if not extracted_drugs:
        logs.append("[WARNING] No known drugs detected in the user query.")
    if not extracted_conditions:
        logs.append("No medical conditions detected in the user query or profile.")
        
    return extracted_drugs, extracted_conditions, normalized_algs, intent, logs

def run(query: str, pre_extracted_drugs: List[str] = None, pre_extracted_conditions: List[str] = None,
        target_illness: str = None, medical_history: str = None, allergies: str = None) -> AgentStepResult:
    """Executes the Input Agent (Agent 1) to parse user query and patient profile."""
    logs = []
    drugs = pre_extracted_drugs or []
    conditions = pre_extracted_conditions or []
    intent = "interaction_check"
    normalized_algs = allergies
    
    # If parameters were already supplied, we can short-circuit or supplement
    if drugs or conditions:
        logs.append("Pre-extracted inputs supplied by user. Skipping full textual extraction.")
        drugs = [d.lower().strip() for d in drugs]
        # Normalize brand names in pre-extracted drugs
        drugs = [BRAND_TO_GENERIC.get(d, d) for d in drugs]
        
        conditions = [c.lower().strip() for c in conditions]
        # Normalize conditions
        normalized_conds = []
        for c in conditions:
            # check synonyms
            matched = False
            for term, norm in KNOWN_CONDITIONS.items():
                if term == c:
                    normalized_conds.append(norm)
                    matched = True
                    break
            if not matched:
                normalized_conds.append(c)
        conditions = list(set(normalized_conds))
        
        if allergies:
            alg_terms = [a.strip().lower() for a in allergies.split(",") if a.strip()]
            normalized_algs = ", ".join([BRAND_TO_GENERIC.get(a, a) for a in alg_terms])
            
        intent = "interaction_check"
        output_data = {
            "drugs": drugs,
            "conditions": conditions,
            "intent": intent,
            "target_illness": target_illness,
            "medical_history": medical_history,
            "allergies": normalized_algs
        }
        return AgentStepResult(
            agent_name="Input Agent",
            description="Extracts drug names, conditions, and clinical intent from query.",
            input_data={
                "query": query, 
                "pre_extracted_drugs": pre_extracted_drugs, 
                "pre_extracted_conditions": pre_extracted_conditions,
                "target_illness": target_illness,
                "medical_history": medical_history,
                "allergies": allergies
            },
            output_data=output_data,
            logs=logs
        )
        
    if is_ai_active():
        try:
            logs.append("Contacting Gemini for semantic extraction...")
            system_instruction = (
                "You are the Input Agent in a clinical drug interaction pipeline. "
                "Analyze the user query, patient medical history, and allergies to extract all drug names, "
                "medical conditions/diseases, and clinical intent. "
                "Standardize all drug names to their common generic lowercase names (e.g. 'Tylenol' -> 'acetaminophen'). "
                "Extract all medical conditions from both the query and the medical history, and normalize them to standard clinical terms "
                "(e.g. mapping kidney issues/CKD/renal failure to 'renal impairment', stomach issues/PUD to 'peptic ulcer disease', "
                "asthma/COPD to 'asthma', high BP to 'hypertension', cirrhosis/hepatic issues to 'liver impairment'). "
                "Output JSON matching the specified schema."
            )
            hist_str = medical_history or "None reported"
            algs_str = allergies or "None reported"
            prompt = f"User query: '{query}'\nPatient Medical History: '{hist_str}'\nPatient Allergies: '{algs_str}'"
            response_text = call_llm(prompt, system_instruction, response_schema=InputAgentSchema)
            parsed = json.loads(response_text)
            
            drugs = [d.lower() for d in parsed.get("drugs", [])]
            conditions = [c.lower() for c in parsed.get("conditions", [])]
            intent = parsed.get("intent", "interaction_check")
            
            normalized_algs_list = parsed.get("normalized_allergies", [])
            normalized_algs = ", ".join(normalized_algs_list) if (normalized_algs_list is not None) else allergies
            
            logs.append("[Success] Gemini extracted entities successfully.")
            logs.append(f"Extracted Drugs: {drugs}")
            logs.append(f"Extracted Conditions: {conditions}")
            logs.append(f"Inferred Intent: {intent}")
            if normalized_algs_list:
                logs.append(f"Normalized Allergies: {normalized_algs_list}")
            
        except Exception as e:
            logs.append(f"[Error] Gemini extraction failed: {str(e)}. Falling back to simulation.")
            sim_drugs, sim_conditions, sim_algs, sim_intent, sim_logs = run_simulation(query, medical_history, allergies)
            drugs = sim_drugs
            conditions = sim_conditions
            normalized_algs = sim_algs
            intent = sim_intent
            logs.extend(sim_logs)
    else:
        sim_drugs, sim_conditions, sim_algs, sim_intent, sim_logs = run_simulation(query, medical_history, allergies)
        drugs = sim_drugs
        conditions = sim_conditions
        normalized_algs = sim_algs
        intent = sim_intent
        logs.extend(sim_logs)
        
    output_data = {
        "drugs": drugs,
        "conditions": conditions,
        "intent": intent,
        "target_illness": target_illness,
        "medical_history": medical_history,
        "allergies": normalized_algs
    }
    
    return AgentStepResult(
        agent_name="Input Agent",
        description="Extracts drug names, conditions, and clinical intent from query and patient profile.",
        input_data={
            "query": query,
            "target_illness": target_illness,
            "medical_history": medical_history,
            "allergies": allergies
        },
        output_data=output_data,
        logs=logs
    )
