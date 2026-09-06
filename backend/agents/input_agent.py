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

import re
from backend.agents.clinique_rxnorm import RxNormClient, DrugRecognized, DrugUnrecognized

_rxnorm_client = RxNormClient()

DRUG_EXTRACTION_PATTERN = re.compile(
    r'\b(?:on|taking|prescribed|started|stopped|allergic to|allergy to)\s+([a-zA-Z0-9\s,\-]+?)(?:\.|,|;|$|\band\b)',
    re.IGNORECASE
)


class InputAgentSchema(BaseModel):
    drugs: List[str] = Field(description="List of drug names extracted from the query, in lowercase.")
    conditions: List[str] = Field(description="List of medical conditions extracted from both the query and the history, normalized in lowercase.")
    intent: str = Field(description="The user's primary goal, e.g. checking drug interactions, safety, dosage or general information.")
    normalized_allergies: Optional[List[str]] = Field(default=None, description="List of drug names the patient is allergic to, normalized in lowercase.")


def extract_candidate_drug_mentions(query: str) -> List[str]:
    """Extracts candidate medication mentions from natural language query or structured text."""
    mentions: List[str] = []
    
    # 1. Regex pattern matches (e.g. "on aspirin", "taking Eliquis")
    matches = DRUG_EXTRACTION_PATTERN.findall(query)
    for match in matches:
        for part in match.split(","):
            cleaned = part.strip()
            if cleaned and len(cleaned.split()) <= 3:
                mentions.append(cleaned)
                
    # 2. Check keyword lists or section prefixes
    lower_q = query.lower()
    if "medications:" in lower_q or "drugs:" in lower_q:
        idx = lower_q.find("medications:") if "medications:" in lower_q else lower_q.find("drugs:")
        after = query[idx:].split(":", 1)[-1].split("\n")[0]
        for part in after.split(","):
            cleaned = part.strip()
            if cleaned:
                mentions.append(cleaned)

    # 3. Check known database drugs & brand names
    for drug in KNOWN_DRUGS:
        if re.search(rf'\b{re.escape(drug)}\b', lower_q):
            mentions.append(drug)
    for brand in BRAND_TO_GENERIC.keys():
        if re.search(rf'\b{re.escape(brand)}\b', lower_q):
            mentions.append(brand)

    # 4. If query is a comma or 'and' separated list of terms, parse tokens
    tokens = [t.strip() for t in re.split(r'[,;]|\band\b', query) if t.strip()]
    for t in tokens:
        # Ignore common condition words or filler words
        t_clean = t.lower().strip()
        words = t_clean.split()
        if len(words) <= 3 and t_clean not in KNOWN_CONDITIONS and t_clean not in ("safe", "safety", "dose", "dosage", "check", "patient", "none"):
            mentions.append(t)

    # Deduplicate while preserving order
    unique_mentions: List[str] = []
    for m in mentions:
        m_strip = m.strip()
        if m_strip and not any(m_strip.lower() == u.lower() for u in unique_mentions):
            unique_mentions.append(m_strip)

    return unique_mentions


def run(
    query: str,
    pre_extracted_drugs: Optional[List[str]] = None,
    pre_extracted_conditions: Optional[List[str]] = None,
    target_illness: Optional[str] = None,
    medical_history: Optional[str] = None,
    allergies: Optional[str] = None
) -> AgentStepResult:
    """
    Executes the Input Agent (Agent 1) with live RxNorm normalization:
    1. Extracts drug candidates from text or pre-extracted list.
    2. Normalizes candidates via RxNormClient.
    3. Separates into drugs_recognized and drugs_unrecognized.
    4. Extracts conditions from query and medical history.
    5. Normalizes allergy mentions.
    """
    logs: List[str] = ["Initializing Input Agent with RxNorm drug normalization..."]
    intent = "interaction_check"
    conditions = list(pre_extracted_conditions or [])

    # 1. Gather Candidate Drug Mentions
    if pre_extracted_drugs:
        logs.append(f"Using pre-extracted drug mentions: {pre_extracted_drugs}")
        candidates = list(pre_extracted_drugs)
    else:
        candidates = extract_candidate_drug_mentions(query)
        logs.append(f"Extracted candidate drug mentions from query: {candidates}")

    # 2. Normalize via RxNorm Client
    recognized_models, unrecognized_models = _rxnorm_client.normalize_drug_list(candidates)
    
    drugs_recognized = [r.to_dict() for r in recognized_models]
    drugs_unrecognized = [u.to_dict() for u in unrecognized_models]

    for rec in recognized_models:
        logs.append(f"[RxNorm Match] '{rec.name}' -> {rec.preferred_name} (RxCUI: {rec.rxcui}, match: {rec.match_type})")
    for unrec in unrecognized_models:
        logs.append(f"[RxNorm Unrecognized] ⚠ '{unrec.name}': {unrec.reason}")

    # Active drugs list for pipeline: use preferred normalized names for recognized + original for unrecognized
    drugs = [r.preferred_name for r in recognized_models]
    for u in unrecognized_models:
        if u.name.lower() not in [d.lower() for d in drugs]:
            drugs.append(u.name.lower())

    # 3. Extract Conditions from Query and Medical History
    lower_context = query.lower()
    if medical_history:
        lower_context += " " + medical_history.lower()

    for term, norm_cond in KNOWN_CONDITIONS.items():
        if term in lower_context and norm_cond not in conditions:
            conditions.append(norm_cond)
            logs.append(f"Detected condition '{term}' -> normalized to '{norm_cond}'")

    # 4. Normalize Allergies
    normalized_algs_list = []
    if allergies:
        alg_terms = [a.strip() for a in allergies.split(",") if a.strip()]
        for a in alg_terms:
            a_rec, _ = _rxnorm_client.normalize_drug(a)
            if a_rec:
                normalized_algs_list.append(a_rec.preferred_name)
            else:
                normalized_algs_list.append(BRAND_TO_GENERIC.get(a.lower(), a.lower()))
    normalized_algs = ", ".join(normalized_algs_list) if normalized_algs_list else allergies

    # 5. Infer Intent
    lower_q = query.lower()
    if "dose" in lower_q or "dosage" in lower_q:
        intent = "dosage_check"
    elif "safe" in lower_q or "safety" in lower_q:
        intent = "safety_evaluation"

    # Confidence calculation
    total_mentions = len(drugs_recognized) + len(drugs_unrecognized)
    confidence = round(len(drugs_recognized) / total_mentions, 2) if total_mentions > 0 else 0.0

    output_data = {
        "drugs": drugs,
        "conditions": conditions,
        "intent": intent,
        "target_illness": target_illness,
        "medical_history": medical_history,
        "allergies": normalized_algs,
        "drugs_recognized": drugs_recognized,
        "drugs_unrecognized": drugs_unrecognized,
        "extraction_confidence": confidence
    }

    logs.append(f"Input Agent completed: {len(drugs_recognized)} recognized, {len(drugs_unrecognized)} unrecognized.")

    return AgentStepResult(
        agent_name="Input Agent",
        description="Extracts drug mentions, normalizes via RxNorm, and flags unrecognized drugs.",
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

