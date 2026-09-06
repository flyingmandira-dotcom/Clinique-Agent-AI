import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field, ValidationError

from backend.schemas import AgentStepResult, ClinicalReport
from backend.agents.llm import call_llm, is_ai_active

logger = logging.getLogger("drug_checker.agents.combined_assessment")

# ============================================================================
# 1. Standardized Clinical Risk Hierarchy & Ranking
# ============================================================================

CLINICAL_RISK_RANK = {
    "SAFE": 0,
    "LOW_RISK": 1,
    "MODERATE_RISK": 2,
    "HIGH_RISK": 3,
}

SEVERITY_RANK = {
    "SAFE": 0,
    "LOW_RISK": 1,
    "REVIEW_REQUIRED": 2,
    "MODERATE_RISK": 3,
    "HIGH_RISK": 4,
}

VALID_SEVERITIES = {"HIGH_RISK", "MODERATE_RISK", "LOW_RISK", "SAFE", "REVIEW_REQUIRED"}


def normalize_severity(raw: str) -> str:
    """Normalizes legacy and incoming severity strings to standardized tiers."""
    s = (raw or "").strip().upper()
    if s in ("CRITICAL", "HIGH", "SEVERE", "HIGH_RISK"):
        return "HIGH_RISK"
    if s in ("WARNING", "MODERATE", "MODERATE_RISK"):
        return "MODERATE_RISK"
    if s in ("LOW", "MINOR", "LOW_RISK"):
        return "LOW_RISK"
    if s in ("REVIEW", "UNCERTAIN", "REVIEW_REQUIRED", "UNVERIFIED"):
        return "REVIEW_REQUIRED"
    if s in ("SAFE", "CLEAN"):
        return "SAFE"
    return "REVIEW_REQUIRED"


# ============================================================================
# 2. Strongly Typed Pydantic Schemas for Output Structure
# ============================================================================

class ItemAssessment(BaseModel):
    type: str = Field(description="Clinical category (e.g. Drug–Drug Interaction, Contraindication, CYP450).")
    title: str = Field(description="Short descriptive title of the finding.")
    severity: str = Field(description="Severity tier: HIGH_RISK, MODERATE_RISK, LOW_RISK, SAFE, or REVIEW_REQUIRED.")
    justification: str = Field(description="Clinical rationale and mechanism.")


class InteractionDetail(BaseModel):
    drugs: str = Field(description="Interacting drug pair (e.g. 'Warfarin and Ibuprofen').")
    severity: str = Field(description="Severity tier of the specific interaction.")
    mechanism: str = Field(description="Pharmacodynamic or pharmacokinetic mechanism.")
    clinical_effects: str = Field(description="Anticipated clinical consequences.")
    management: str = Field(description="Recommended clinical mitigation or management.")


class DiseaseWarning(BaseModel):
    drug: str = Field(description="Medication involved in the condition conflict.")
    condition: str = Field(description="Patient disease, organ impairment, or allergy.")
    severity: str = Field(description="Severity tier of this contraindication or precaution.")
    mechanism: str = Field(description="Underlying pathophysiological interaction mechanism.")
    clinical_effects: str = Field(description="Adverse clinical outcomes or complications.")
    management: str = Field(description="Dose adjustment, monitoring, or discontinuation guidance.")


class MetabolismInteraction(BaseModel):
    substrate: str = Field(description="Substrate medication whose metabolism is impacted.")
    modulator: str = Field(description="Inhibitor or inducer medication.")
    enzyme: str = Field(description="Cytochrome P450 enzyme (e.g. CYP3A4, CYP2D6, CYP2C9).")
    interaction_type: str = Field(default="CYP modulator", description="Strong/moderate inhibitor or inducer.")
    severity: str = Field(description="Severity tier of the metabolic conflict.")
    clinical_details: str = Field(description="Pharmacokinetic consequences and exposure changes.")


class SuggestedAlternative(BaseModel):
    original_drugs: str = Field(description="The medication or combination being substituted.")
    alternative: str = Field(description="Safer evidence-supported alternative agent or class.")
    reasoning: str = Field(description="Clinical explanation of why the alternative is safer.")


class CombinedClinicalAssessmentSchema(BaseModel):
    severity: str = Field(
        description="One of: HIGH_RISK, MODERATE_RISK, LOW_RISK, SAFE, REVIEW_REQUIRED"
    )
    reasoning: str = Field(
        description="Concise clinical reasoning supporting the assigned severity."
    )
    item_assessments: List[ItemAssessment] = Field(
        default_factory=list,
        description="One entry per identified clinical issue."
    )
    summary: str = Field(
        description="One or two sentence overall summary."
    )
    allergies_checked: List[str] = Field(
        default_factory=list,
        description="All allergies evaluated during the assessment."
    )
    ddi_assessment: str = Field(
        description="Dedicated drug-drug interaction assessment."
    )
    drug_condition_assessment: str = Field(
        description="Dedicated drug-condition contraindication/precaution assessment."
    )
    cyp450_assessment: str = Field(
        description="Dedicated CYP450 metabolic pathway assessment."
    )
    clinical_advisory: str = Field(
        description="Detailed clinician guidance based strictly on available evidence."
    )
    interactions_details: List[InteractionDetail] = Field(
        default_factory=list,
        description="Structured drug-drug interaction findings."
    )
    disease_warnings: List[DiseaseWarning] = Field(
        default_factory=list,
        description="Structured drug-condition/allergy warnings."
    )
    metabolism_interactions: List[MetabolismInteraction] = Field(
        default_factory=list,
        description="Structured CYP450 metabolism conflicts."
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Prioritized actionable clinical recommendations."
    )
    suggested_alternatives: List[SuggestedAlternative] = Field(
        default_factory=list,
        description="Safer alternatives only when supported by available evidence."
    )
    citations: List[str] = Field(
        default_factory=list,
        description="Source types used to ground the assessment."
    )


# ============================================================================
# 3. Database Reference & Unknown Drug Safeguards
# ============================================================================

_VERIFIED_DRUGS_CACHE: Optional[set] = None


def get_verified_drugs() -> set:
    """Collects and caches all verified medications across reference databases."""
    global _VERIFIED_DRUGS_CACHE
    if _VERIFIED_DRUGS_CACHE is not None:
        return _VERIFIED_DRUGS_CACHE

    verified = {
        "aspirin", "ibuprofen", "warfarin", "sildenafil", "lisinopril", "propranolol",
        "metoprolol", "metformin", "clarithromycin", "nitroglycerin", "spironolactone",
        "acetaminophen", "paracetamol", "pseudoephedrine", "prednisone", "fluoxetine",
        "phenelzine", "atorvastatin", "simvastatin", "lovastatin", "amlodipine",
        "grapefruit juice", "erythromycin", "verapamil", "diltiazem", "fluconazole",
        "amoxicillin", "penicillin", "tramadol", "codeine", "donepezil", "haloperidol",
        "venlafaxine", "paroxetine", "bupropion", "quinidine", "sertraline", "duloxetine",
        "amiodarone", "diphenhydramine", "celecoxib", "phenytoin", "losartan", "glipizide",
        "metronidazole", "miconazole", "voriconazole", "rifampin", "carbamazepine",
        "phenobarbital", "ketoconazole", "itraconazole", "ritonavir", "nefazodone",
        "cimetidine", "secobarbital", "st. john's wort", "fentanyl", "midazolam",
        "nifedipine", "cyclosporine", "sulfamethoxazole", "cetirizine"
    }

    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(current_dir, "..", "db")

    for fname in ("fda_interactions.json", "disease_contraindications.json", "metabolism_db.json"):
        p = os.path.join(db_dir, fname)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "interactions" in data:
                        for item in data["interactions"]:
                            verified.add(item.get("drug1", "").strip().lower())
                            verified.add(item.get("drug2", "").strip().lower())
                    if "contraindications" in data:
                        for item in data["contraindications"]:
                            verified.add(item.get("drug", "").strip().lower())
                    if "enzymes" in data:
                        for enz in data["enzymes"]:
                            for s in enz.get("substrates", []):
                                verified.add(s.strip().lower())
                            for inh_list in enz.get("inhibitors", {}).values():
                                for i in inh_list:
                                    verified.add(i.strip().lower())
                            for ind in enz.get("inducers", []):
                                verified.add(ind.strip().lower())
            except Exception as e:
                logger.warning(f"Failed to load DB {fname} for verified drug cache: {e}")

    _VERIFIED_DRUGS_CACHE = {d for d in verified if d}
    return _VERIFIED_DRUGS_CACHE


from backend.agents.clinique_rxnorm import RxNormClient
_rxnorm_verifier = RxNormClient()

def identify_unverified_drugs(drugs: List[str]) -> List[str]:
    """Identifies medications not verified by RxNorm or local clinical reference databases."""
    verified = get_verified_drugs()
    unverified = []
    for d in drugs:
        d_norm = d.strip().lower()
        if not d_norm:
            continue
        if d_norm in verified:
            continue
        # Verify against RxNorm
        rec, _ = _rxnorm_verifier.normalize_drug(d)
        if rec:
            verified.add(d_norm)
            verified.add(rec.preferred_name.lower())
        else:
            unverified.append(d)
    return unverified



# ============================================================================
# 4. Deterministic Safety Floor & Risk Resolution
# ============================================================================

def evaluate_deterministic_severity(
    retrieval_data: Dict[str, Any],
    drugs: Optional[List[str]] = None,
    conditions: Optional[List[str]] = None,
    allergies: Optional[str] = None
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Authoritative deterministic clinical baseline across all safety dimensions:
    1. Patient allergy / absolute contraindication -> HIGH_RISK
    2. Drug-condition contraindication -> HIGH_RISK or MODERATE_RISK
    3. Major drug-drug interaction -> HIGH_RISK
    4. Significant CYP450 metabolism conflict -> MODERATE_RISK or HIGH_RISK
    5. Minor interaction -> LOW_RISK
    6. Presence of unverified conditions / allergies -> REVIEW_REQUIRED
    7. Truly clean across all dimensions -> SAFE
    """
    items: List[Dict[str, Any]] = []
    drugs = [d.strip().lower() for d in (drugs or []) if d.strip()]
    conditions = [c.strip().lower() for c in (conditions or []) if c.strip()]
    highest_severity = "SAFE"

    def upgrade_severity(candidate: str):
        nonlocal highest_severity
        norm = normalize_severity(candidate)
        if SEVERITY_RANK.get(norm, 0) > SEVERITY_RANK.get(highest_severity, 0):
            highest_severity = norm

    # 1. Drug-Drug Interactions (FDA)
    for inter in retrieval_data.get("fda_interactions", []):
        sev = normalize_severity(inter.get("severity", "WARNING"))
        upgrade_severity(sev)
        items.append({
            "type": "Drug–Drug Interaction",
            "title": f"{inter.get('drug1', '').capitalize()} + {inter.get('drug2', '').capitalize()}",
            "severity": sev,
            "justification": f"Mechanism: {inter.get('mechanism', 'N/A')}. Effect: {inter.get('effects', 'N/A')}"
        })

    # 2. Disease Contraindications and Allergies
    for contra in retrieval_data.get("disease_contraindications", []):
        sev = normalize_severity(contra.get("severity", "CRITICAL"))
        upgrade_severity(sev)
        item_type = "Patient Allergy Alert" if "allergy" in contra.get("disease", "").lower() else "Drug–Condition Contraindication"
        items.append({
            "type": item_type,
            "title": f"{contra.get('drug', '').capitalize()} in {contra.get('disease', '').capitalize()}",
            "severity": sev,
            "justification": f"Mechanism: {contra.get('mechanism', 'N/A')}. Effect: {contra.get('effects', 'N/A')}"
        })

    # 3. CYP450 Pharmacokinetic Metabolism Conflicts
    for metab in retrieval_data.get("metabolism_interactions", []):
        sev = normalize_severity(metab.get("severity", "WARNING"))
        upgrade_severity(sev)
        items.append({
            "type": "CYP450 Metabolism Conflict",
            "title": f"{metab.get('substrate', '').capitalize()} metabolized by {metab.get('enzyme', '')} modulated by {metab.get('modulator', '').capitalize()}",
            "severity": sev,
            "justification": metab.get("details", "")
        })

    # 4. Unknown/Unverified Drugs Check
    unverified = identify_unverified_drugs(drugs)
    if unverified and highest_severity in ("SAFE", "LOW_RISK"):
        highest_severity = "REVIEW_REQUIRED"
        reason = (
            f"Medication(s) [{', '.join(unverified)}] could not be verified in the local clinical reference databases. "
            f"Clinical safety cannot be established without verified pharmacological data. Clinician review is required."
        )
        return highest_severity, reason, items

    # 5. Patient Conditions Present but No Direct Contraindication Rule Matched
    if highest_severity == "SAFE" and conditions:
        highest_severity = "REVIEW_REQUIRED"
        reason = (
            f"No direct drug–drug interaction was detected among {', '.join(drugs) if drugs else 'the medications'}. "
            f"However, the patient has reported health conditions ({', '.join(conditions)}) which require specific "
            f"clinician review before treatment. The absence of a detected database match is not proof of safety."
        )
        return highest_severity, reason, items

    # 6. Patient Allergies Present
    if highest_severity == "SAFE" and allergies and allergies.strip().lower() not in ("none", "none reported"):
        highest_severity = "REVIEW_REQUIRED"
        reason = (
            f"No drug–drug interaction was found, but the patient profile indicates allergies ({allergies}). "
            f"Specific cross-reactivity and excipient verification is required prior to administration."
        )
        return highest_severity, reason, items

    # Construct concise summary reasoning for the assigned baseline
    if highest_severity == "HIGH_RISK":
        reason = (
            "A HIGH_RISK clinical concern has been detected. Critical contraindications, serious allergy warnings, "
            "or severe drug–drug interactions were identified that present life-threatening dangers. "
            "The regimen should be halted or replaced under immediate medical direction."
        )
    elif highest_severity == "MODERATE_RISK":
        reason = (
            "MODERATE_RISK considerations detected. Clinically significant drug interactions, drug–condition precautions, "
            "or CYP450 metabolism clearance alterations exist. Monitoring or dose adjustment is advised."
        )
    elif highest_severity == "LOW_RISK":
        reason = (
            "LOW_RISK considerations detected. Only minor, non-serious interactions or precautions were identified. "
            "Therapy may proceed under standard clinical monitoring."
        )
    elif highest_severity == "REVIEW_REQUIRED":
        reason = (
            "REVIEW_REQUIRED. Clinical data is insufficient to verify routine safety under standard conditions. "
            "A qualified clinician or clinical pharmacist must evaluate patient-specific factors before administration."
        )
    else:
        reason = (
            "SAFE. No clinically significant drug–drug interactions, contraindications, allergy conflicts, or CYP450 "
            "pathway risks were identified. Available evidence supports routine use under standard clinical conditions."
        )

    return highest_severity, reason, items


def resolve_safety_floor(
    ai_severity: str,
    det_severity: str,
    ai_reasoning: str,
    det_reasoning: str
) -> Tuple[str, str]:
    """
    Decouples clinical risk hierarchy from uncertainty (REVIEW_REQUIRED).
    Ensures that:
    - AI cannot downgrade a deterministic HIGH_RISK or MODERATE_RISK finding.
    - REVIEW_REQUIRED does not hide a confirmed HIGH_RISK finding.
    - AI cannot return SAFE if deterministic floor requires REVIEW_REQUIRED (e.g. unverified conditions/drugs).
    """
    norm_ai = normalize_severity(ai_severity)
    norm_det = normalize_severity(det_severity)

    ai_risk = CLINICAL_RISK_RANK.get(norm_ai, -1)
    det_risk = CLINICAL_RISK_RANK.get(norm_det, -1)

    # 1. If deterministic rules confirmed HIGH_RISK, never allow downgrade
    if norm_det == "HIGH_RISK":
        return "HIGH_RISK", (ai_reasoning if norm_ai == "HIGH_RISK" else det_reasoning)

    # 2. If deterministic rules confirmed MODERATE_RISK
    if norm_det == "MODERATE_RISK":
        if norm_ai == "HIGH_RISK":
            return "HIGH_RISK", ai_reasoning
        return "MODERATE_RISK", (ai_reasoning if norm_ai == "MODERATE_RISK" else det_reasoning)

    # 3. If deterministic rules require REVIEW_REQUIRED (uncertainty, unverified condition/drug)
    if norm_det == "REVIEW_REQUIRED":
        # If AI identified a genuine high or moderate clinical risk, risk takes precedence
        if norm_ai in ("HIGH_RISK", "MODERATE_RISK"):
            return norm_ai, ai_reasoning
        # Otherwise, prevent false SAFE or false LOW_RISK reassurance
        return "REVIEW_REQUIRED", (det_reasoning if norm_ai == "SAFE" else (ai_reasoning or det_reasoning))

    # 4. If deterministic was LOW_RISK or SAFE
    if norm_ai in ("HIGH_RISK", "MODERATE_RISK", "LOW_RISK"):
        # AI can escalate risk based on retrieved clinical details
        if ai_risk >= det_risk:
            return norm_ai, ai_reasoning
        return norm_det, det_reasoning
    elif norm_ai == "REVIEW_REQUIRED":
        # AI identified uncertainty not captured in basic DB rules
        return "REVIEW_REQUIRED", ai_reasoning

    # Default to deterministic baseline
    return norm_det, det_reasoning


# ============================================================================
# 5. Token Optimization: Filter & Deduplicate Retrieval Context
# ============================================================================

def optimize_retrieval_context(retrieval_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filters, deduplicates, and condenses retrieved records before prompt construction.
    Avoids sending duplicate records, excess whitespace, and empty lists.
    """
    cleaned: Dict[str, Any] = {
        "fda_interactions": [],
        "disease_contraindications": [],
        "metabolism_interactions": []
    }

    # 1. Deduplicate FDA interactions by normalized drug pair
    seen_fda = set()
    for item in retrieval_data.get("fda_interactions", []):
        pair_key = tuple(sorted([item.get("drug1", "").strip().lower(), item.get("drug2", "").strip().lower()]))
        if pair_key not in seen_fda and any(pair_key):
            seen_fda.add(pair_key)
            cleaned["fda_interactions"].append({
                "drug1": item.get("drug1", "").strip().lower(),
                "drug2": item.get("drug2", "").strip().lower(),
                "severity": item.get("severity", "WARNING"),
                "mechanism": item.get("mechanism", ""),
                "effects": item.get("effects", ""),
                "management": item.get("clinical_management", item.get("management", "")),
                "alternatives": item.get("alternatives", "")
            })

    # 2. Deduplicate Disease contraindications
    seen_disease = set()
    for item in retrieval_data.get("disease_contraindications", []):
        d_key = (item.get("drug", "").strip().lower(), item.get("disease", "").strip().lower())
        if d_key not in seen_disease and d_key[0]:
            seen_disease.add(d_key)
            cleaned["disease_contraindications"].append({
                "drug": item.get("drug", "").strip().lower(),
                "disease": item.get("disease", "").strip().lower(),
                "severity": item.get("severity", "CRITICAL"),
                "mechanism": item.get("mechanism", ""),
                "effects": item.get("effects", ""),
                "management": item.get("clinical_management", item.get("management", "")),
                "alternatives": item.get("alternatives", "")
            })

    # 3. Deduplicate CYP450 metabolism records
    seen_metab = set()
    for item in retrieval_data.get("metabolism_interactions", []):
        m_key = (
            item.get("substrate", "").strip().lower(),
            item.get("modulator", "").strip().lower(),
            item.get("enzyme", "").strip().upper()
        )
        if m_key not in seen_metab and m_key[0]:
            seen_metab.add(m_key)
            cleaned["metabolism_interactions"].append({
                "substrate": item.get("substrate", "").strip().lower(),
                "modulator": item.get("modulator", "").strip().lower(),
                "enzyme": item.get("enzyme", "").strip().upper(),
                "type": item.get("modulator_type", item.get("interaction_type", "modulator")),
                "severity": item.get("severity", "WARNING"),
                "details": item.get("details", item.get("clinical_details", ""))
            })

    return cleaned


# ============================================================================
# 6. Fallback & Simulation Report Generator (Zero-LLM Mode)
# ============================================================================

def generate_simulation_assessment(
    drugs: List[str],
    conditions: List[str],
    severity: str,
    retrieval_data: Dict[str, Any],
    target_illness: Optional[str] = None,
    medical_history: Optional[str] = None,
    allergies: Optional[str] = None,
    unverified_drugs: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Generates an evidence-grounded, conservative structured report using deterministic logic."""
    severity = normalize_severity(severity)
    algs_list = [a.strip() for a in (allergies or "").split(",") if a.strip() and a.strip().lower() not in ("none", "none reported")]

    fda_int = retrieval_data.get("fda_interactions", [])
    dis_int = retrieval_data.get("disease_contraindications", [])
    met_int = retrieval_data.get("metabolism_interactions", [])

    # DDI Assessment
    if fda_int:
        ddi_lines = [f"{i.get('drug1', '').capitalize()} + {i.get('drug2', '').capitalize()}: {i.get('effects', '')}" for i in fda_int]
        ddi_assessment = f"Detected {len(fda_int)} drug–drug interaction(s): " + "; ".join(ddi_lines)
    elif len(drugs) > 1:
        ddi_assessment = f"No direct drug–drug interaction record was identified between {', '.join([d.capitalize() for d in drugs])} in reference databases."
    elif len(drugs) == 1:
        ddi_assessment = f"Single medication evaluated ({drugs[0].capitalize()}). No co-administered interacting medication detected."
    else:
        ddi_assessment = "No active medications provided for interaction assessment."

    # Drug-Condition Assessment
    if dis_int:
        cond_lines = [f"{c.get('drug', '').capitalize()} with {c.get('disease', '').capitalize()} ({c.get('effects', '')})" for c in dis_int]
        drug_condition_assessment = f"Clinically significant drug–condition concern(s) identified: " + "; ".join(cond_lines)
    elif conditions:
        drug_condition_assessment = (
            f"Patient reported conditions ({', '.join([c.capitalize() for c in conditions])}). "
            f"While no absolute contraindication was retrieved from the local database, clinical review is required "
            f"to verify organ clearance, disease-specific precautions, and tolerance."
        )
    else:
        drug_condition_assessment = "No patient medical conditions or disease states were reported."

    # CYP450 Assessment
    if met_int:
        cyp_lines = [f"{m.get('substrate', '').capitalize()} ({m.get('enzyme', '')}) modulated by {m.get('modulator', '').capitalize()}" for m in met_int]
        cyp450_assessment = f"CYP450 metabolic pathway conflict(s) detected: " + "; ".join(cyp_lines)
    elif len(drugs) > 1:
        cyp450_assessment = "No enzymatic competitive inhibition or induction conflicts were identified among the evaluated medications."
    else:
        cyp450_assessment = "Routine CYP450 pharmacokinetic pathways expected; no competing modulators identified."

    # Unverified drug note
    if unverified_drugs:
        uv_note = f" Medication(s) [{', '.join(unverified_drugs)}] could not be matched against verified clinical databases."
        ddi_assessment += uv_note
        summary = f"REVIEW_REQUIRED: Unverified medication(s) detected ({', '.join(unverified_drugs)}). Clinical safety cannot be confirmed."
        clinical_advisory = "Do not administer unverified medications without consulting authoritative pharmacological monographs or a clinical pharmacist."
        reasoning = "Unverified medications present unknown interaction and toxicity profiles. Safety cannot be assumed."
    else:
        # Summary & Advisory
        if severity == "HIGH_RISK":
            summary = "HIGH_RISK ALERT: Critical clinical risks, major contraindications, or severe interactions detected."
            clinical_advisory = "Regimen presents high-risk clinical dangers. Immediate clinician intervention required to discontinue or substitute contraindicated agents."
            reasoning = "Dangerous drug-drug interactions, critical disease contraindications, or serious allergy concerns identified."
        elif severity == "MODERATE_RISK":
            summary = "MODERATE_RISK ALERT: Clinically significant precautions, drug–condition concerns, or CYP450 effects identified."
            clinical_advisory = "Therapy requires close clinician monitoring, dosage adjustments, or individualized patient risk assessment."
            reasoning = "Clinically significant interactions or metabolic clearance alterations exist that require active clinical oversight."
        elif severity == "REVIEW_REQUIRED":
            summary = "REVIEW_REQUIRED: Clinical information is insufficient to confirm routine safety. Absence of detected interaction is not proof of safety."
            clinical_advisory = "Clinician review is required before treatment, specifically regarding patient condition clearance and allergy status."
            reasoning = "Active patient pathology, unverified allergy profiles, or incomplete pharmacological coverage requires professional human review."
        elif severity == "LOW_RISK":
            summary = "LOW_RISK: Minor or non-serious considerations identified."
            clinical_advisory = "Therapy may proceed under standard clinical monitoring."
            reasoning = "Only minor, non-serious interactions or precautions were identified."
        else:
            summary = "SAFE: Multi-dimensional evaluation (DDI, Conditions, Allergies, CYP450) verified no clinical concerns under standard dosing."
            clinical_advisory = "No clinical contraindications detected across evaluated dimensions under standard dosing protocols."
            reasoning = "Available evidence supports routine use across all assessed safety dimensions."

    # Item details mapping
    details = []
    recommendations = []
    alternatives = []

    for inter in fda_int:
        d1, d2 = inter.get("drug1", "").capitalize(), inter.get("drug2", "").capitalize()
        details.append({
            "drugs": f"{d1} and {d2}",
            "severity": inter.get("severity", "HIGH_RISK"),
            "mechanism": inter.get("mechanism", "Pharmacodynamic interaction"),
            "clinical_effects": inter.get("effects", ""),
            "management": inter.get("clinical_management", inter.get("management", "Consult physician"))
        })
        mgmt = inter.get("clinical_management", inter.get("management", ""))
        if mgmt:
            recommendations.append(f"For {d1} + {d2}: {mgmt}")
        alt = inter.get("alternatives", "")
        if alt:
            alternatives.append({
                "original_drugs": f"{d1} + {d2}",
                "alternative": alt,
                "reasoning": f"Avoids interaction risks between {d1} and {d2}."
            })

    warnings = []
    for contra in dis_int:
        drg, dis = contra.get("drug", "").capitalize(), contra.get("disease", "").capitalize()
        warnings.append({
            "drug": drg,
            "condition": dis,
            "severity": contra.get("severity", "HIGH_RISK"),
            "mechanism": contra.get("mechanism", "Pathological interaction"),
            "clinical_effects": contra.get("effects", ""),
            "management": contra.get("clinical_management", contra.get("management", "Avoid or adjust dose"))
        })
        mgmt = contra.get("clinical_management", contra.get("management", ""))
        if mgmt:
            recommendations.append(f"Regarding {drg} in {dis}: {mgmt}")
        alt = contra.get("alternatives", "")
        if alt:
            alternatives.append({
                "original_drugs": f"{drg} (in {dis})",
                "alternative": alt,
                "reasoning": f"Safer profile for patients with {dis}."
            })

    metabolisms = []
    for metab in met_int:
        sub = metab.get("substrate", "").capitalize()
        mod = metab.get("modulator", "").capitalize()
        enz = metab.get("enzyme", "")
        metabolisms.append({
            "substrate": sub,
            "modulator": mod,
            "enzyme": enz,
            "interaction_type": metab.get("type", metab.get("modulator_type", "CYP modulator")),
            "severity": metab.get("severity", "MODERATE_RISK"),
            "clinical_details": metab.get("details", metab.get("clinical_details", ""))
        })
        recommendations.append(f"CYP450 precaution: {mod} alters clearance of {sub} via {enz}.")

    if not recommendations:
        if severity == "SAFE":
            recommendations.append("Continue prescribed therapy under routine standard monitoring.")
        else:
            recommendations.append("Consult prescribing clinician or clinical pharmacist before administration.")

    citations = [
        "FDA Center for Drug Evaluation and Research Approved Drug Database",
        "CYP450 Enzyme Clearance Consensus Databases",
        "Clinical Practice Guidelines for Disease-Specific Precautions"
    ]

    item_assessments = []
    for d in details:
        item_assessments.append({
            "type": "Drug–Drug Interaction",
            "title": d["drugs"],
            "severity": d["severity"],
            "justification": f"{d['mechanism']}. {d['clinical_effects']}"
        })
    for w in warnings:
        item_assessments.append({
            "type": "Drug–Condition Contraindication",
            "title": f"{w['drug']} in {w['condition']}",
            "severity": w["severity"],
            "justification": f"{w['mechanism']}. {w['clinical_effects']}"
        })
    for m in metabolisms:
        item_assessments.append({
            "type": "CYP450 Metabolism Conflict",
            "title": f"{m['substrate']} + {m['modulator']} ({m['enzyme']})",
            "severity": m["severity"],
            "justification": m["clinical_details"]
        })

    return {
        "severity": severity,
        "reasoning": reasoning,
        "item_assessments": item_assessments,
        "summary": summary,
        "allergies_checked": algs_list,
        "ddi_assessment": ddi_assessment,
        "drug_condition_assessment": drug_condition_assessment,
        "cyp450_assessment": cyp450_assessment,
        "clinical_advisory": clinical_advisory,
        "interactions_details": details,
        "disease_warnings": warnings,
        "metabolism_interactions": metabolisms,
        "recommendations": list(dict.fromkeys(recommendations)),
        "suggested_alternatives": alternatives,
        "citations": list(dict.fromkeys(citations)),
        "engine_mode": "simulation"
    }


# ============================================================================
# 7. CALL 1 System Prompt
# ============================================================================

CALL_1_SYSTEM_INSTRUCTION = """You are the Combined Clinical Assessment Agent in a medication safety pipeline.

Your task is to perform a complete evidence-grounded medication safety assessment in ONE pass.

You have two responsibilities:

1. Determine the overall clinical severity.
2. Generate the complete structured clinical assessment supporting that severity.

The severity and report MUST be internally consistent.

Never generate a reassuring conclusion unless the available evidence actually supports it.

==================================================
SAFETY-FIRST ASSESSMENT
==================================================

Evaluate ALL available dimensions:

1. Drug-drug interactions
2. Drug-condition contraindications and precautions
3. Patient allergies
4. CYP450/metabolic interactions
5. Patient medical history and clinical context

Do not assume that absence of a drug-drug interaction means the regimen is safe.

A medication may be unsafe because of:
- a patient's disease
- an allergy
- altered metabolism
- organ impairment
- medical history
- insufficient information

==================================================
SEVERITY HIERARCHY
==================================================

Use exactly one of:

HIGH_RISK
MODERATE_RISK
LOW_RISK
SAFE
REVIEW_REQUIRED

Apply the highest supported risk level.

HIGH_RISK:
- Major contraindication
- Serious allergy concern
- Dangerous drug-drug interaction
- Potentially life-threatening interaction
- Serious toxicity risk

MODERATE_RISK:
- Clinically significant interaction
- Important precaution
- Meaningful CYP450 exposure change
- Condition-related risk requiring monitoring
- Clinically relevant but non-immediate concern

LOW_RISK:
- Minor interaction
- Minor precaution
- Non-serious consideration
- Monitoring consideration with limited clinical impact

SAFE:
Only use SAFE when the available evidence supports routine use and there are no unresolved concerns across the assessed dimensions.

REVIEW_REQUIRED:
Use when:
- evidence is insufficient
- required patient information is missing
- allergy status is unclear
- a condition cannot be adequately evaluated
- database evidence is inconclusive
- the system cannot confidently determine safety

Never guess in order to produce SAFE.

==================================================
CRITICAL RULE
==================================================

"No drug-drug interaction found" does NOT mean "safe."

For example:

Simvastatin + liver disease

Even if no second interacting drug exists, the patient's condition may require clinical evaluation.

Therefore do not classify such a case as SAFE merely because DDI analysis is negative.

==================================================
EVIDENCE BOUNDARY
==================================================

Use ONLY:

- patient information supplied in the request
- retrieved database context supplied in the request
- established information explicitly available in that context

Do NOT invent:

- drug interactions
- contraindications
- CYP450 pathways
- dosages
- laboratory values
- diagnoses
- patient history
- treatment recommendations
- clinical studies
- citations

If the retrieved evidence does not establish a conclusion, explicitly state that the evidence is insufficient and use REVIEW_REQUIRED when appropriate.

Do not fill fields with fabricated information merely to satisfy the schema.

==================================================
REPORT REQUIREMENTS
==================================================

ddi_assessment:
Provide a standalone assessment of drug-drug interactions.

drug_condition_assessment:
Provide a standalone assessment of drug-condition and allergy concerns.

cyp450_assessment:
Provide a standalone assessment of metabolic/CYP450 interactions.

reasoning:
Explain why the final severity was selected.

summary:
Give a concise overall clinical summary.

clinical_advisory:
Explain the most clinically important next step based on available evidence.

recommendations:
Provide prioritized, concrete actions supported by the evidence.

suggested_alternatives:
Only provide alternatives when a genuine evidence-supported alternative is available.
Otherwise return an empty list.

citations:
Reference only sources actually represented in the supplied retrieval context.
Do not invent study names, URLs, papers, or regulatory documents.

==================================================
CONSERVATIVE DECISION MAKING
==================================================

When uncertain between SAFE and REVIEW_REQUIRED:

choose REVIEW_REQUIRED.

When multiple risk findings exist:

select the highest supported severity.

Do not allow a low-risk finding to override a high-risk finding.

==================================================
OUTPUT
==================================================

Return ONLY structured data matching the provided Pydantic schema.

Do not return markdown.

Do not return explanatory prose outside the structured response."""


# ============================================================================
# 8. Main Entrypoint: CALL 1 Combined Clinical Assessment Agent
# ============================================================================

def run(
    drugs: List[str],
    conditions: List[str],
    retrieval_data: Dict[str, Any],
    target_illness: Optional[str] = None,
    medical_history: Optional[str] = None,
    allergies: Optional[str] = None
) -> AgentStepResult:
    """
    Executes CALL 1 (Combined Clinical Assessment Agent):
    1. Evaluates deterministic baseline across all 5 dimensions.
    2. Checks for unverified/unknown medications.
    3. Optimizes and deduplicates retrieval context (compact serialization).
    4. Runs CALL 1 against the LLM provider chain in a single unified inference.
    5. Applies the deterministic safety floor to guarantee no false reassurance or downgraded risk.
    """
    logs: List[str] = ["Executing Combined Clinical Assessment Agent (CALL 1)..."]

    # 1. Handle Empty Input Edge Cases
    if not drugs:
        logs.append("[Input Validation] No medications supplied. Returning controlled assessment.")
        sim_res = generate_simulation_assessment(
            drugs=[], conditions=conditions, severity="REVIEW_REQUIRED",
            retrieval_data={"fda_interactions": [], "disease_contraindications": [], "metabolism_interactions": []},
            target_illness=target_illness, medical_history=medical_history, allergies=allergies
        )
        return AgentStepResult(
            agent_name="Combined Clinical Assessment Agent",
            description="Performs unified evidence-grounded clinical assessment and severity scoring in ONE pass.",
            input_data={"drugs": drugs, "conditions": conditions, "allergies": allergies},
            output_data=sim_res,
            logs=logs
        )

    # 2. Check for Unknown / Unverified Medications
    unverified_drugs = identify_unverified_drugs(drugs)
    if unverified_drugs:
        logs.append(f"[Unknown Drug Warning] Unverified medications detected: {unverified_drugs}")

    # 3. Always compute the deterministic baseline first
    det_severity, det_reason, det_items = evaluate_deterministic_severity(
        retrieval_data, drugs=drugs, conditions=conditions, allergies=allergies
    )
    logs.append(f"[Deterministic Baseline] Calculated safety floor: {det_severity}")

    # 4. If unknown drugs exist and cannot be evaluated, enforce controlled REVIEW_REQUIRED
    # to avoid LLM hallucination of unknown properties
    if unverified_drugs:
        logs.append("[Controlled Safeguard] Halting LLM inference for unverified medication to prevent hallucinated pharmacology.")
        controlled_res = generate_simulation_assessment(
            drugs=drugs, conditions=conditions, severity="REVIEW_REQUIRED",
            retrieval_data=retrieval_data, target_illness=target_illness,
            medical_history=medical_history, allergies=allergies,
            unverified_drugs=unverified_drugs
        )
        controlled_res["drugs_recognized"] = retrieval_data.get("drugs_recognized", [])
        controlled_res["drugs_unrecognized"] = retrieval_data.get("drugs_unrecognized", [])
        return AgentStepResult(
            agent_name="Combined Clinical Assessment Agent",
            description="Performs unified evidence-grounded clinical assessment and severity scoring in ONE pass.",
            input_data={"drugs": drugs, "conditions": conditions, "allergies": allergies},
            output_data=controlled_res,
            logs=logs
        )

    # 5. Token Optimization: Filter and deduplicate retrieved context
    optimized_retrieval = optimize_retrieval_context(retrieval_data)

    # 6. Fallback simulation baseline
    sim_assessment = generate_simulation_assessment(
        drugs=drugs, conditions=conditions, severity=det_severity,
        retrieval_data=optimized_retrieval, target_illness=target_illness,
        medical_history=medical_history, allergies=allergies
    )
    final_output = sim_assessment

    # 7. Execute CALL 1 if AI is active
    if is_ai_active():
        try:
            logs.append("Phase 1: Generating unified severity and clinical advisory report (CALL 1)...")
            
            # Compact JSON serialization
            compact_retrieval = json.dumps(optimized_retrieval, separators=(",", ":"))

            prompt = (
                f"Active Medications:\n{drugs}\n\n"
                f"Patient Health Conditions:\n{conditions or 'None reported'}\n\n"
                f"Patient Allergies:\n{allergies or 'None reported'}\n\n"
                f"Target Illness:\n{target_illness or 'Not specified'}\n\n"
                f"Medical History:\n{medical_history or 'None reported'}\n\n"
                f"Retrieved Clinical Evidence:\n{compact_retrieval}\n\n"
                "Using ONLY the supplied patient information and retrieved evidence:\n\n"
                "1. Assess drug-drug interactions.\n"
                "2. Assess drug-condition and allergy risks.\n"
                "3. Assess CYP450/metabolic interactions.\n"
                "4. Consider relevant medical history.\n"
                "5. Determine the highest supported severity.\n"
                "6. Generate the complete structured clinical report.\n\n"
                "Do not infer unsupported medical facts.\n"
                "Do not invent evidence.\n"
                "Do not classify unresolved safety concerns as SAFE."
            )

            response_text, provider = call_llm(
                prompt,
                CALL_1_SYSTEM_INSTRUCTION,
                response_schema=CombinedClinicalAssessmentSchema
            )

            if provider in ("simulation", "simulation_fallback") or not response_text:
                raise RuntimeError(f"Switched to offline fallback ({provider})")

            # Parse and validate structured output
            parsed = json.loads(response_text)
            
            # Pydantic schema validation
            validated_model = CombinedClinicalAssessmentSchema(**parsed)
            ai_data = validated_model.model_dump()

            ai_sev = normalize_severity(ai_data.get("severity", det_severity))
            ai_data["severity"] = ai_sev
            ai_reason = ai_data.get("reasoning", "")

            # 8. Deterministic Safety Floor Resolution
            final_sev, final_reason = resolve_safety_floor(
                ai_severity=ai_sev,
                det_severity=det_severity,
                ai_reasoning=ai_reason,
                det_reasoning=det_reason
            )

            if final_sev != ai_sev:
                logs.append(
                    f"[Safety Floor Applied] AI assigned '{ai_sev}', but deterministic clinical rules require "
                    f"'{final_sev}'. Overrode severity to prevent under-calling."
                )
                ai_data["severity"] = final_sev
                ai_data["reasoning"] = f"{final_reason} (AI original note: {ai_reason})"
            else:
                logs.append(f"[Validated] AI severity '{ai_sev}' aligns with clinical safety floor.")

            ai_data["engine_mode"] = f"live_ai:{provider}"
            final_output = ai_data
            logs.append(f"[CALL 1 Success] Completed via {provider.upper()}. Final Severity: {final_output['severity']}")

        except ValidationError as ve:
            logs.append(f"[Validation Warning] Model output did not fully match Pydantic schema ({ve}). Using safe deterministic report.")
            final_output = sim_assessment
            final_output["engine_mode"] = "fallback_static"
        except Exception as e:
            logs.append(f"[Fallback] CALL 1 failed ({str(e)}). Gracefully falling back to deterministic clinical report.")
            final_output = sim_assessment
            final_output["engine_mode"] = "fallback_static"
    else:
        logs.append("Operating in offline simulation mode. Deterministic clinical database rules applied.")
        final_output["engine_mode"] = "simulation"

    # Forward recognized and unrecognized RxNorm metadata
    final_output["drugs_recognized"] = retrieval_data.get("drugs_recognized", [])
    final_output["drugs_unrecognized"] = retrieval_data.get("drugs_unrecognized", [])

    return AgentStepResult(
        agent_name="Combined Clinical Assessment Agent",
        description="Performs unified evidence-grounded clinical assessment and severity scoring in ONE pass.",
        input_data={
            "drugs": drugs,
            "conditions": conditions,
            "target_illness": target_illness,
            "medical_history": medical_history,
            "allergies": allergies
        },
        output_data=final_output,
        logs=logs
    )
