import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field
from backend.schemas import AgentStepResult
from backend.agents.llm import call_llm, is_ai_active

logger = logging.getLogger("drug_checker.agents.severity")

# Standardized clinical risk hierarchy
# Higher index = higher risk level
RISK_HIERARCHY = {
    "SAFE": 0,
    "LOW_RISK": 1,
    "REVIEW_REQUIRED": 2,
    "MODERATE_RISK": 3,
    "HIGH_RISK": 4
}

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


class SeverityAssessmentSchema(BaseModel):
    overall_severity: str = Field(
        description="Unified interaction severity. MUST be HIGH_RISK, MODERATE_RISK, LOW_RISK, SAFE, or REVIEW_REQUIRED."
    )
    reasoning: str = Field(description="A concise clinical reasoning explaining the assigned severity across all dimensions.")
    item_assessments: List[Dict[str, Any]] = Field(description="List of clinical assessments for each identified issue.")


def evaluate_deterministic_severity(
    retrieval_data: Dict[str, Any],
    drugs: Optional[List[str]] = None,
    conditions: Optional[List[str]] = None,
    allergies: Optional[str] = None
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Evaluates clinical severity across ALL safety dimensions:
    1. Patient allergy / absolute contraindication -> HIGH_RISK
    2. Drug-condition contraindication -> HIGH_RISK or MODERATE_RISK
    3. Major drug-drug interaction -> HIGH_RISK
    4. Significant CYP450 metabolism conflict -> MODERATE_RISK or HIGH_RISK
    5. Minor interaction -> LOW_RISK
    6. Presence of patient conditions with unconfirmed clearance -> REVIEW_REQUIRED
    7. Truly clean across all dimensions -> SAFE
    """
    items: List[Dict[str, Any]] = []
    drugs = drugs or []
    conditions = conditions or []
    highest_severity = "SAFE"

    def upgrade_severity(candidate: str):
        nonlocal highest_severity
        norm_candidate = normalize_severity(candidate)
        if RISK_HIERARCHY.get(norm_candidate, 0) > RISK_HIERARCHY.get(highest_severity, 0):
            highest_severity = norm_candidate

    # 1. Evaluate Drug-Drug Interactions (FDA)
    ddi_count = len(retrieval_data.get("fda_interactions", []))
    for inter in retrieval_data.get("fda_interactions", []):
        sev = normalize_severity(inter.get("severity", "WARNING"))
        upgrade_severity(sev)
        items.append({
            "type": "Drug–Drug Interaction",
            "title": f"{inter.get('drug1', '').capitalize()} + {inter.get('drug2', '').capitalize()}",
            "severity": sev,
            "justification": f"Mechanism: {inter.get('mechanism', 'N/A')}. Effect: {inter.get('effects', 'N/A')}"
        })

    # 2. Evaluate Drug-Disease Contraindications and Allergies
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

    # 3. Evaluate CYP450 Pharmacokinetic Metabolism Conflicts
    for metab in retrieval_data.get("metabolism_interactions", []):
        sev = normalize_severity(metab.get("severity", "WARNING"))
        upgrade_severity(sev)
        items.append({
            "type": "CYP450 Metabolism Conflict",
            "title": f"{metab.get('substrate', '').capitalize()} metabolized by {metab.get('enzyme', '')} inhibited/induced by {metab.get('modulator', '').capitalize()}",
            "severity": sev,
            "justification": metab.get("details", "")
        })

    # 4. CRITICAL SAFETY RULE: Never classify as SAFE if patient has conditions or allergies
    # Even if no DDI or database record was matched, an active patient disease requires clinical review!
    if highest_severity == "SAFE" and conditions:
        # Conditions are present, but no direct DB contraindication rule matched
        highest_severity = "REVIEW_REQUIRED"
        reason = (
            f"No direct drug–drug interaction was detected among {', '.join(drugs) if drugs else 'the medications'}. "
            f"However, the patient has reported health conditions ({', '.join(conditions)}) which require specific "
            f"clinician review before treatment. The absence of a detected database match is not proof of safety."
        )
        return highest_severity, reason, items

    if highest_severity == "SAFE" and allergies and allergies.strip().lower() not in ("none", "none reported"):
        highest_severity = "REVIEW_REQUIRED"
        reason = (
            f"No drug–drug interaction was found, but the patient profile indicates allergies ({allergies}). "
            f"Specific cross-reactivity and excipient verification is required prior to administration."
        )
        return highest_severity, reason, items

    # Build concise reasoning for the highest severity found
    if highest_severity == "HIGH_RISK":
        reason = (
            "A HIGH_RISK clinical concern has been detected! Critical contraindications, serious allergy warnings, "
            "or severe drug–drug interactions were identified that present life-threatening dangers (e.g. fatal hypotension, "
            "acute organ failure, or severe hemorrhages). The regimen should be halted or replaced under immediate medical direction."
        )
    elif highest_severity == "MODERATE_RISK":
        reason = (
            "MODERATE_RISK considerations detected. Clinically significant drug interactions, drug–condition precautions, "
            "or CYP450 metabolism clearance alterations exist. Monitoring, dose adjustment, or clinical consultation is advised."
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


def run(
    retrieval_data: Dict[str, Any],
    drugs: Optional[List[str]] = None,
    conditions: Optional[List[str]] = None,
    allergies: Optional[str] = None
) -> AgentStepResult:
    """Executes the Severity Agent (Agent 3) enforcing strict multi-dimensional risk hierarchy."""
    logs = ["Analyzing retrieved data packages across all 5 safety dimensions..."]
    
    # Always compute deterministic scores first as the authoritative clinical safety baseline
    det_overall, det_reason, det_items = evaluate_deterministic_severity(
        retrieval_data, drugs=drugs, conditions=conditions, allergies=allergies
    )
    
    overall_severity = det_overall
    reasoning = det_reason
    item_assessments = det_items
    
    if is_ai_active():
        try:
            logs.append("Contacting LLM provider chain for multi-dimensional clinical risk scoring...")
            system_instruction = (
                "You are the Severity Agent in a clinical drug safety pipeline.\n"
                "CRITICAL SAFETY RULE: Never classify a regimen as 'SAFE' merely because no drug-drug interaction was detected.\n"
                "You must evaluate ALL dimensions: Drug-drug interactions, Drug-condition contraindications/precautions, "
                "Allergies, CYP450 metabolism, and patient history.\n"
                "SEVERITY TIERS:\n"
                "- HIGH_RISK: Major contraindications, serious allergy alerts, dangerous DDIs, life-threatening concerns.\n"
                "- MODERATE_RISK: Clinically significant interactions, condition precautions, CYP450 exposure alterations.\n"
                "- LOW_RISK: Minor/non-serious considerations.\n"
                "- SAFE: ONLY when all dimensions are verified clean and support routine use.\n"
                "- REVIEW_REQUIRED: Insufficient information or unverified condition impact. Never guess; choose REVIEW_REQUIRED.\n"
                "The highest risk category from any dimension ALWAYS determines the overall severity.\n"
                "Output JSON matching the schema."
            )
            prompt = (
                f"Active Drugs: {drugs or 'None'}\n"
                f"Patient Conditions: {conditions or 'None'}\n"
                f"Patient Allergies: {allergies or 'None reported'}\n\n"
                f"Retrieved Database Context Chunks:\n{json.dumps(retrieval_data, indent=2)}"
            )
            
            response_text, provider = call_llm(prompt, system_instruction, response_schema=SeverityAssessmentSchema)
            if provider in ("simulation", "simulation_fallback") or not response_text:
                raise RuntimeError(f"Switched to offline fallback ({provider})")
            parsed = json.loads(response_text)
            
            ai_raw_severity = parsed.get("overall_severity", "REVIEW_REQUIRED")
            ai_severity = normalize_severity(ai_raw_severity)
            
            # ENFORCE CONSERVATIVE SAFETY HIERARCHY:
            # Deterministic safety rule floor cannot be downgraded by AI
            det_level = RISK_HIERARCHY.get(det_overall, 0)
            ai_level = RISK_HIERARCHY.get(ai_severity, 0)
            
            if det_level > ai_level:
                logs.append(
                    f"[Override Safety Floor] AI evaluated severity as '{ai_severity}', but deterministic rules found "
                    f"'{det_overall}'. Enforcing '{det_overall}' to prevent falsely reassuring the clinician."
                )
                overall_severity = det_overall
            else:
                overall_severity = ai_severity
                
            reasoning = parsed.get("reasoning", det_reason)
            item_assessments = parsed.get("item_assessments", det_items)
            logs.append(f"[Success] {provider.upper()} completed multi-dimensional risk scoring: {overall_severity}")
            
        except Exception as e:
            logs.append(f"[Fallback] LLM scoring failed: {str(e)}. Defaulting to clinical rule processor.")
            logs.append(f"Fallback Severity: {overall_severity}")
    else:
        logs.append("Executing clinical rule engine...")
        logs.append(f"Calculated Multi-Dimensional Severity: {overall_severity}")
        
    output_data = {
        "overall_severity": overall_severity,
        "reasoning": reasoning,
        "item_assessments": item_assessments
    }
    
    return AgentStepResult(
        agent_name="Severity Agent",
        description="Assesses pharmacodynamic/pharmacokinetic risks across all 5 clinical safety dimensions.",
        input_data={"retrieval_data": retrieval_data, "drugs": drugs, "conditions": conditions, "allergies": allergies},
        output_data=output_data,
        logs=logs
    )
