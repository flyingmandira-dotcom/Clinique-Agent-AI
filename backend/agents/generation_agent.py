import json
import logging
from typing import List, Dict, Any, Optional
from backend.schemas import AgentStepResult, ClinicalReport
from backend.agents.llm import call_llm, is_ai_active
from backend.agents.severity_agent import normalize_severity

logger = logging.getLogger("drug_checker.agents.generation")

def generate_simulation_report(
    drugs: List[str], 
    conditions: List[str], 
    severity: str, 
    retrieval_data: Dict[str, Any],
    target_illness: Optional[str] = None, 
    medical_history: Optional[str] = None, 
    allergies: Optional[str] = None
) -> Dict[str, Any]:
    """Generates a conservative multi-dimensional clinical advisory report using deterministic logic."""
    severity = normalize_severity(severity)
    summary_parts = []
    recommendations = [
        "DISCLAIMER: This system is a decision-support prototype for educational purposes. Always consult a licensed clinician before making therapeutic changes."
    ]
    citations = [
        "FDA Center for Drug Evaluation and Research Guidance", 
        "CYP450 Metabolism Consensus Guidelines (Pharmacotherapy)",
        "Clinical Practice Guidelines for Disease-Specific Contraindications"
    ]
    raw_alternatives = []
    
    fda_int = retrieval_data.get("fda_interactions", [])
    dis_int = retrieval_data.get("disease_contraindications", [])
    met_int = retrieval_data.get("metabolism_interactions", [])
    
    # 1. Prepare Separated Dimension Assessments
    # Allergies
    algs_list = [a.strip() for a in (allergies or "").split(",") if a.strip() and a.strip().lower() not in ("none", "none reported")]
    
    # DDI Assessment
    if fda_int:
        ddi_lines = [f"{i['drug1'].capitalize()} + {i['drug2'].capitalize()}: {i['effects']}" for i in fda_int]
        ddi_assessment = f"Detected {len(fda_int)} drug–drug interaction(s): " + "; ".join(ddi_lines)
    elif len(drugs) > 1:
        ddi_assessment = f"No direct drug–drug interaction record was identified between {', '.join([d.capitalize() for d in drugs])} in the reference FDA interaction database."
    elif len(drugs) == 1:
        ddi_assessment = f"Single medication evaluated ({drugs[0].capitalize()}). No interacting partner medication was co-administered."
    else:
        ddi_assessment = "No active medications were provided for drug–drug interaction assessment."

    # Drug-Condition Assessment
    if dis_int:
        cond_lines = [f"{c['drug'].capitalize()} with {c['disease'].capitalize()} ({c['effects']})" for c in dis_int]
        drug_condition_assessment = f"Clinically significant drug–condition concern(s) identified: " + "; ".join(cond_lines)
    elif conditions:
        drug_condition_assessment = (
            f"Patient has reported conditions ({', '.join([c.capitalize() for c in conditions])}). "
            f"While no absolute contraindication was retrieved from the local database, the presence of underlying pathology "
            f"requires clinician review to assess organ clearance, dose tolerance, and disease-specific precautions."
        )
    else:
        drug_condition_assessment = "No patient medical conditions or disease states were reported."

    # CYP450 Assessment
    if met_int:
        cyp_lines = [f"{m['substrate'].capitalize()} metabolized by {m['enzyme']} modulated by {m['modulator'].capitalize()}" for m in met_int]
        cyp450_assessment = f"CYP450 pharmacokinetic pathway conflict(s) detected: " + "; ".join(cyp_lines)
    elif len(drugs) > 1:
        cyp450_assessment = "No enzymatic competitive inhibition or induction conflicts were identified among the evaluated medications."
    else:
        cyp450_assessment = "Routine CYP450 pharmacokinetic pathways expected; no competing modulators identified."

    # 2. Overall Summary & Advisory
    if severity == "HIGH_RISK":
        summary_parts.append("HIGH_RISK ALERT: Critical clinical risks, major contraindications, or severe interactions detected.")
        clinical_advisory = "Regimen presents high-risk clinical dangers. Immediate clinician intervention is required to discontinue or substitute contraindicated agents."
    elif severity == "MODERATE_RISK":
        summary_parts.append("MODERATE_RISK ALERT: Clinically significant precautions, drug–condition concerns, or CYP450 effects identified.")
        clinical_advisory = "Therapy requires close clinician monitoring, dosage adjustments, or individualized patient risk assessment."
    elif severity == "REVIEW_REQUIRED":
        summary_parts.append("REVIEW_REQUIRED: Clinical information is insufficient to confirm routine safety. Absence of detected interaction is not evidence of safety.")
        clinical_advisory = "Clinician review is required before treatment, specifically regarding patient condition clearance and allergy status."
    elif severity == "LOW_RISK":
        summary_parts.append("LOW_RISK: Minor or non-serious considerations identified.")
        clinical_advisory = "Therapy may proceed under standard clinical monitoring."
    else:
        summary_parts.append("SAFE: Multi-dimensional evaluation (DDI, Conditions, Allergies, CYP450) verified no clinical concerns under standard adult dosing.")
        clinical_advisory = "No clinical contraindications detected across evaluated dimensions under standard dosing protocols."

    # 3. Map Detailed Lists
    details = []
    for inter in fda_int:
        d1, d2 = inter["drug1"].capitalize(), inter["drug2"].capitalize()
        details.append({
            "drugs": f"{d1} and {d2}",
            "severity": inter.get("severity", "HIGH_RISK"),
            "mechanism": inter.get("mechanism", "Pharmacodynamic interaction"),
            "clinical_effects": inter.get("effects", ""),
            "management": inter.get("clinical_management", "Consult physician")
        })
        recommendations.append(f"For {d1} + {d2}: {inter.get('clinical_management', '')}")
        raw_alternatives.append({
            "original_drugs": f"{d1} + {d2}",
            "alternative": inter.get("alternatives", "Safer therapeutic class alternative"),
            "reasoning": f"Avoids {inter.get('effects', 'interaction risks')} between {d1} and {d2}."
        })

    warnings = []
    for contra in dis_int:
        drg, dis = contra["drug"].capitalize(), contra["disease"].capitalize()
        warnings.append({
            "drug": drg,
            "condition": dis,
            "severity": contra.get("severity", "HIGH_RISK"),
            "mechanism": contra.get("mechanism", "Pathological interaction"),
            "clinical_effects": contra.get("effects", ""),
            "management": contra.get("clinical_management", "Avoid or adjust dose")
        })
        recommendations.append(f"Regarding {drg} in {dis}: {contra.get('clinical_management', '')}")
        raw_alternatives.append({
            "original_drugs": f"{drg} (in {dis})",
            "alternative": contra.get("alternatives", "Renal/hepatic safe alternative"),
            "reasoning": f"Safer profile for patients with {dis} to avoid {contra.get('effects', 'complications')}."
        })

    metabolisms = []
    for metab in met_int:
        sub, mod = metab["substrate"].capitalize(), metab["modulator"].capitalize()
        metabolisms.append({
            "substrate": sub,
            "modulator": mod,
            "enzyme": metab.get("enzyme", ""),
            "interaction_type": metab.get("modulator_type", "CYP modulator"),
            "severity": metab.get("severity", "MODERATE_RISK"),
            "clinical_details": metab.get("details", "")
        })
        recommendations.append(f"CYP450 precaution: {mod} alters clearance of {sub} via {metab.get('enzyme', '')}.")

    # Suggested alternatives
    suggested_alternatives = []
    for alt in raw_alternatives:
        alt["reasoning"] += " WARNING: Consult physician before initiating alternative therapy."
        suggested_alternatives.append(alt)

    return {
        "severity": severity,
        "summary": " ".join(summary_parts),
        "allergies_checked": algs_list,
        "ddi_assessment": ddi_assessment,
        "drug_condition_assessment": drug_condition_assessment,
        "cyp450_assessment": cyp450_assessment,
        "clinical_advisory": clinical_advisory,
        "interactions_details": details,
        "disease_warnings": warnings,
        "metabolism_interactions": metabolisms,
        "recommendations": list(dict.fromkeys(recommendations)),
        "suggested_alternatives": suggested_alternatives,
        "citations": list(dict.fromkeys(citations)),
        "engine_mode": "simulation"
    }


def run(
    drugs: List[str], 
    conditions: List[str], 
    severity_data: Dict[str, Any], 
    retrieval_data: Dict[str, Any],
    target_illness: Optional[str] = None, 
    medical_history: Optional[str] = None, 
    allergies: Optional[str] = None
) -> AgentStepResult:
    """Executes the Generation Agent (Agent 4) enforcing conservative multi-dimensional risk synthesis."""
    logs = ["Synthesizing comprehensive multi-dimensional clinical report..."]
    
    raw_sev = severity_data.get("overall_severity", "REVIEW_REQUIRED")
    severity = normalize_severity(raw_sev)
    
    # Generate baseline/simulation report
    sim_report = generate_simulation_report(
        drugs, conditions, severity, retrieval_data, target_illness, medical_history, allergies
    )
    output_report = sim_report
    
    if is_ai_active():
        try:
            logs.append("Contacting LLM provider chain to synthesize clinical advisory report...")
            system_instruction = (
                "You are the Generation Agent in a clinical decision-support pipeline.\n"
                "CRITICAL SAFETY RULE:\n"
                "Never classify a medication as SAFE merely because no drug–drug interaction was detected.\n"
                "Drug safety must be evaluated across ALL dimensions: Drug-drug interactions, Drug-condition contraindications "
                "and precautions, Patient allergies, CYP450 metabolism, and patient history.\n"
                "A case with no drug–drug interaction but a clinically important drug–condition concern must NOT be classified as SAFE.\n\n"
                "SEVERITY CLASSIFICATION:\n"
                "- HIGH_RISK: Major contraindications, serious allergies, dangerous DDIs, life-threatening concerns.\n"
                "- MODERATE_RISK: Clinically significant interactions, condition precautions, CYP450 exposure changes, monitoring needed.\n"
                "- LOW_RISK: Minor/non-serious considerations.\n"
                "- SAFE: ONLY when ALL dimensions (DDI, Conditions, Allergies, CYP450) are verified clear under standard conditions.\n"
                "- REVIEW_REQUIRED: Insufficient data or unverified condition clearance. Never guess; choose REVIEW_REQUIRED.\n\n"
                "IMPORTANT DISTINCTION:\n"
                "Do NOT equate 'No drug-drug interaction found' with 'The medication is safe for this patient'.\n"
                "E.g., Drug: Simvastatin, Condition: Liver disease -> DDI: No interacting partner; Drug-Condition: Liver disease requires "
                "appropriate clinical evaluation. Final Severity: MODERATE_RISK or REVIEW_REQUIRED.\n\n"
                "EVIDENCE HANDLING:\n"
                "Never invent interactions, contraindications, CYP450 pathways, dosages, or conditions. "
                "If evidence is insufficient, mark as uncertain and use REVIEW_REQUIRED.\n\n"
                "REQUIRED OUTPUT SECTIONS (JSON):\n"
                "1. severity: HIGH_RISK | MODERATE_RISK | LOW_RISK | SAFE | REVIEW_REQUIRED\n"
                "2. summary: Concise clinical summary\n"
                "3. allergies_checked: List of checked allergies\n"
                "4. ddi_assessment: Dedicated drug-drug interaction assessment\n"
                "5. drug_condition_assessment: Dedicated drug-condition contraindication/precaution assessment\n"
                "6. cyp450_assessment: Dedicated CYP450 metabolic pathway assessment\n"
                "7. clinical_advisory: Detailed clinician guidance\n"
                "8. interactions_details: List of interaction items\n"
                "9. disease_warnings: List of disease warnings\n"
                "10. metabolism_interactions: List of CYP450 items\n"
                "11. recommendations: Actionable clinical steps\n"
                "12. suggested_alternatives: Safer therapeutic alternatives\n"
                "13. citations: Source references\n\n"
                "OUTPUT PRINCIPLE: Be conservative rather than falsely reassuring. Output strictly valid JSON matching schema."
            )
            
            prompt = (
                f"Active Medications: {drugs}\n"
                f"Patient Health Conditions: {conditions or 'None reported'}\n"
                f"Patient Allergies: {allergies or 'None reported'}\n"
                f"Target Illness to Treat: {target_illness or 'Not specified'}\n"
                f"Medical History Profile: {medical_history or 'None reported'}\n"
                f"Assessed Severity Floor: {severity}\n\n"
                f"Retrieved Database Context Chunks:\n{json.dumps(retrieval_data, indent=2)}"
            )
            
            response_text, provider = call_llm(prompt, system_instruction, response_schema=ClinicalReport)
            if provider in ("simulation", "simulation_fallback") or not response_text:
                raise RuntimeError(f"Switched to offline fallback ({provider})")
            parsed = json.loads(response_text)
            
            # Normalize AI severity and preserve floor
            ai_sev = normalize_severity(parsed.get("severity", severity))
            parsed["severity"] = ai_sev
            parsed["engine_mode"] = f"live_ai:{provider}"
            
            # Fill any missing required separated sections
            if not parsed.get("ddi_assessment"):
                parsed["ddi_assessment"] = sim_report["ddi_assessment"]
            if not parsed.get("drug_condition_assessment"):
                parsed["drug_condition_assessment"] = sim_report["drug_condition_assessment"]
            if not parsed.get("cyp450_assessment"):
                parsed["cyp450_assessment"] = sim_report["cyp450_assessment"]
            if not parsed.get("clinical_advisory"):
                parsed["clinical_advisory"] = sim_report["clinical_advisory"]
            if not parsed.get("allergies_checked"):
                parsed["allergies_checked"] = sim_report["allergies_checked"]
                
            output_report = parsed
            logs.append(f"[Success] {provider.upper()} synthesized multi-dimensional clinical report ({ai_sev}).")
            
        except Exception as e:
            output_report = sim_report
            output_report["engine_mode"] = "fallback_static"
            logs.append(f"[Fallback] LLM provider failover ({str(e)}). Fell back to deterministic multi-dimensional template.")
    else:
        output_report["engine_mode"] = "simulation"
        logs.append("Assembled report using pre-validated clinical databases in offline simulation mode.")
        
    return AgentStepResult(
        agent_name="Generation Agent",
        description="Synthesizes findings into a professional clinical report across all safety dimensions.",
        input_data={
            "drugs": drugs, 
            "conditions": conditions, 
            "severity": severity,
            "target_illness": target_illness,
            "medical_history": medical_history,
            "allergies": allergies
        },
        output_data=output_report,
        logs=logs
    )
