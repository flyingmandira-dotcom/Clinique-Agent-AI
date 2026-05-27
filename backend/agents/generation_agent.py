import json
import logging
from typing import List, Dict, Any
from backend.schemas import AgentStepResult, ClinicalReport
from backend.agents.llm import call_llm, is_ai_active

logger = logging.getLogger("drug_checker.agents.generation")

def generate_simulation_report(drugs: List[str], conditions: List[str], severity: str, retrieval_data: Dict[str, Any],
                               target_illness: str = None, medical_history: str = None, allergies: str = None) -> Dict[str, Any]:
    """Generates a highly structured clinical advisory report using template rendering."""
    summary_parts = []
    recommendations = ["Always consult with your prescribing physician before altering your medication regimen."]
    citations = ["FDA Orange Book: Approved Drug Products with Therapeutic Equivalence Evaluations", "CYP450 Metabolism Consensus Guidelines (Pharmacotherapy 2020)"]
    raw_alternatives = []
    
    # Analyze issues
    fda_int = retrieval_data.get("fda_interactions", [])
    dis_int = retrieval_data.get("disease_contraindications", [])
    met_int = retrieval_data.get("metabolism_interactions", [])
    
    # 1. Build Summary
    if severity == "CRITICAL":
        summary_parts.append(f"CRITICAL RISK DETECTED: The requested combination of drugs ({', '.join([d.capitalize() for d in drugs])}) or medical conditions ({', '.join([c.capitalize() for c in conditions])}) presents high-risk clinical dangers.")
    elif severity == "WARNING":
        summary_parts.append(f"WARNING: Moderate risk detected. Spacing of doses, monitoring of vital signs, or minor therapy adjustments are recommended for {', '.join([d.capitalize() for d in drugs])}.")
    else:
        summary_parts.append("SAFE: No high-risk or moderate-risk interaction pathways were located in the current databases. The combination appears safe under standard adult dosing guidelines.")
        
    # 2. Map Details & Recommendations
    details = []
    for inter in fda_int:
        d1, d2 = inter["drug1"].capitalize(), inter["drug2"].capitalize()
        details.append({
            "drugs": f"{d1} and {d2}",
            "severity": inter["severity"],
            "mechanism": inter["mechanism"],
            "clinical_effects": inter["effects"],
            "management": inter["clinical_management"]
        })
        recommendations.append(f"For {d1} and {d2}: {inter['clinical_management']}")
        recommendations.append(f"Consider therapeutic alternative: Use {inter['alternatives']}.")
        
        raw_alternatives.append({
            "original_drugs": f"{d1} + {d2}",
            "alternative": inter["alternatives"],
            "reasoning": f"Replaces the interacting combination of {d1} and {d2} to avoid: {inter['effects'].lower()}."
        })
        citations.append(f"FDA Approved Labeling & Drug Interaction Guidance Database ({d1} / {d2})")
        
    warnings = []
    for contra in dis_int:
        drg, dis = contra["drug"].capitalize(), contra["disease"].capitalize()
        warnings.append({
            "drug": drg,
            "condition": dis,
            "severity": contra["severity"],
            "mechanism": contra["mechanism"],
            "clinical_effects": contra["effects"],
            "management": contra["clinical_management"]
        })
        recommendations.append(f"Regarding {drg} in {dis}: {contra['clinical_management']}")
        recommendations.append(f"Avoid {drg}. Alternative recommendation: {contra['alternatives']}.")
        
        # Avoid duplicating custom allergy/disease warnings as standard drug-drug alternatives
        if "allergy" not in dis.lower():
            raw_alternatives.append({
                "original_drugs": f"{drg} (with {dis})",
                "alternative": contra["alternatives"],
                "reasoning": f"Safer pharmacological profile for patients suffering from {dis} to avoid: {contra['effects'].lower()}."
            })
            citations.append(f"Clinical Practice Guidelines for Disease State Management in {dis} ({drg})")
        
    metabolisms = []
    for metab in met_int:
        sub, mod = metab["substrate"].capitalize(), metab["modulator"].capitalize()
        metabolisms.append({
            "substrate": sub,
            "modulator": mod,
            "enzyme": metab["enzyme"],
            "interaction_type": metab["modulator_type"],
            "severity": metab["severity"],
            "clinical_details": metab["details"]
        })
        recommendations.append(f"Pharmacokinetic CYP450 caution: {mod} affects the enzymatic clearance of {sub} via {metab['enzyme']}. Adjust dosing or hold {sub}.")
        
    if severity == "SAFE":
        recommendations.append("No special dietary or administrative spacing requirements are indicated.")
        
    # Apply dynamic patient context (history, allergies, illness) filtering for simulated alternatives
    suggested_alternatives = []
    for alt in raw_alternatives:
        alt_name = alt["alternative"].lower()
        original = alt["original_drugs"].lower()
        
        is_allergic = False
        if allergies:
            allergy_terms = [a.strip().lower() for a in allergies.split(",") if a.strip()]
            for allergy in allergy_terms:
                if allergy and (allergy in alt_name or alt_name in allergy):
                    is_allergic = True
                    break
        
        is_contraindicated = False
        if medical_history:
            history_lower = medical_history.lower()
            if "renal" in history_lower or "kidney" in history_lower:
                if "ibuprofen" in alt_name or "aspirin" in alt_name or "nsaid" in alt_name:
                    is_contraindicated = True
            if "asthma" in history_lower:
                if "propranolol" in alt_name or "beta blocker" in alt_name:
                    is_contraindicated = True
            if "ulcer" in history_lower:
                if "ibuprofen" in alt_name or "aspirin" in alt_name:
                    is_contraindicated = True
                    
        if is_allergic:
            if "acetaminophen" in alt_name or "paracetamol" in alt_name:
                alt["alternative"] = "Tramadol (low dose) or Topical Lidocaine"
                alt["reasoning"] = f"Replaces interacting NSAIDs. Acetaminophen alternative was avoided due to patient allergy."
            else:
                alt["alternative"] = "Consult prescribing specialist"
                alt["reasoning"] = f"Standard alternatives like {alt['alternative']} were excluded due to patient allergy profile."
        elif is_contraindicated:
            if "acetaminophen" not in original:
                alt["alternative"] = "Acetaminophen (paracetamol)"
                alt["reasoning"] = f"Switched to Acetaminophen to avoid NSAIDs due to patient history of kidney/ulcer issues."
            else:
                alt["alternative"] = "Consult healthcare provider"
                alt["reasoning"] = f"Alternative therapy selected to avoid NSAIDs due to kidney/ulcer history."
                
        if target_illness and "illness" not in alt["reasoning"]:
            alt["reasoning"] += f" Suitable for treating reported illness: '{target_illness}'."
            
        # Append doctor disclaimer specifically for alternatives
        alt["reasoning"] += " WARNING: Consult a doctor before starting this alternative."
        suggested_alternatives.append(alt)
        
    # Deduplicate lists
    recommendations = list(dict.fromkeys(recommendations))
    citations = list(dict.fromkeys(citations))
    
    return {
        "severity": severity,
        "summary": " ".join(summary_parts),
        "interactions_details": details,
        "disease_warnings": warnings,
        "metabolism_interactions": metabolisms,
        "recommendations": recommendations,
        "suggested_alternatives": suggested_alternatives,
        "citations": citations
    }
 
def run(drugs: List[str], conditions: List[str], severity_data: Dict[str, Any], retrieval_data: Dict[str, Any],
        target_illness: str = None, medical_history: str = None, allergies: str = None) -> AgentStepResult:
    """Executes the Generation Agent (Agent 4) to compile a rich medical report."""
    logs = ["Beginning synthesis of clinical advisory report..."]
    
    severity = severity_data.get("overall_severity", "SAFE")
    
    # Generate baseline/simulation report
    sim_report = generate_simulation_report(drugs, conditions, severity, retrieval_data, target_illness, medical_history, allergies)
    
    output_report = sim_report
    
    if is_ai_active():
        try:
            logs.append("Contacting Gemini to synthesize clinical advisory report...")
            system_instruction = (
                "You are the Generation Agent in a clinical drug interaction pipeline. "
                "Synthesize a highly professional, scientifically rigorous clinical advisory report. "
                "Combine the extracted drugs, medical conditions, retrieved context database chunks, "
                "and severity score. Write a clear summary, expand on the pharmacological mechanisms "
                "for all interactions, detail drug-disease warnings, discuss CYP450 metabolism issues, "
                "suggest drug alternatives that are safer and have fewer or no interaction risks, especially "
                "tailored to treat the patient's target illness (if specified), and strictly avoiding any "
                "drugs that are contraindicated by their medical history or could trigger their listed allergies. "
                "Also draft specific actionable recommendations and citations. Output JSON matching the specified schema."
            )
            
            prompt = (
                f"Drugs: {drugs}\n"
                f"Conditions: {conditions}\n"
                f"Severity: {severity}\n"
                f"Patient Profile Context:\n"
                f"  - Target Illness to Treat: {target_illness or 'Not specified'}\n"
                f"  - Medical History: {medical_history or 'None reported'}\n"
                f"  - Known Allergies: {allergies or 'None reported'}\n\n"
                f"Retrieved Database context chunks:\n{json.dumps(retrieval_data, indent=2)}"
            )
            
            response_text = call_llm(prompt, system_instruction, response_schema=ClinicalReport)
            parsed = json.loads(response_text)
            
            # Populate response
            output_report = parsed
            logs.append("[Success] Gemini generated a highly descriptive, clinical-grade advisory note.")
            
        except Exception as e:
            logs.append(f"[Error] Gemini synthesis failed: {str(e)}. Falling back to deterministic structured template.")
    else:
        logs.append("Assembled report using pre-validated clinical databases.")
        
    return AgentStepResult(
        agent_name="Generation Agent",
        description="Synthesizes findings into a professional clinical report with recommendations.",
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
