import json
import logging
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from backend.schemas import AgentStepResult
from backend.agents.llm import call_llm, is_ai_active

logger = logging.getLogger("drug_checker.agents.severity")

class SeverityAssessmentSchema(BaseModel):
    overall_severity: str = Field(description="Unified interaction severity. MUST be CRITICAL, WARNING, or SAFE.")
    reasoning: str = Field(description="A concise clinical reasoning explaining the assigned severity.")
    item_assessments: List[Dict[str, Any]] = Field(description="List of clinical assessments for each identified issue.")

def evaluate_deterministic_severity(retrieval_data: Dict[str, Any]) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Applies standard clinical guidelines to determine severity rules."""
    items = []
    has_critical = False
    has_warning = False
    
    # Process FDA
    for inter in retrieval_data.get("fda_interactions", []):
        sev = inter["severity"].upper()
        if sev == "CRITICAL":
            has_critical = True
        elif sev == "WARNING":
            has_warning = True
        items.append({
            "type": "Drug-Drug Interaction",
            "title": f"{inter['drug1'].capitalize()} + {inter['drug2'].capitalize()}",
            "severity": sev,
            "justification": f"Mechanism: {inter['mechanism']}. Effect: {inter['effects']}"
        })
        
    # Process Disease
    for contra in retrieval_data.get("disease_contraindications", []):
        sev = contra["severity"].upper()
        if sev == "CRITICAL":
            has_critical = True
        elif sev == "WARNING":
            has_warning = True
        items.append({
            "type": "Drug-Disease Contraindication",
            "title": f"{contra['drug'].capitalize()} in {contra['disease'].capitalize()}",
            "severity": sev,
            "justification": f"Mechanism: {contra['mechanism']}. Effect: {contra['effects']}"
        })
        
    # Process Metabolism
    for metab in retrieval_data.get("metabolism_interactions", []):
        sev = metab["severity"].upper()
        if sev == "CRITICAL":
            has_critical = True
        elif sev == "WARNING":
            has_warning = True
        items.append({
            "type": "CYP450 Metabolism Conflict",
            "title": f"{metab['substrate'].capitalize()} metabolized by {metab['enzyme']} inhibited/induced by {metab['modulator'].capitalize()}",
            "severity": sev,
            "justification": metab["details"]
        })
        
    # Final overall score
    if has_critical:
        overall = "CRITICAL"
        reason = "A critical risk has been detected! The combined medications or drug-disease conditions present high-risk clinical dangers (e.g. fatal hypotension, severe hemorrhages) and should be avoided or replaced."
    elif has_warning:
        overall = "WARNING"
        reason = "Potential clinical risks detected. The combined medications or drug-disease conditions may lead to side effects or reduced efficacy. Monitoring, dosage spacing, or adjustments are advised."
    else:
        overall = "SAFE"
        reason = "No high-risk clinical interactions or contraindications were found in the database. The therapy appears safe to proceed under general medical guidelines."
        
    return overall, reason, items

def run(retrieval_data: Dict[str, Any]) -> AgentStepResult:
    """Executes the Severity Agent (Agent 3) to score and prioritize interactions."""
    logs = ["Analyzing retrieved data packages for severity evaluation..."]
    
    # Always compute deterministic scores first as the reference safety baseline
    det_overall, det_reason, det_items = evaluate_deterministic_severity(retrieval_data)
    
    overall_severity = det_overall
    reasoning = det_reason
    item_assessments = det_items
    
    if is_ai_active():
        try:
            logs.append("Contacting Gemini for clinical risk scoring and reasoning...")
            system_instruction = (
                "You are the Severity Agent in a clinical drug interaction pipeline. "
                "Analyze the retrieved interaction chunks and assign an overall severity score: "
                "CRITICAL, WARNING, or SAFE. Provide structured logical reasoning for this score "
                "based on pharmacodynamics and pharmacokinetics. Return JSON matching the schema."
            )
            prompt = f"Retrieved Context Data:\n{json.dumps(retrieval_data, indent=2)}"
            
            response_text = call_llm(prompt, system_instruction, response_schema=SeverityAssessmentSchema)
            parsed = json.loads(response_text)
            
            # Ensure safety constraint: if deterministic checks found CRITICAL, enforce CRITICAL
            ai_severity = parsed.get("overall_severity", "SAFE").upper()
            if det_overall == "CRITICAL" and ai_severity != "CRITICAL":
                logs.append("[Override] Gemini evaluated severity lower than local clinical rules. Overriding to 'CRITICAL' for safety.")
                overall_severity = "CRITICAL"
            else:
                overall_severity = ai_severity
                
            reasoning = parsed.get("reasoning", det_reason)
            item_assessments = parsed.get("item_assessments", det_items)
            logs.append(f"[Success] Gemini completed risk scoring. Severity: {overall_severity}")
            
        except Exception as e:
            logs.append(f"[Error] Gemini scoring failed: {str(e)}. Defaulting to clinical rule processor.")
            logs.append(f"Fallback Severity: {overall_severity}")
    else:
        logs.append("Executing clinical rule engine...")
        logs.append(f"Calculated Severity: {overall_severity}")
        
    output_data = {
        "overall_severity": overall_severity,
        "reasoning": reasoning,
        "item_assessments": item_assessments
    }
    
    return AgentStepResult(
        agent_name="Severity Agent",
        description="Assesses pharmacodynamic/pharmacokinetic risks to classify clinical severity.",
        input_data=retrieval_data,
        output_data=output_data,
        logs=logs
    )
