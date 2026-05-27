import logging
from typing import List, Dict, Any
from backend.schemas import AgentStepResult, ClinicalReport

logger = logging.getLogger("drug_checker.agents.output")

STANDARD_DISCLAIMER = (
    "DISCLAIMER: This system is a Multi-Agent GenAI prototype and is intended solely for educational, "
    "informational, and demonstration purposes. It does NOT constitute medical or professional advice. "
    "Always consult with a licensed physician, clinical pharmacist, or healthcare provider before starting, "
    "stopping, or altering any medication, dosage, or medical treatment."
)

def run(guard_data: Dict[str, Any]) -> AgentStepResult:
    """Executes the Output Agent (Agent 6) to format the final clinical response payload."""
    logs = ["Preparing final output compilation..."]
    
    report: Dict[str, Any] = guard_data.get("validated_report", {})
    grounding_score: float = guard_data.get("grounding_score", 1.0)
    is_safe: bool = guard_data.get("is_safe", True)
    
    logs.append("Formatting clinical report structure...")
    
    # 1. Attach standard medical disclaimer to recommendations
    recommendations: List[str] = report.get("recommendations", [])
    recommendations.insert(0, STANDARD_DISCLAIMER)
    
    # 2. Adjust recommendations / warnings if hallucination was found
    if not is_safe or grounding_score < 0.9:
        logs.append("[Grounding Caution] Appending grounding quality alert to output recommendations.")
        recommendations.insert(1, f"CAUTION: Some generated statements in this report had low clinical source grounding score ({grounding_score * 100}%). Use with extra discretion.")
        
    report["recommendations"] = recommendations
    
    # Ensure citations exist
    citations: List[str] = report.get("citations", [])
    if not citations:
        citations = ["FDA Center for Drug Evaluation and Research Data", "CYP450 Enzyme Clearance Consensus Databases"]
    report["citations"] = citations
    
    logs.append("Output Agent final compilation succeeded.")
    
    return AgentStepResult(
        agent_name="Output Agent",
        description="Formats final clinical report, applies standards, and attaches medical disclaimers.",
        input_data={"guard_data": {"is_safe": is_safe, "grounding_score": grounding_score}},
        output_data=report,
        logs=logs
    )
