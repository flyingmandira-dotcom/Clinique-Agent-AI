import logging
from typing import List, Dict, Any
from backend.schemas import AgentStepResult
from backend.agents.severity_agent import normalize_severity

logger = logging.getLogger("drug_checker.agents.output")

STANDARD_DISCLAIMER = (
    "CLINICAL DECISION-SUPPORT NOTICE: This system is a Multi-Agent GenAI decision-support prototype "
    "designed for educational, informational, and demonstration purposes. It does NOT constitute medical diagnosis "
    "or prescribing advice. The absence of detected interactions is not a guarantee of safety. All clinical decisions "
    "must be made in consultation with a licensed physician, clinical pharmacist, or qualified healthcare professional."
)

def run(guard_data: Dict[str, Any]) -> AgentStepResult:
    """Executes the Output Agent (Agent 6) to compile, validate, and standardize the final clinical report."""
    logs = ["Preparing final multi-dimensional clinical output compilation..."]
    
    report: Dict[str, Any] = guard_data.get("validated_report", {})
    grounding_score: float = guard_data.get("grounding_score", 1.0)
    is_safe: bool = guard_data.get("is_safe", True)
    
    logs.append("Standardizing severity rating and attaching clinical decision-support disclaimers...")
    
    # 1. Standardize and normalize severity rating
    report["severity"] = normalize_severity(report.get("severity", "REVIEW_REQUIRED"))
    
    # 2. Attach standard medical disclaimer to recommendations
    recommendations: List[str] = report.get("recommendations", [])
    if STANDARD_DISCLAIMER not in recommendations:
        recommendations.insert(0, STANDARD_DISCLAIMER)
        
    # 3. Adjust recommendations if grounding or safety alert was raised
    if not is_safe or grounding_score < 0.9:
        logs.append("[Grounding Quality Alert] Appending verification note to output.")
        caution_msg = f"SAFETY NOTICE: Some AI-generated statements had lower database grounding ({round(grounding_score * 100, 1)}%). Prioritize clinician evaluation."
        if caution_msg not in recommendations:
            recommendations.insert(1, caution_msg)
            
    report["recommendations"] = recommendations
    
    # 4. Ensure all separated assessment fields are populated with clear defaults if missing
    if not report.get("ddi_assessment"):
        report["ddi_assessment"] = "No drug–drug interactions evaluated or reported."
    if not report.get("drug_condition_assessment"):
        report["drug_condition_assessment"] = "No drug–condition contraindications or precautions reported."
    if not report.get("cyp450_assessment"):
        report["cyp450_assessment"] = "No CYP450 metabolic pathway conflicts reported."
    if not report.get("clinical_advisory"):
        report["clinical_advisory"] = report.get("summary", "Clinician review recommended prior to administration.")
    if "allergies_checked" not in report:
        report["allergies_checked"] = []
        
    # Ensure citations exist
    citations: List[str] = report.get("citations", [])
    if not citations:
        citations = [
            "FDA Center for Drug Evaluation and Research Approved Drug Database",
            "CYP450 Enzyme Clearance Consensus Databases",
            "Clinical Practice Guidelines for Disease-Specific Precautions"
        ]
    report["citations"] = citations
    
    # Ensure engine mode is populated
    report["engine_mode"] = report.get("engine_mode", "simulation")
    
    logs.append(f"Output Agent final compilation completed. Final Severity: {report['severity']}")
    
    return AgentStepResult(
        agent_name="Output Agent",
        description="Formats final clinical report, applies multi-dimensional standards, and attaches disclaimers.",
        input_data={"guard_data": {"is_safe": is_safe, "grounding_score": grounding_score}},
        output_data=report,
        logs=logs
    )
