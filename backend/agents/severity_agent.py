import logging
from typing import List, Dict, Any, Tuple, Optional
from backend.schemas import AgentStepResult
from backend.agents.generation_severity_agent import (
    normalize_severity,
    evaluate_deterministic_severity,
    SEVERITY_RANK,
    CLINICAL_RISK_RANK,
    CombinedClinicalAssessmentSchema,
    run as combined_run
)

logger = logging.getLogger("drug_checker.agents.severity")

# Standardized clinical risk hierarchy (retained for backward compatibility)
RISK_HIERARCHY = SEVERITY_RANK

# Re-export schema for backward compatibility
from pydantic import BaseModel, Field

class SeverityAssessmentSchema(BaseModel):
    overall_severity: str = Field(
        description="Unified interaction severity. MUST be HIGH_RISK, MODERATE_RISK, LOW_RISK, SAFE, or REVIEW_REQUIRED."
    )
    reasoning: str = Field(description="A concise clinical reasoning explaining the assigned severity across all dimensions.")
    item_assessments: List[Dict[str, Any]] = Field(description="List of clinical assessments for each identified issue.")


def run(
    retrieval_data: Dict[str, Any],
    drugs: Optional[List[str]] = None,
    conditions: Optional[List[str]] = None,
    allergies: Optional[str] = None
) -> AgentStepResult:
    """
    Backward-compatibility runner for legacy callers expecting standalone Severity Agent.
    Runs deterministic baseline calculation.
    """
    logger.info("[Compatibility Wrapper] Standalone severity_agent.run() called.")
    det_overall, det_reason, det_items = evaluate_deterministic_severity(
        retrieval_data, drugs=drugs, conditions=conditions, allergies=allergies
    )
    return AgentStepResult(
        agent_name="Severity Agent (Compatibility Mode)",
        description="Assesses pharmacodynamic/pharmacokinetic risks across clinical safety dimensions.",
        input_data={"retrieval_data": retrieval_data, "drugs": drugs, "conditions": conditions, "allergies": allergies},
        output_data={
            "overall_severity": det_overall,
            "reasoning": det_reason,
            "item_assessments": det_items
        },
        logs=[
            f"Compatibility Mode: Evaluated deterministic severity baseline: {det_overall}"
        ]
    )

