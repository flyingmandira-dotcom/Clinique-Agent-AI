import logging
from typing import List, Dict, Any, Optional
from backend.schemas import AgentStepResult
from backend.agents.generation_severity_agent import (
    generate_simulation_assessment,
    normalize_severity,
    run as combined_run
)

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
    """Backward-compatibility shim for generate_simulation_report."""
    return generate_simulation_assessment(
        drugs=drugs,
        conditions=conditions,
        severity=severity,
        retrieval_data=retrieval_data,
        target_illness=target_illness,
        medical_history=medical_history,
        allergies=allergies
    )


def run(
    drugs: List[str],
    conditions: List[str],
    severity_data: Dict[str, Any],
    retrieval_data: Dict[str, Any],
    target_illness: Optional[str] = None,
    medical_history: Optional[str] = None,
    allergies: Optional[str] = None
) -> AgentStepResult:
    """
    Backward-compatibility shim for legacy callers expecting standalone Generation Agent.
    Delegates to combined assessment.
    """
    logger.info("[Compatibility Wrapper] Standalone generation_agent.run() called. Delegating to combined runner.")
    return combined_run(
        drugs=drugs,
        conditions=conditions,
        retrieval_data=retrieval_data,
        target_illness=target_illness,
        medical_history=medical_history,
        allergies=allergies
    )

