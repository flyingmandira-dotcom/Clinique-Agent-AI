from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class CheckRequest(BaseModel):
    query: str = Field(..., max_length=1000, description="The user's query containing drugs and health conditions.")
    drugs: Optional[List[str]] = Field(None, max_length=50, description="Optional pre-extracted list of drugs.")
    conditions: Optional[List[str]] = Field(None, max_length=50, description="Optional pre-extracted list of conditions.")
    target_illness: Optional[str] = Field(None, max_length=200, description="The illness or symptom the patient is trying to treat.")
    medical_history: Optional[str] = Field(None, max_length=500, description="The patient's medical history / previous diagnoses.")
    allergies: Optional[str] = Field(None, max_length=500, description="The patient's known drug or environmental allergies.")

class AgentStepResult(BaseModel):
    agent_name: str
    description: str
    status: str = "success"
    input_data: Any
    output_data: Any
    logs: List[str] = Field(default_factory=list)

class ClinicalReport(BaseModel):
    severity: str = Field(
        default="SAFE", 
        description="Overall interaction severity: HIGH_RISK, MODERATE_RISK, LOW_RISK, SAFE, or REVIEW_REQUIRED."
    )
    summary: str = Field(default="", description="High-level summary of the interactions found.")
    
    # Separated assessment dimensions
    allergies_checked: List[str] = Field(default_factory=list, description="Patient allergies audited during assessment.")
    ddi_assessment: str = Field(default="", description="Dedicated assessment of drug-drug interactions.")
    drug_condition_assessment: str = Field(default="", description="Dedicated assessment of drug-condition contraindications and precautions.")
    cyp450_assessment: str = Field(default="", description="Dedicated assessment of CYP450 pharmacokinetic and metabolic pathways.")
    clinical_advisory: str = Field(default="", description="Comprehensive clinical advisory note and guidance.")
    
    # Detailed structured item lists
    interactions_details: List[Any] = Field(default_factory=list, description="Detailed clinical breakdown of each interaction.")
    disease_warnings: List[Any] = Field(default_factory=list, description="Breakdown of drug-disease contraindications.")
    metabolism_interactions: List[Any] = Field(default_factory=list, description="Pharmacokinetic CYP450 interactions.")
    recommendations: List[str] = Field(default_factory=list, description="Clinical action items and safe alternatives.")
    suggested_alternatives: List[Any] = Field(default_factory=list, description="Suggested safer drug alternatives with fewer or no interaction risks.")
    citations: List[str] = Field(default_factory=list, description="Source citations/references.")
    engine_mode: str = Field(default="simulation", description="Engine mode used: e.g. live_ai:groq, live_ai:gemini, fallback_static, or simulation.")

class CheckResponse(BaseModel):
    drugs: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    severity: str
    report: ClinicalReport
    pipeline_steps: List[AgentStepResult] = Field(default_factory=list)
