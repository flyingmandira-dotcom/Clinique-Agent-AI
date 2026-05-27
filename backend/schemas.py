from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class CheckRequest(BaseModel):
    query: str = Field(..., description="The user's query containing drugs and health conditions.")
    drugs: Optional[List[str]] = Field(None, description="Optional pre-extracted list of drugs.")
    conditions: Optional[List[str]] = Field(None, description="Optional pre-extracted list of conditions.")
    target_illness: Optional[str] = Field(None, description="The illness or symptom the patient is trying to treat.")
    medical_history: Optional[str] = Field(None, description="The patient's medical history / previous diagnoses.")
    allergies: Optional[str] = Field(None, description="The patient's known drug or environmental allergies.")

class AgentStepResult(BaseModel):
    agent_name: str
    description: str
    status: str = "success"
    input_data: Any
    output_data: Any
    logs: List[str]

class ClinicalReport(BaseModel):
    severity: str = Field(..., description="Overall interaction severity: CRITICAL, WARNING, or SAFE")
    summary: str = Field(..., description="High-level summary of the interactions found.")
    interactions_details: List[Dict[str, Any]] = Field(..., description="Detailed clinical breakdown of each interaction.")
    disease_warnings: List[Dict[str, Any]] = Field(..., description="Breakdown of drug-disease contraindications.")
    metabolism_interactions: List[Dict[str, Any]] = Field(..., description="Pharmacokinetic CYP450 interactions.")
    recommendations: List[str] = Field(..., description="Clinical action items and safe alternatives.")
    suggested_alternatives: List[Dict[str, Any]] = Field(default=[], description="Suggested safer drug alternatives with fewer or no interaction risks.")
    citations: List[str] = Field(..., description="Source citations/references.")

class CheckResponse(BaseModel):
    drugs: List[str]
    conditions: List[str]
    severity: str
    report: ClinicalReport
    pipeline_steps: List[AgentStepResult]
