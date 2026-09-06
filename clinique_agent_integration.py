"""
Clinique AI Agent Integration: Input & Retrieval Agents with RxNorm
================================================================================
Drop-in module providing RxNorm-backed InputAgent and RetrievalAgent.
Distinguishes SAFE, REVIEW_REQUIRED, CONTRAINDICATED, and DATA_UNAVAILABLE.
"""

import json
import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from backend.agents.clinique_rxnorm import RxNormClient, DrugRecognized, DrugUnrecognized


class AssessmentStatus(str, Enum):
    SAFE = "SAFE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    CONTRAINDICATED = "CONTRAINDICATED"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"


class DataAvailability(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    NONE = "none"


@dataclass
class InputAgentOutput:
    """Structured output from Input Agent (drug extraction + RxNorm normalization)."""
    raw_input: str
    drugs_recognized: List[Dict[str, Any]]
    drugs_unrecognized: List[Dict[str, Any]]
    extraction_confidence: float

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class DrugInteractionRecord:
    drug1: str
    drug2: str
    severity: str
    mechanism: str
    management: str
    data_source: str = "curated"


@dataclass
class DrugContraindicationRecord:
    drug: str
    condition: str
    severity: str
    reason: str
    management: str
    data_source: str = "curated"


@dataclass
class DrugAssessmentResult:
    drug_name: str
    rxcui: str
    status: AssessmentStatus
    data_availability: DataAvailability
    interactions: List[DrugInteractionRecord] = None
    contraindications: List[DrugContraindicationRecord] = None
    message: str = ""

    def __post_init__(self):
        if self.interactions is None:
            self.interactions = []
        if self.contraindications is None:
            self.contraindications = []

    def to_dict(self):
        return {
            "drug_name": self.drug_name,
            "rxcui": self.rxcui,
            "status": self.status.value,
            "data_availability": self.data_availability.value,
            "interactions": [asdict(i) for i in self.interactions],
            "contraindications": [asdict(c) for c in self.contraindications],
            "message": self.message
        }


@dataclass
class RetrievalAgentOutput:
    input_drugs_recognized: List[Dict[str, Any]]
    input_drugs_unrecognized: List[Dict[str, Any]]
    assessments: List[DrugAssessmentResult]
    summary_status: AssessmentStatus
    note: str

    def to_dict(self):
        return {
            "input_drugs_recognized": self.input_drugs_recognized,
            "input_drugs_unrecognized": self.input_drugs_unrecognized,
            "assessments": [a.to_dict() for a in self.assessments],
            "summary_status": self.summary_status.value,
            "note": self.note
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)


class InputAgent:
    """RxNorm-backed Input Agent for extracting and normalizing drug mentions."""
    def __init__(self):
        self.rxnorm_client = RxNormClient()
        self.drug_pattern = re.compile(
            r'\b(?:on|taking|prescribed|started|stopped|allergic to|allergy to)\s+([a-zA-Z0-9\s,\-]+?)(?:\.|,|;|$)',
            re.IGNORECASE
        )

    def extract_drug_mentions(self, raw_text: str) -> List[str]:
        mentions = []
        matches = self.drug_pattern.findall(raw_text)
        for match in matches:
            drugs = [d.strip() for d in match.split(",") if d.strip()]
            mentions.extend(drugs)

        if "medications:" in raw_text.lower() or "drugs:" in raw_text.lower():
            lower = raw_text.lower()
            idx = lower.find("medications:") if "medications:" in lower else lower.find("drugs:")
            after = raw_text[idx:].split(":", 1)[-1].split("\n")[0]
            drugs = [d.strip() for d in after.split(",") if d.strip()]
            mentions.extend(drugs)

        # Also parse comma-delimited phrases
        if not mentions:
            candidates = [d.strip() for d in re.split(r'[,;]|\band\b', raw_text) if d.strip()]
            for c in candidates:
                words = c.split()
                if len(words) <= 3:
                    mentions.append(c)

        return list(dict.fromkeys(mentions))

    def run(self, raw_input: str) -> InputAgentOutput:
        drug_mentions = self.extract_drug_mentions(raw_input)
        if not drug_mentions:
            return InputAgentOutput(
                raw_input=raw_input,
                drugs_recognized=[],
                drugs_unrecognized=[],
                extraction_confidence=0.0
            )

        recognized, unrecognized = self.rxnorm_client.normalize_drug_list(drug_mentions)
        recognized_dicts = [
            {
                "original_mention": d.name,
                "rxcui": d.rxcui,
                "preferred_name": d.preferred_name,
                "match_type": d.match_type
            }
            for d in recognized
        ]
        unrecognized_dicts = [
            {
                "original_mention": d.name,
                "reason": d.reason
            }
            for d in unrecognized
        ]

        total = len(recognized) + len(unrecognized)
        confidence = len(recognized) / total if total > 0 else 0.0

        return InputAgentOutput(
            raw_input=raw_input,
            drugs_recognized=recognized_dicts,
            drugs_unrecognized=unrecognized_dicts,
            extraction_confidence=confidence
        )
