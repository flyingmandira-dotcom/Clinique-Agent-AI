import unittest
import json
from unittest.mock import patch

from backend.agents.generation_severity_agent import (
    evaluate_deterministic_severity,
    normalize_severity,
    resolve_safety_floor,
    identify_unverified_drugs,
    optimize_retrieval_context,
    CombinedClinicalAssessmentSchema,
    generate_simulation_assessment,
    SEVERITY_RANK,
    CLINICAL_RISK_RANK,
    run as combined_assessment_run
)
from backend.agents.severity_agent import RISK_HIERARCHY
from backend.agents import retrieval_agent, hallucination_guard, output_agent
from backend.schemas import ClinicalReport


class TestClinicalSafetyArchitecture(unittest.TestCase):

    # ------------------------------------------------------------------------
    # 1. SAFE: Paracetamol + Cetirizine
    # ------------------------------------------------------------------------
    def test_safe_paracetamol_cetirizine(self):
        """Paracetamol + Cetirizine with no conditions/allergies should be evaluated as SAFE."""
        drugs = ["paracetamol", "cetirizine"]
        conditions = []
        allergies = None

        retrieval_res = retrieval_agent.run(drugs, conditions, allergies=allergies)
        retrieval_data = retrieval_res.output_data

        sev, reason, items = evaluate_deterministic_severity(
            retrieval_data, drugs=drugs, conditions=conditions, allergies=allergies
        )

        self.assertEqual(sev, "SAFE")
        self.assertIn("routine use", reason.lower())

        # Test combined agent in simulation mode
        step_res = combined_assessment_run(drugs, conditions, retrieval_data, allergies=allergies)
        self.assertEqual(step_res.output_data["severity"], "SAFE")

    # ------------------------------------------------------------------------
    # 2. DDI: Warfarin + Ibuprofen (HIGH_RISK)
    # ------------------------------------------------------------------------
    def test_ddi_warfarin_ibuprofen(self):
        """Warfarin + Ibuprofen presents major bleeding interaction -> HIGH_RISK."""
        drugs = ["warfarin", "ibuprofen"]
        retrieval_res = retrieval_agent.run(drugs, [])
        retrieval_data = retrieval_res.output_data

        sev, reason, items = evaluate_deterministic_severity(retrieval_data, drugs=drugs)
        self.assertEqual(sev, "HIGH_RISK")
        self.assertTrue(any("bleeding" in str(item).lower() or "hemorrhage" in str(item).lower() or "ibuprofen" in str(item).lower() for item in items))

        step_res = combined_assessment_run(drugs, [], retrieval_data)
        self.assertEqual(step_res.output_data["severity"], "HIGH_RISK")
        self.assertTrue(len(step_res.output_data["interactions_details"]) > 0)

    # ------------------------------------------------------------------------
    # 3. CYP450: Simvastatin + Clarithromycin (HIGH_RISK)
    # ------------------------------------------------------------------------
    def test_cyp450_simvastatin_clarithromycin(self):
        """Simvastatin (CYP3A4 substrate) + Clarithromycin (strong inhibitor) -> HIGH_RISK."""
        drugs = ["simvastatin", "clarithromycin"]
        retrieval_res = retrieval_agent.run(drugs, [])
        retrieval_data = retrieval_res.output_data

        sev, reason, items = evaluate_deterministic_severity(retrieval_data, drugs=drugs)
        self.assertEqual(sev, "HIGH_RISK")
        self.assertTrue(any("cyp" in str(item).lower() or "metabol" in str(item).lower() for item in items))

        step_res = combined_assessment_run(drugs, [], retrieval_data)
        self.assertEqual(step_res.output_data["severity"], "HIGH_RISK")
        self.assertTrue(len(step_res.output_data["metabolism_interactions"]) > 0)

    # ------------------------------------------------------------------------
    # 4. Drug-Condition: Ibuprofen + Peptic Ulcer Disease / GI Bleeding (HIGH_RISK)
    # ------------------------------------------------------------------------
    def test_drug_condition_ibuprofen_peptic_ulcer(self):
        """Ibuprofen in a patient with peptic ulcer disease -> HIGH_RISK contraindication."""
        drugs = ["ibuprofen"]
        conditions = ["peptic ulcer disease"]

        retrieval_res = retrieval_agent.run(drugs, conditions)
        retrieval_data = retrieval_res.output_data

        sev, reason, items = evaluate_deterministic_severity(retrieval_data, drugs=drugs, conditions=conditions)
        self.assertEqual(sev, "HIGH_RISK")
        self.assertTrue(any("ulcer" in str(item).lower() for item in items))

        step_res = combined_assessment_run(drugs, conditions, retrieval_data)
        self.assertEqual(step_res.output_data["severity"], "HIGH_RISK")

    # ------------------------------------------------------------------------
    # 5. Allergy: Amoxicillin + Penicillin Allergy (HIGH_RISK)
    # ------------------------------------------------------------------------
    def test_allergy_amoxicillin_penicillin(self):
        """Amoxicillin with severe penicillin allergy profile -> HIGH_RISK."""
        drugs = ["amoxicillin"]
        conditions = []
        allergies = "penicillin, amoxicillin"

        retrieval_res = retrieval_agent.run(drugs, conditions, allergies=allergies)
        retrieval_data = retrieval_res.output_data

        sev, reason, items = evaluate_deterministic_severity(
            retrieval_data, drugs=drugs, conditions=conditions, allergies=allergies
        )
        self.assertEqual(sev, "HIGH_RISK")
        self.assertTrue(any("allergy" in str(item).lower() for item in items))

    # ------------------------------------------------------------------------
    # 6. Unknown Drug: XYZabutinol (REVIEW_REQUIRED)
    # ------------------------------------------------------------------------
    def test_unknown_drug_xyzabutinol(self):
        """Unverified drug must NOT be guessed; must return REVIEW_REQUIRED."""
        drugs = ["XYZabutinol"]
        unverified = identify_unverified_drugs(drugs)
        self.assertIn("XYZabutinol", unverified)

        retrieval_data = {"fda_interactions": [], "disease_contraindications": [], "metabolism_interactions": []}
        sev, reason, items = evaluate_deterministic_severity(retrieval_data, drugs=drugs)
        self.assertEqual(sev, "REVIEW_REQUIRED")
        self.assertIn("could not be verified", reason.lower())

        step_res = combined_assessment_run(drugs, [], retrieval_data)
        self.assertEqual(step_res.output_data["severity"], "REVIEW_REQUIRED")
        self.assertIn("XYZabutinol", step_res.output_data["summary"])

    # ------------------------------------------------------------------------
    # 7. Polypharmacy: Warfarin + Amiodarone + Aspirin + Ibuprofen (HIGH_RISK)
    # ------------------------------------------------------------------------
    def test_polypharmacy_multiple_interactions(self):
        """Polypharmacy regimen with multiple compounding bleed risks must trigger HIGH_RISK."""
        drugs = ["warfarin", "amiodarone", "aspirin", "ibuprofen"]
        retrieval_res = retrieval_agent.run(drugs, [])
        retrieval_data = retrieval_res.output_data

        sev, reason, items = evaluate_deterministic_severity(retrieval_data, drugs=drugs)
        self.assertEqual(sev, "HIGH_RISK")
        self.assertGreaterEqual(len(items), 2)

        step_res = combined_assessment_run(drugs, [], retrieval_data)
        self.assertEqual(step_res.output_data["severity"], "HIGH_RISK")

    # ------------------------------------------------------------------------
    # 8. Missing Information / Unregistered Condition (REVIEW_REQUIRED)
    # ------------------------------------------------------------------------
    def test_missing_info_unregistered_condition(self):
        """Active patient pathology without a direct DB match requires REVIEW_REQUIRED, not SAFE."""
        drugs = ["paracetamol"]
        conditions = ["rare undiagnosed metabolic syndrome"]
        allergies = None

        retrieval_data = {"fda_interactions": [], "disease_contraindications": [], "metabolism_interactions": []}
        sev, reason, items = evaluate_deterministic_severity(
            retrieval_data, drugs=drugs, conditions=conditions, allergies=allergies
        )
        self.assertEqual(sev, "REVIEW_REQUIRED")
        self.assertIn("absence of a detected database match is not proof of safety", reason.lower())

    # ------------------------------------------------------------------------
    # 9. Deterministic Safety Floor Override (Prevents Under-Calling)
    # ------------------------------------------------------------------------
    def test_deterministic_safety_floor_override(self):
        """If AI claims SAFE or LOW_RISK but rules detect HIGH_RISK, floor forces HIGH_RISK."""
        ai_sev = "SAFE"
        det_sev = "HIGH_RISK"
        ai_reason = "Model inferred no problem."
        det_reason = "Critical fatal contraindication detected."

        final_sev, final_reason = resolve_safety_floor(ai_sev, det_sev, ai_reason, det_reason)
        self.assertEqual(final_sev, "HIGH_RISK")
        self.assertEqual(final_reason, det_reason)

    def test_deterministic_safety_floor_preserves_review_required_over_false_safe(self):
        """If deterministic checker requires REVIEW_REQUIRED, AI cannot falsely claim SAFE."""
        ai_sev = "SAFE"
        det_sev = "REVIEW_REQUIRED"
        ai_reason = "Model claimed safe."
        det_reason = "Patient has unverified conditions requiring clinician review."

        final_sev, final_reason = resolve_safety_floor(ai_sev, det_sev, ai_reason, det_reason)
        self.assertEqual(final_sev, "REVIEW_REQUIRED")
        self.assertEqual(final_reason, det_reason)

    def test_deterministic_safety_floor_allows_ai_risk_escalation(self):
        """If deterministic is LOW_RISK, AI can escalate to MODERATE_RISK or HIGH_RISK."""
        ai_sev = "MODERATE_RISK"
        det_sev = "LOW_RISK"
        ai_reason = "Clinical context indicates elevated risk."
        det_reason = "Only minor baseline interaction."

        final_sev, final_reason = resolve_safety_floor(ai_sev, det_sev, ai_reason, det_reason)
        self.assertEqual(final_sev, "MODERATE_RISK")
        self.assertEqual(final_reason, ai_reason)

    # ------------------------------------------------------------------------
    # 10. Hallucination Guard Catches False SAFE (CALL 2)
    # ------------------------------------------------------------------------
    def test_hallucination_guard_catches_false_safe(self):
        """Hallucination Guard must override false 'SAFE' when retrieved clinical concerns exist."""
        false_report = {
            "severity": "SAFE",
            "summary": "Everything is completely safe!",
            "recommendations": ["No concerns."],
            "citations": []
        }
        retrieval_data = {
            "fda_interactions": [],
            "disease_contraindications": [{
                "drug": "simvastatin",
                "disease": "liver impairment",
                "severity": "CRITICAL",
                "mechanism": "Hepatic toxicity",
                "effects": "Rhabdomyolysis and hepatic injury"
            }],
            "metabolism_interactions": []
        }

        res = hallucination_guard.run(false_report, retrieval_data)
        out = res.output_data
        self.assertFalse(out["is_safe"])
        self.assertEqual(out["validated_report"]["severity"], "HIGH_RISK")
        self.assertTrue(any("False Reassurance" in c for c in out["detected_unsupported_claims"]))

    # ------------------------------------------------------------------------
    # 11. Token Optimization Context Deduplication
    # ------------------------------------------------------------------------
    def test_optimize_retrieval_context_deduplication(self):
        """Ensure duplicate interaction records are eliminated to save tokens."""
        raw_retrieval = {
            "fda_interactions": [
                {"drug1": "warfarin", "drug2": "aspirin", "severity": "HIGH_RISK", "effects": "bleeding"},
                {"drug1": "aspirin", "drug2": "warfarin", "severity": "HIGH_RISK", "effects": "bleeding"}
            ],
            "disease_contraindications": [
                {"drug": "ibuprofen", "disease": "asthma", "severity": "CRITICAL"},
                {"drug": "ibuprofen", "disease": "asthma", "severity": "CRITICAL"}
            ],
            "metabolism_interactions": []
        }

        cleaned = optimize_retrieval_context(raw_retrieval)
        self.assertEqual(len(cleaned["fda_interactions"]), 1)
        self.assertEqual(len(cleaned["disease_contraindications"]), 1)

    # ------------------------------------------------------------------------
    # 12. Combined Pydantic Schema Validation
    # ------------------------------------------------------------------------
    def test_combined_clinical_assessment_schema(self):
        """Verifies strict Pydantic parsing of CombinedClinicalAssessmentSchema."""
        valid_payload = {
            "severity": "MODERATE_RISK",
            "reasoning": "Clinically significant precaution detected.",
            "item_assessments": [{
                "type": "Drug–Drug Interaction",
                "title": "Aspirin + Lisinopril",
                "severity": "MODERATE_RISK",
                "justification": "Reduced antihypertensive efficacy."
            }],
            "summary": "Moderate risk regimen requiring blood pressure monitoring.",
            "allergies_checked": ["penicillin"],
            "ddi_assessment": "Mild pharmacodynamic antagonism.",
            "drug_condition_assessment": "No absolute disease contraindications.",
            "cyp450_assessment": "No CYP450 conflicts.",
            "clinical_advisory": "Monitor blood pressure regularly.",
            "interactions_details": [{
                "drugs": "Aspirin and Lisinopril",
                "severity": "MODERATE_RISK",
                "mechanism": "Prostaglandin inhibition",
                "clinical_effects": "Decreased ACE-inhibitor efficacy",
                "management": "Monitor BP"
            }],
            "disease_warnings": [],
            "metabolism_interactions": [],
            "recommendations": ["Monitor blood pressure."],
            "suggested_alternatives": [],
            "citations": ["FDA CDER Guidance"]
        }

        model = CombinedClinicalAssessmentSchema(**valid_payload)
        self.assertEqual(model.severity, "MODERATE_RISK")
        self.assertEqual(len(model.item_assessments), 1)
        self.assertEqual(model.item_assessments[0].title, "Aspirin + Lisinopril")

    # ------------------------------------------------------------------------
    # 13. Output Agent Standard Medical Disclaimers
    # ------------------------------------------------------------------------
    def test_output_agent_attaches_standard_disclaimer(self):
        """Verifies that Output Agent standardizes severity and attaches clinical decision disclaimers."""
        guard_data = {
            "validated_report": {
                "severity": "SAFE",
                "summary": "All checks clear.",
                "recommendations": ["Routine administration."]
            },
            "grounding_score": 1.0,
            "is_safe": True
        }
        res = output_agent.run(guard_data)
        report = res.output_data
        self.assertEqual(report["severity"], "SAFE")
        self.assertTrue(any("CLINICAL DECISION-SUPPORT NOTICE" in r for r in report["recommendations"]))



    # ------------------------------------------------------------------------
    # 14. RxNorm Brand-to-Generic Normalization
    # ------------------------------------------------------------------------
    def test_rxnorm_brand_to_generic(self):
        """RxNormClient must normalize brand names to generic canonical names and assign RxCUI."""
        from backend.agents.clinique_rxnorm import RxNormClient
        client = RxNormClient()

        # Eliquis -> apixaban
        rec, unrec = client.normalize_drug("Eliquis")
        self.assertIsNotNone(rec)
        self.assertIsNone(unrec)
        self.assertEqual(rec.preferred_name.lower(), "apixaban")
        self.assertEqual(rec.rxcui, "1364430")

        # Lipitor -> atorvastatin
        rec, unrec = client.normalize_drug("Lipitor")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.preferred_name.lower(), "atorvastatin")
        self.assertEqual(rec.rxcui, "83367")

    # ------------------------------------------------------------------------
    # 15. RxNorm Fuzzy Misspelling Match
    # ------------------------------------------------------------------------
    def test_rxnorm_fuzzy_misspelling(self):
        """RxNormClient approximate matching must correct typos like 'asprin'."""
        from backend.agents.clinique_rxnorm import RxNormClient
        client = RxNormClient()

        rec, unrec = client.normalize_drug("asprin")
        self.assertIsNotNone(rec)
        self.assertIn("aspirin", rec.preferred_name.lower())
        self.assertEqual(rec.rxcui, "7052")

    # ------------------------------------------------------------------------
    # 16. Unrecognized Drug Flags DATA_UNAVAILABLE (Never Safe)
    # ------------------------------------------------------------------------
    def test_unrecognized_drug_flags_data_unavailable(self):
        """Unrecognized drug must NOT default to SAFE; must be flagged as DATA_UNAVAILABLE with REVIEW_REQUIRED."""
        from backend.agents.clinique_rxnorm import RxNormClient
        from backend.agents import input_agent

        # Input Agent separates recognized and unrecognized
        res = input_agent.run("Patient on aspirin and fakedrugXYZ daily")
        out = res.output_data

        self.assertTrue(any(r["preferred_name"] == "aspirin" for r in out["drugs_recognized"]))
        self.assertTrue(any(u["original_mention"] == "fakedrugXYZ" for u in out["drugs_unrecognized"]))

        # Retrieval Agent attaches explicit DATA_UNAVAILABLE record
        ret_res = retrieval_agent.run(
            drugs=out["drugs"],
            conditions=[],
            drugs_recognized=out["drugs_recognized"],
            drugs_unrecognized=out["drugs_unrecognized"]
        )
        ret_out = ret_res.output_data
        self.assertTrue(len(ret_out["unrecognized_assessments"]) > 0)
        self.assertEqual(ret_out["unrecognized_assessments"][0]["status"], "DATA_UNAVAILABLE")

        # Combined assessment enforces REVIEW_REQUIRED floor
        comb_res = combined_assessment_run(
            drugs=out["drugs"],
            conditions=[],
            retrieval_data=ret_out
        )
        comb_out = comb_res.output_data
        self.assertEqual(comb_out["severity"], "REVIEW_REQUIRED")
        self.assertNotEqual(comb_out["severity"], "SAFE")


if __name__ == "__main__":
    unittest.main()

