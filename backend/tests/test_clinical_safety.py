import unittest
from backend.agents.severity_agent import evaluate_deterministic_severity, normalize_severity, RISK_HIERARCHY
from backend.agents import retrieval_agent
from backend.schemas import ClinicalReport

class TestClinicalSafetyArchitecture(unittest.TestCase):

    def test_never_safe_with_disease_and_single_drug(self):
        """
        CRITICAL SAFETY TEST:
        Input: Drug: Simvastatin, Condition: Liver disease, Allergy: None.
        Must NOT be SAFE merely because no drug-drug interaction was detected.
        """
        drugs = ["simvastatin"]
        conditions = ["liver impairment"]
        allergies = None

        retrieval_res = retrieval_agent.run(drugs, conditions, allergies=allergies)
        retrieval_data = retrieval_res.output_data

        sev, reason, items = evaluate_deterministic_severity(
            retrieval_data, drugs=drugs, conditions=conditions, allergies=allergies
        )

        # Must be HIGH_RISK or MODERATE_RISK (contraindicated in liver disease)
        self.assertIn(sev, ["HIGH_RISK", "MODERATE_RISK", "REVIEW_REQUIRED"])
        self.assertNotEqual(sev, "SAFE")
        self.assertTrue(any("liver" in str(item).lower() or "simvastatin" in str(item).lower() for item in items) or sev == "REVIEW_REQUIRED")

    def test_never_safe_with_unregistered_condition(self):
        """
        If a patient has a condition (e.g. hypertension) but no direct DDI exists,
        the presence of an unverified condition must trigger REVIEW_REQUIRED rather than false SAFE.
        """
        drugs = ["paracetamol"]
        conditions = ["uncommon chronic disorder"]
        allergies = None

        retrieval_data = {
            "fda_interactions": [],
            "disease_contraindications": [],
            "metabolism_interactions": []
        }

        sev, reason, items = evaluate_deterministic_severity(
            retrieval_data, drugs=drugs, conditions=conditions, allergies=allergies
        )

        self.assertNotEqual(sev, "SAFE")
        self.assertEqual(sev, "REVIEW_REQUIRED")
        self.assertIn("absence of a detected database match is not proof of safety", reason.lower())

    def test_allergy_triggers_high_risk(self):
        """Drug matching patient allergy must be HIGH_RISK."""
        drugs = ["amoxicillin"]
        conditions = []
        allergies = "amoxicillin, penicillin"

        retrieval_res = retrieval_agent.run(drugs, conditions, allergies=allergies)
        retrieval_data = retrieval_res.output_data

        sev, reason, items = evaluate_deterministic_severity(
            retrieval_data, drugs=drugs, conditions=conditions, allergies=allergies
        )

        self.assertEqual(sev, "HIGH_RISK")
        self.assertTrue(any("allergy" in str(item).lower() for item in items))

    def test_renal_impairment_with_ibuprofen_single_drug(self):
        """Single NSAID with renal impairment must be HIGH_RISK even without a 2nd drug."""
        drugs = ["ibuprofen"]
        conditions = ["renal impairment"]
        allergies = None

        retrieval_res = retrieval_agent.run(drugs, conditions, allergies=allergies)
        retrieval_data = retrieval_res.output_data

        sev, reason, items = evaluate_deterministic_severity(
            retrieval_data, drugs=drugs, conditions=conditions, allergies=allergies
        )

        self.assertEqual(sev, "HIGH_RISK")
        self.assertIn("renal impairment", str(items).lower())

    def test_safe_only_when_all_dimensions_clear(self):
        """SAFE is returned ONLY when no DDIs, no conditions, no allergies, and no CYP450 conflicts exist."""
        drugs = ["acetaminophen"]
        conditions = []
        allergies = None

        retrieval_data = {
            "fda_interactions": [],
            "disease_contraindications": [],
            "metabolism_interactions": []
        }

        sev, reason, items = evaluate_deterministic_severity(
            retrieval_data, drugs=drugs, conditions=conditions, allergies=allergies
        )

        self.assertEqual(sev, "SAFE")
        self.assertIn("no clinically significant", reason.lower())

    def test_risk_hierarchy_precedence(self):
        """Verify RISK_HIERARCHY orders HIGH_RISK > MODERATE_RISK > REVIEW_REQUIRED > LOW_RISK > SAFE."""
        self.assertGreater(RISK_HIERARCHY["HIGH_RISK"], RISK_HIERARCHY["MODERATE_RISK"])
        self.assertGreater(RISK_HIERARCHY["MODERATE_RISK"], RISK_HIERARCHY["REVIEW_REQUIRED"])
        self.assertGreater(RISK_HIERARCHY["REVIEW_REQUIRED"], RISK_HIERARCHY["LOW_RISK"])
        self.assertGreater(RISK_HIERARCHY["LOW_RISK"], RISK_HIERARCHY["SAFE"])

    def test_clinical_report_schema_separated_fields(self):
        """Ensure ClinicalReport model includes all separated assessment dimensions."""
        report = ClinicalReport(
            severity="MODERATE_RISK",
            summary="Test summary",
            allergies_checked=["aspirin"],
            ddi_assessment="No interacting partner",
            drug_condition_assessment="Hepatic precautions apply",
            cyp450_assessment="Routine pathways",
            clinical_advisory="Follow up with hepatologist",
            recommendations=["Monitor ALT/AST"]
        )
        self.assertEqual(report.severity, "MODERATE_RISK")
        self.assertEqual(report.ddi_assessment, "No interacting partner")
        self.assertEqual(report.drug_condition_assessment, "Hepatic precautions apply")
        self.assertEqual(report.cyp450_assessment, "Routine pathways")
        self.assertEqual(report.clinical_advisory, "Follow up with hepatologist")


if __name__ == "__main__":
    unittest.main()
