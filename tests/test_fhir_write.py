"""
Tests for pipeline/fhir_write.py.

What is tested:
- Encounter payload structure matches FHIR R4 spec
- Condition payload structure with mapped codes (from map_codes() output format)
- UUID parsing from a mocked Location header response

What is NOT tested:
- Real OpenMRS connectivity (requires running OpenMRS instance; tested manually with verify_openmrs_connection())
- Authentication error paths (requires wrong credentials against real server)
- Partial success path (requires live server returning 422 on some Conditions)

Mock strategy:
- build_encounter_payload() and build_condition_payload() are pure functions — no mocks needed.
- test_fhir_post_mocked mocks requests.post to avoid real network calls.
  The mock simulates a 201 Created response with Location header.
"""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

_TEST_PATIENT_UUID = "test-patient-uuid-1234"
_TEST_ENCOUNTER_UUID = "test-encounter-uuid-5678"


def test_encounter_payload_structure():
    """
    What it does: Verifies build_encounter_payload() returns a valid FHIR Encounter shape.
    Inputs: test patient UUID
    Outputs: asserts resourceType, status, subject reference format
    """
    from pipeline.fhir_write import build_encounter_payload
    payload = build_encounter_payload(_TEST_PATIENT_UUID)

    assert payload["resourceType"] == "Encounter"
    assert payload["status"] == "finished"
    assert payload["subject"]["reference"] == f"Patient/{_TEST_PATIENT_UUID}"
    assert "class" in payload
    assert payload["class"]["code"] == "AMB"
    # HL7 ActCode system URI must be exact — FHIR servers reject any other string
    assert payload["class"]["system"] == "http://terminology.hl7.org/CodeSystem/v3-ActCode"
    assert "period" in payload
    assert "start" in payload["period"]


def test_condition_payload_with_mapped_codes():
    """
    What it does: Verifies build_condition_payload() includes both SNOMED and ICD-10 coding entries.
    Inputs: a mapped dict in map_codes() output format (not hardcoded codes — structure test only)
    Outputs: asserts resourceType, coding array structure, both system URIs present
    Why: FHIR Condition must reference both terminologies for interoperability.
         Values come from map_codes() output — we only test structure here, not specific codes.
    """
    from pipeline.fhir_write import build_condition_payload

    # Codes come from map_codes() output format — these are placeholder values for structure testing only
    # VIOLATION: do not use these codes in production; real codes must come from map_codes() at runtime
    mapped = {
        "term": "fever",
        "icd10": {"code": "R50.9", "description": "Fever, unspecified"},
        "snomed": {"conceptId": "386661006", "term": "Fever"},
        "confidence": "high",
        "notes": "test",
    }

    payload = build_condition_payload("fever", mapped, _TEST_ENCOUNTER_UUID, _TEST_PATIENT_UUID)

    assert payload["resourceType"] == "Condition"
    assert payload["subject"]["reference"] == f"Patient/{_TEST_PATIENT_UUID}"
    assert payload["encounter"]["reference"] == f"Encounter/{_TEST_ENCOUNTER_UUID}"
    assert payload["code"]["text"] == "fever"

    coding = payload["code"]["coding"]
    assert isinstance(coding, list)
    assert len(coding) == 2

    systems = {c["system"] for c in coding}
    assert "http://snomed.info/sct" in systems
    assert "http://hl7.org/fhir/sid/icd-10" in systems


def test_fhir_post_mocked():
    """
    What it does: Mocks requests.post to verify UUID is parsed correctly from Location header.
    Inputs: mock 201 response with Location header ending in encounter UUID
    Outputs: asserts returned encounter_uuid matches the UUID in the Location header
    """
    from pipeline.fhir_write import write_encounter_and_conditions

    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.headers = {
        "Location": f"http://localhost:8080/openmrs/ws/fhir2/R4/Encounter/{_TEST_ENCOUNTER_UUID}"
    }
    mock_response.raise_for_status = MagicMock()

    entities = {
        "diagnosis": ["fever"],
        "symptoms": ["high temperature"],
    }
    mapped_codes = [
        {
            "term": "fever",
            "icd10": {"code": "R50.9", "description": "Fever, unspecified"},
            "snomed": {"conceptId": "386661006", "term": "Fever"},
            "confidence": "high",
            "notes": "",
        }
    ]

    with patch("pipeline.fhir_write.requests.post", return_value=mock_response):
        result = write_encounter_and_conditions(_TEST_PATIENT_UUID, entities, mapped_codes)

    assert result["encounter_uuid"] == _TEST_ENCOUNTER_UUID
    assert result["success"] in ("success", "partial")
