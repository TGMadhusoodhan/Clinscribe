"""
ClinScribe write-back to OpenMRS.
All writes go to local OpenMRS only — writing to external systems risks data integrity failure.
"""

import logging
from datetime import datetime, timezone

import requests

import config

logger = logging.getLogger(__name__)


# ── Custom exceptions ─────────────────────────────────────────────────────────

class OpenMRSUnavailableError(Exception):
    pass

class AuthenticationError(Exception):
    pass

class PatientNotFoundError(Exception):
    pass

class InvalidFHIRError(Exception):
    pass


# ── Constants ─────────────────────────────────────────────────────────────────

_ACT_CODE_SYSTEM = "http://terminology.hl7.org/CodeSystem/v3-ActCode"
_OPENMRS_ENCOUNTER_TYPE_SYSTEM = "http://fhir.openmrs.org/code-system/encounter-type"
# OpenMRS "Consultation" encounter type UUID — fixed per OpenMRS Reference Application schema
_CONSULTATION_ENCOUNTER_TYPE_UUID = "dd528487-82a5-4082-9c72-ed246bd49591"


def verify_openmrs_connection() -> bool:
    """
    Pings OpenMRS session endpoint to confirm connectivity and auth.
    """
    url = f"{config.OPENMRS_BASE_URL}/ws/rest/v1/session"
    try:
        resp = requests.get(url, auth=(config.OPENMRS_USER, config.OPENMRS_PASSWORD), timeout=5)
    except requests.ConnectionError:
        raise OpenMRSUnavailableError(f"Cannot connect to OpenMRS at {config.OPENMRS_BASE_URL}")
    except requests.Timeout:
        raise OpenMRSUnavailableError(f"Timeout connecting to OpenMRS at {config.OPENMRS_BASE_URL}")

    if resp.status_code == 401:
        raise AuthenticationError(
            f"OpenMRS rejected credentials for user '{config.OPENMRS_USER}'. "
            "Check OPENMRS_USER and OPENMRS_PASSWORD in .env"
        )
    resp.raise_for_status()
    return resp.json().get("authenticated", False)


def _post_fhir(resource_type: str, payload: dict, session: requests.Session | None = None) -> requests.Response:
    url = f"{config.FHIR_BASE_URL}/{resource_type}"
    requester = session or requests
    kwargs = {
        "json": payload,
        "headers": {"Content-Type": "application/json", "Accept": "application/json"},
        "timeout": 15,
    }
    if session is None:
        kwargs["auth"] = (config.OPENMRS_USER, config.OPENMRS_PASSWORD)

    try:
        resp = requester.post(url, **kwargs)
    except requests.ConnectionError:
        raise OpenMRSUnavailableError(f"Cannot connect to OpenMRS at {url}")
    except requests.Timeout:
        raise OpenMRSUnavailableError(f"Timeout posting {resource_type} to OpenMRS")

    if resp.status_code == 401:
        raise AuthenticationError("OpenMRS rejected credentials. Check OPENMRS_USER and OPENMRS_PASSWORD in .env")
    if resp.status_code == 404:
        raise PatientNotFoundError(f"Patient UUID not found in OpenMRS: {payload.get('subject', {}).get('reference', 'unknown')}")
    if resp.status_code in (400, 422):
        raise InvalidFHIRError(f"OpenMRS rejected {resource_type}: {resp.text}")

    resp.raise_for_status()
    return resp


def _simplify_term(term: str) -> list[str]:
    """
    Returns progressively simpler search queries from a clinical term.
    Strips modifiers so 'Suspected arthritis (age-related)' → ['arthritis', 'Suspected arthritis'].
    """
    import re
    # Remove parenthetical qualifiers and common modifiers
    clean = re.sub(r"\(.*?\)", "", term).strip()
    modifiers = {"suspected", "possible", "probable", "chronic", "acute", "early",
                 "late", "severe", "mild", "moderate", "unspecified", "primary",
                 "secondary", "age-related", "type", "stage"}
    words = [w for w in clean.split() if w.lower() not in modifiers]
    simplified = " ".join(words).strip()
    candidates = []
    if simplified and simplified.lower() != term.lower():
        candidates.append(simplified)
    if clean and clean not in candidates:
        candidates.append(clean)
    candidates.append(term)
    return candidates


_CLINICAL_CONCEPT_CLASSES = {"Finding", "Diagnosis", "Symptom", "Symptom/Finding"}


def _lookup_concept(term: str, session: requests.Session | None = None) -> str | None:
    """
    Searches OpenMRS for a clinical concept matching term. Returns UUID or None.
    Filters to Finding/Diagnosis/Symptom classes — the concept search otherwise returns
    Drug concepts (e.g. searching 'arthritis' returns Aspirin, Ibuprofen) when no exact match exists.
    Tries progressively simplified versions of the term.
    """
    url = f"{config.OPENMRS_BASE_URL}/ws/rest/v1/concept"
    requester = session or requests
    base_kwargs: dict = {"timeout": 5}
    if session is None:
        base_kwargs["auth"] = (config.OPENMRS_USER, config.OPENMRS_PASSWORD)

    for candidate in _simplify_term(term):
        try:
            resp = requester.get(url, params={"q": candidate, "limit": 5, "v": "full"}, **base_kwargs)
            results = resp.json().get("results", [])
            for result in results:
                concept_class = result.get("conceptClass", {}).get("display", "")
                if concept_class in _CLINICAL_CONCEPT_CLASSES:
                    logger.info(f"[concept] '{candidate}' → {result['uuid']} ({concept_class})")
                    return result["uuid"]
            if results:
                classes_found = [r.get("conceptClass", {}).get("display", "?") for r in results]
                logger.warning(f"[concept] '{candidate}' found {len(results)} results but none in clinical classes. Got: {classes_found}")
            else:
                logger.warning(f"[concept] '{candidate}' — no results in OpenMRS")
        except Exception as e:
            logger.warning(f"[concept] '{candidate}' — lookup error: {e}")
    logger.warning(f"[concept] '{term}' — no clinical concept found, condition will save without concept (blank in SPA)")
    return None


def _fetch_concept_candidates(term: str, session: requests.Session | None = None) -> list[dict]:
    """
    Searches OpenMRS for clinical concepts matching the term.
    Tries simplified term, then individual words as fallback to maximize candidates for Claude.
    """
    url = f"{config.OPENMRS_BASE_URL}/ws/rest/v1/concept"
    requester = session or requests
    base_kwargs: dict = {"timeout": 5}
    if session is None:
        base_kwargs["auth"] = (config.OPENMRS_USER, config.OPENMRS_PASSWORD)

    # Build search queries: simplified forms + individual words (for multi-word terms)
    queries = list(_simplify_term(term))
    words = [w for w in term.lower().split() if len(w) > 3]
    for w in words:
        if w not in queries:
            queries.append(w)

    candidates = []
    seen = set()
    for query in queries:
        try:
            resp = requester.get(url, params={"q": query, "limit": 10, "v": "full"}, **base_kwargs)
            for result in resp.json().get("results", []):
                uuid = result.get("uuid", "")
                if uuid in seen:
                    continue
                seen.add(uuid)
                name = result.get("name", {}).get("display", "") or result.get("display", "")
                concept_class = result.get("conceptClass", {}).get("display", "")
                candidates.append({"uuid": uuid, "name": name, "conceptClass": concept_class})
        except Exception:
            pass

    # If still no results, do a very broad single-letter search won't help —
    # instead fetch all Finding/Diagnosis concepts (limited) so Claude has something to work with
    if not candidates:
        try:
            resp = requester.get(url, params={"q": "pain", "limit": 25, "v": "full"}, **base_kwargs)
            for result in resp.json().get("results", []):
                uuid = result.get("uuid", "")
                if uuid in seen:
                    continue
                seen.add(uuid)
                name = result.get("name", {}).get("display", "") or result.get("display", "")
                concept_class = result.get("conceptClass", {}).get("display", "")
                candidates.append({"uuid": uuid, "name": name, "conceptClass": concept_class})
        except Exception:
            pass

    logger.warning(f"[concept] '{term}' — found {len(candidates)} candidates for Claude matching")
    return candidates


def _claude_semantic_match(term: str, candidates: list[dict]) -> str | None:
    """
    Asks Claude to pick the closest matching OpenMRS concept for a clinical term.
    Returns the UUID of the best match, or None if nothing is close enough.
    """
    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    options = "\n".join(
        f"- UUID: {c['uuid']} | Name: {c['name']} | Class: {c['conceptClass']}"
        for c in candidates
    )
    prompt = (
        f"A doctor's transcript mentions the diagnosis: \"{term}\"\n\n"
        f"The following concepts exist in the OpenMRS clinical database:\n{options}\n\n"
        f"Which concept is the closest clinical match to \"{term}\"?\n"
        f"Rules:\n"
        f"- Only pick a concept if it is clinically equivalent or a clear synonym (e.g. arthritis ↔ joint pain)\n"
        f"- Do NOT pick a match if the concepts are only loosely related\n"
        f"- Return ONLY the UUID string of the best match, or the word 'none' if nothing fits\n"
        f"No explanation."
    )

    try:
        response = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": prompt}],
        )
        result = response.content[0].text.strip().strip('"').strip("'")
        if result.lower() == "none":
            return None
        # Validate it's actually one of the candidate UUIDs
        valid_uuids = {c["uuid"] for c in candidates}
        if result in valid_uuids:
            matched_name = next(c["name"] for c in candidates if c["uuid"] == result)
            logger.info(f"[concept] Claude matched '{term}' → '{matched_name}' ({result})")
            return result
        return None
    except Exception as e:
        logger.warning(f"[concept] Claude semantic match failed for '{term}': {e}")
        return None


def _create_concept(term: str, session: requests.Session | None = None) -> str | None:
    """
    Creates a new concept in OpenMRS. Tries Diagnosis, then Finding, then Misc class.
    Returns the new concept UUID, or None on failure.
    """
    url = f"{config.OPENMRS_BASE_URL}/ws/rest/v1/concept"
    requester = session or requests
    base_kwargs: dict = {
        "headers": {"Content-Type": "application/json", "Accept": "application/json"},
        "timeout": 10,
    }
    if session is None:
        base_kwargs["auth"] = (config.OPENMRS_USER, config.OPENMRS_PASSWORD)

    # Fetch actual UUIDs for datatype and concept class — OpenMRS requires UUID refs not name strings
    datatype_uuid = _get_concept_ref("conceptdatatype", "N/A", session=session)

    for class_name in ("Diagnosis", "Finding", "Misc"):
        class_uuid = _get_concept_ref("conceptclass", class_name, session=session)
        payload = {
            "names": [{
                "name": term,
                "locale": "en",
                "localePreferred": True,
                "conceptNameType": "FULLY_SPECIFIED",
            }],
            "datatype": datatype_uuid,
            "conceptClass": class_uuid,
        }
        try:
            resp = requester.post(url, json=payload, **base_kwargs)
            if resp.status_code in (200, 201):
                uuid = resp.json().get("uuid")
                logger.warning(f"[concept] Created '{term}' as {class_name} → {uuid}")
                return uuid
            logger.warning(f"[concept] Create failed ({class_name}): {resp.status_code} {resp.text[:300]}")
        except Exception as e:
            logger.warning(f"[concept] Create error ({class_name}): {e}")

    return None


def _get_concept_ref(resource: str, name: str, session: requests.Session | None = None) -> str:
    """Fetches the UUID of a named OpenMRS concept datatype or class."""
    url = f"{config.OPENMRS_BASE_URL}/ws/rest/v1/{resource}"
    requester = session or requests
    kwargs: dict = {"params": {"v": "default"}, "timeout": 5}
    if session is None:
        kwargs["auth"] = (config.OPENMRS_USER, config.OPENMRS_PASSWORD)
    try:
        resp = requester.get(url, **kwargs)
        for item in resp.json().get("results", []):
            if item.get("display", "").lower() == name.lower():
                return item["uuid"]
    except Exception:
        pass
    return name  # fallback to name string if UUID not found


def _get_or_create_concept(term: str, session: requests.Session | None = None) -> str | None:
    """
    1. Try exact/simplified text match in OpenMRS
    2. If no match, fetch candidates and ask Claude for semantic match
    3. If Claude finds nothing close, create a new concept
    """
    # Step 1: exact text match
    uuid = _lookup_concept(term, session=session)
    if uuid:
        return uuid

    # Step 2: Claude semantic match against broader candidate set
    candidates = _fetch_concept_candidates(term, session=session)
    if candidates:
        uuid = _claude_semantic_match(term, candidates)
        if uuid:
            return uuid

    # Step 3: create new concept so it always shows named in SPA
    return _create_concept(term, session=session)


def _post_condition_rest(
    term: str,
    concept_uuid: str | None,
    encounter_uuid: str,
    patient_uuid: str,
    session: requests.Session | None = None,
) -> requests.Response:
    """
    POSTs a condition via OpenMRS REST v1 API (not FHIR).
    The SPA Conditions widget reads from REST v1, not FHIR — FHIR conditions with no
    concept UUID show as blank in the SPA even though they are stored.
    Uses concept UUID when found; falls back to additionalDetail free text.
    """
    url = f"{config.OPENMRS_BASE_URL}/ws/rest/v1/condition"
    payload: dict = {
        "patient": patient_uuid,
        "clinicalStatus": "ACTIVE",
        "onsetDate": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "additionalDetail": term,
    }
    if concept_uuid:
        payload["condition"] = {"coded": concept_uuid}

    logger.warning(f"[condition] POST payload: {payload}")

    requester = session or requests
    kwargs: dict = {
        "json": payload,
        "headers": {"Content-Type": "application/json", "Accept": "application/json"},
        "timeout": 15,
    }
    if session is None:
        kwargs["auth"] = (config.OPENMRS_USER, config.OPENMRS_PASSWORD)

    try:
        resp = requester.post(url, **kwargs)
        logger.warning(f"[condition] Response {resp.status_code}: {resp.text[:400]}")
    except requests.ConnectionError:
        raise OpenMRSUnavailableError(f"Cannot connect to OpenMRS at {url}")
    except requests.Timeout:
        raise OpenMRSUnavailableError("Timeout posting Condition to OpenMRS")

    if resp.status_code == 401:
        raise AuthenticationError("OpenMRS rejected credentials.")
    if resp.status_code == 404:
        raise PatientNotFoundError(f"Patient UUID not found: {patient_uuid}")
    if resp.status_code in (400, 422):
        raise InvalidFHIRError(f"OpenMRS rejected Condition: {resp.text}")

    resp.raise_for_status()
    return resp


def write_encounter_and_conditions(
    patient_uuid: str,
    entities: dict,
    mapped_codes: list,
    session: requests.Session | None = None,
) -> dict:
    """
    Posts one Encounter (via FHIR R4) and one Condition per diagnosis (via REST v1) to OpenMRS.

    # HUMAN APPROVAL GATE — only called after explicit doctor approval in /approve endpoint.
    # Removing this gate means unreviewed AI data enters patient records — patient safety risk.
    """
    if not patient_uuid or not patient_uuid.strip():
        raise ValueError("patient_uuid must be non-empty")

    # Step 1: POST Encounter via FHIR
    encounter_payload = {
        "resourceType": "Encounter",
        "status": "finished",
        "class": {"system": _ACT_CODE_SYSTEM, "code": "AMB", "display": "ambulatory"},
        "type": [{
            "coding": [{
                "system": _OPENMRS_ENCOUNTER_TYPE_SYSTEM,
                "code": _CONSULTATION_ENCOUNTER_TYPE_UUID,
                "display": "Consultation",
            }]
        }],
        "subject": {"reference": f"Patient/{patient_uuid}"},
        "period": {"start": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
    }
    enc_resp = _post_fhir("Encounter", encounter_payload, session=session)

    location = enc_resp.headers.get("Location", "")
    encounter_uuid = location.split("/")[-1]
    if not encounter_uuid:
        raise InvalidFHIRError("OpenMRS did not return a Location header for the Encounter")

    # Step 2: POST each Condition via REST v1 so it appears in the SPA Conditions widget
    conditions_saved = 0
    conditions_failed = []

    mapped_lookup = {m["term"]: m for m in (mapped_codes or [])}

    diagnoses = entities.get("diagnosis", [])
    logger.info(f"[conditions] diagnoses to save: {diagnoses}")
    for diag in diagnoses:
        term = diag if isinstance(diag, str) else diag.get("text", str(diag))
        mapped = mapped_lookup.get(term, {})

        # Prefer English display text from ICD-10/SNOMED for concept lookup and SPA display.
        # Diagnoses extracted from Hindi audio are in Devanagari; OpenMRS has no Hindi concepts.
        english_term = (
            (mapped.get("icd10") or {}).get("description")
            or (mapped.get("snomed") or {}).get("term")
            or term
        )

        try:
            concept_uuid = _get_or_create_concept(english_term, session=session)
            _post_condition_rest(english_term, concept_uuid, encounter_uuid, patient_uuid, session=session)
            conditions_saved += 1
        except (InvalidFHIRError, PatientNotFoundError) as e:
            logger.error(f"Failed to save Condition '{term}': {e}")
            conditions_failed.append({"term": term, "error": str(e)})

    status = "success" if not conditions_failed else "partial"
    return {
        "success": status,
        "encounter_uuid": encounter_uuid,
        "conditions_saved": conditions_saved,
        "conditions_failed": conditions_failed,
    }
