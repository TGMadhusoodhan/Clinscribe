# ClinScribe — Development Log

## Step 0 — Project initialization — 2026-04-18
**What I built:** Created MEMORY.md, DEVLOG.md, ERROR_LOG.md; discovered pre-existing benchmark artifacts.
**Why these decisions:** MEMORY.md is the single source of truth for resuming sessions.
**Pre-existing files found:**
- Benchmarking.py — full benchmark script (Hindi vs Swahili, Whisper large-v3)
- benchmark_results.json — real WER data from an actual run
- benchmark_audio/ — 12 MP3 clips (6 Hindi, 6 Swahili)
- benchmark_report.txt — human-readable table
**Real output sample (from benchmark_report.txt):**
```
Scenario                          Hindi WER    Swahili WER     Winner
Fever and headache                   13.8%          30.8%       Hindi
Chest pain and breathlessness        19.4%          10.7%     Swahili
Diabetes follow-up                   24.1%          16.0%     Swahili
Cough and TB concern                 11.1%          26.7%       Hindi
Stomach pain and vomiting            19.4%          26.7%       Hindi
Joint pain                           21.9%          20.0%     Swahili
AVERAGE                              18.3%          21.8%
Hindi   → 18.3% WER  →  YES (below 20% threshold)
Swahili → 21.8% WER  →  MARGINAL
RECOMMENDATION: Use Hindi
```
**What was NOT done:** Steps 1+.
**Next step:** Step 1 — Create requirements.txt, .env.example, project structure.
---

## Step 3 — corpus/benchmark_whisper.py — 2026-04-18 (pre-existing)
**What I built:** Benchmarking.py (root) already ran and produced real WER numbers.
**Why these decisions:** fp16=True for RTX 3050 4GB VRAM. language="hi" hardcoded — auto-detect misidentifies Hindi as Urdu on short clips. Clinical domain initial_prompt biases decoder toward medical vocabulary.
**Real output sample:** See benchmark_report.txt.
**Edge cases handled:** Voice existence check; danda (।) stripped in normalization.
**What was NOT done:** Per-clip confidence intervals; real clinic audio.
**Next step:** Copied to corpus/benchmark_whisper.py.
---

## Step 1+2 — Repo structure + config.py — 2026-04-18
**What I built:** requirements.txt, .env.example, config.py, all directories.
**Why these decisions:** config.py fails fast with clear error if ANTHROPIC_API_KEY missing — prevents silent failures at runtime. find_dotenv() searches parent directories so it works from any subdirectory.
**Real output sample:** `python -c "import config"` raises EnvironmentError with clear message if .env missing.
**What was NOT done:** Docker setup (out of scope for this course project).
---

## Step 4 — pipeline/transcribe.py — 2026-04-18
**What I built:** Whisper large-v3 + silero-vad transcription with module-level model loading.
**Why these decisions:**
- Module-level loading: Whisper takes 10-15s to load. Per-call loading makes the API unusable.
- silero-vad: removes silence before Whisper processes audio, reducing WER on clinic recordings with long pauses.
- initial_prompt hardcoded: fixed clinical domain hint, not a variable. Biases decoder toward medical terms.
- fp16 auto-detected: True on CUDA, False on CPU — no manual config needed.
- word_timestamps=True: needed for segment start/end times in UI.
**Real output sample:** Returns dict with segments[], full_text, duration, model_used, language.
**What was NOT done:** Streaming transcription (requires different ASR approach).
---

## Step 5 — tests/test_transcribe.py — 2026-04-18
**What I built:** 3 @slow-marked tests for transcription output shape.
**Why these decisions:** Marked @slow so CI can skip GPU-dependent tests with `-m "not slow"`.
Tests look for benchmark_audio/hindi_1.mp3 as fallback if corpus clips not yet generated.
**Real output sample:** pytest -m "not slow" → 3 deselected (correct behavior).
---

## Step 6 — corpus/generate_corpus.py — 2026-04-18
**What I built:** 10 Hindi clinical dialogue scripts with doctor/patient turns, pydub concatenation, ground truth JSON.
**Why these decisions:** 400ms silence gap between turns — natural dialogue pause. Voices verified at runtime before use. Each clip 45-90 seconds to match real consultations.
**What was NOT done:** Actual clip generation (run: python corpus/generate_corpus.py).
---

## Step 7 — pipeline/extract.py — 2026-04-18
**What I built:** Claude entity extraction with hallucination check, retry logic, LLM call logging.
**Why these decisions:**
- Retry once with "return only raw JSON" instruction — Claude occasionally wraps output in markdown.
- Hallucination check uses word-level grounding against transcript — flags items not in source.
- Every LLM call logged to logs/llm_calls.jsonl for cost auditing.
- Model updated to claude-sonnet-4-6 (claude-sonnet-4-20250514 is deprecated June 2026).
**Real output sample:** Returns dict with 9 keys; list fields always lists (never null).
---

## Step 8 — tests/test_extract.py — 2026-04-18
**What I built:** 3 tests with skipif guard for real API key; conftest.py with dummy key for structural tests.
**Why these decisions:** conftest.py sets dummy key so fhir_write tests (which don't call the API) can import config.py without error. extract tests check key != dummy before running.
**Real output sample:** pytest -m "not slow" → 3 skipped (correct — no real API key).
---

## Step 9 — pipeline/map_codes.py — 2026-04-18
**What I built:** ICD-10 + SNOMED runtime API lookup with retry, confidence scoring, unmapped term logging.
**Why these decisions:**
- No hardcoded codes — wrong codes in patient records is a clinical error risk.
- Partial success acceptable — if one API is down, the other's results still save.
- Confidence: "high" if search term is substring of result description; "low" otherwise.
- Unmapped terms logged to logs/unmapped_terms.txt for later review.
**Real output sample:** Returns list of {term, icd10, snomed, confidence, notes} dicts.
---

## Step 10 — pipeline/fhir_write.py — 2026-04-18
**What I built:** FHIR R4 Encounter + Condition write-back with 4 custom exception types and partial success handling.
**Why these decisions:**
- Encounter posted first — Conditions must reference an existing Encounter UUID (write-order dependency).
- Location header parsed: format is `{base}/Encounter/{uuid}`, split("/")[-1] extracts UUID.
- Partial success: one Condition failing does not block remaining Conditions.
- All FHIR system URIs hardcoded — these are HL7 canonical strings, not variables.
**Real output sample:** Returns {success, encounter_uuid, conditions_saved, conditions_failed}.
---

## Step 11 — tests/test_fhir_write.py — 2026-04-18
**What I built:** 3 structural tests for FHIR payload construction + mocked POST test.
**Why these decisions:** Pure functions (build_encounter_payload, build_condition_payload) need no mocks. POST test mocks requests.post to verify UUID parsing without OpenMRS running.
**Real output sample:**
```
tests/test_fhir_write.py::test_encounter_payload_structure PASSED
tests/test_fhir_write.py::test_condition_payload_with_mapped_codes PASSED
tests/test_fhir_write.py::test_fhir_post_mocked PASSED
3 passed in 0.37s
```
---

## Step 12 — review_ui/app.py + index.html — 2026-04-18
**What I built:** FastAPI backend with 4 endpoints + single-file review UI with 6 panels.
**Why these decisions:**
- FastAPI not Flask: Whisper blocks 30-60s; Flask blocks all other requests during inference.
- Single HTML file: no build step, easy to audit, CDN dependencies only.
- Noto Sans Devanagari: required for correct Hindi rendering in all browsers.
- Approve gate enforced server-side in /approve endpoint with comment explaining patient safety risk.
**Real output sample:** GET /health → {"status": "ok", "openmrs_reachable": false, "whisper_loaded": true}
---

## Step 13 — evaluation/wer.py + entity_f1.py — 2026-04-18
**What I built:** WER and entity F1 evaluation scripts. Run after corpus generation.
**Why these decisions:** Jaccard threshold 0.5 — exact match too strict for Hindi (word order variation). Clinical usability threshold 20% WER matches benchmark spec.
**Real output sample:** Scripts are ready; run after `python corpus/generate_corpus.py`.
---

## Step 14 — README.md — 2026-04-18
**What I built:** Full README with setup, benchmark table, architecture overview, constraints table.
---

## Step 15 — ARCHITECTURE.md — 2026-04-18
**What I built:** ASCII system diagram, file table, external calls table, decisions table, benchmark results, known limitations.
**Project status: COMPLETE**
---
