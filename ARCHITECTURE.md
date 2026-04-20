# ClinScribe — Architecture

## 1. System diagram

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                  Doctor's Machine                       │
                    │                                                         │
  [Doctor speaks] ──► MP3/WAV ──► [review_ui/app.py: POST /transcribe]       │
  (Hindi, clinic)                           │                                 │
                                            ▼  bytes (audio)                 │
                                  [pipeline/transcribe.py]                   │
                                  silero-vad: removes silence                │
                                  Whisper large-v3: local inference          │
                                  language="hi" (ISO 639-1, hardcoded)       │
                                            │                                 │
                                            ▼  {segments[], full_text: str}  │
                                  [pipeline/extract.py]                      │
                                  Claude API: entity extraction              │
                                            │                                 │
                                            ▼  {chief_complaint, symptoms,   │
                                            │   diagnosis, medications, ...}  │
                                  [pipeline/map_codes.py]                    │
                                  ICD-10 API: icd10api.com                  │
                                  SNOMED API: browser.ihtsdotools.org        │
                                            │                                 │
                                            ▼  [{term, icd10, snomed,        │
                                            │   confidence}]                  │
                                            │                                 │
                    │           [Browser — review_ui/static/index.html]      │
                    │           Doctor reads, edits, and clicks Approve       │
                    │                       │                                 │
                    │                       ▼  POST /approve (explicit gate)  │
                    │           [pipeline/fhir_write.py]                     │
                    │           POST Encounter → get UUID                    │
                    │           POST Condition(s) → reference Encounter       │
                    │                       │                                 │
                    └───────────────────────┼─────────────────────────────────┘
                                            │  FHIR R4 JSON
                                            ▼
                              [OpenMRS FHIR2 endpoint]
                              localhost:8080/openmrs/ws/fhir2/R4
```

---

## 2. File table

| File | Purpose | Key functions | Imports from |
|---|---|---|---|
| `config.py` | Loads all env vars; fails fast if required vars missing | — | `dotenv` |
| `pipeline/transcribe.py` | Whisper inference + silero-vad silence removal | `transcribe()`, `translate_segments()` | `config`, `whisper`, `torch`, `anthropic` |
| `pipeline/extract.py` | Claude entity extraction from Hindi transcript | `extract()`, `_check_hallucinations()` | `config`, `anthropic` |
| `pipeline/map_codes.py` | ICD-10 + SNOMED-CT code lookup | `map_codes()`, `_fetch_icd10()`, `_fetch_snomed()` | `requests` |
| `pipeline/fhir_write.py` | FHIR R4 resource construction + OpenMRS write | `write_encounter_and_conditions()`, `verify_openmrs_connection()`, `build_encounter_payload()`, `build_condition_payload()` | `config`, `requests` |
| `review_ui/app.py` | FastAPI backend; 4 endpoints | `health()`, `search_patients()`, `transcribe_audio()`, `approve_and_write()` | all pipeline modules, `config`, `fastapi` |
| `review_ui/static/index.html` | Single-file review UI (6 panels, no build step) | — | CDN: Pico CSS, Google Fonts |
| `corpus/generate_corpus.py` | 10 Hindi clinical dialogue clips via Edge TTS | `generate_all()`, `_generate_clip()` | `edge_tts`, `pydub` |
| `corpus/benchmark_whisper.py` | WER benchmark: Hindi vs Swahili | `run_benchmark()`, `transcribe_clip()`, `normalize()` | `whisper`, `jiwer`, `edge_tts` |
| `evaluation/wer.py` | WER evaluation on 10 corpus clips | `run_wer_evaluation()`, `normalize()` | `pipeline.transcribe`, `jiwer` |
| `evaluation/entity_f1.py` | Entity extraction F1 on 10 corpus clips | `run_entity_f1_evaluation()`, `jaccard_match()`, `precision_recall_f1()` | `pipeline.transcribe`, `pipeline.extract` |
| `tests/test_transcribe.py` | Transcription output shape tests | 3 test functions | `pipeline.transcribe` |
| `tests/test_extract.py` | Extraction schema and null handling tests | 3 test functions | `pipeline.extract` |
| `tests/test_fhir_write.py` | FHIR payload structure tests; mocked POST | 3 test functions | `pipeline.fhir_write`, `unittest.mock` |

---

## 3. External calls table

| Service | Endpoint URL | Called from | Auth method | What it returns |
|---|---|---|---|---|
| Anthropic API | `https://api.anthropic.com/v1/messages` | `pipeline/extract.py`, `pipeline/transcribe.py` | Bearer token (`ANTHROPIC_API_KEY`) | JSON with extracted entities or translated segments |
| ICD-10 API | `https://icd10api.com/?s={term}&desc=short&r=json` | `pipeline/map_codes.py` | None (public) | JSON with `Response`, `Notes[].ICD10Code`, `Notes[].Description` |
| SNOMED CT NLM Browser | `https://browser.ihtsdotools.org/snowstorm/snomed-ct/browser/MAIN/descriptions?term={term}&active=true&limit=3` | `pipeline/map_codes.py` | None (public) | JSON with `items[].concept.conceptId`, `items[].term` |
| OpenMRS FHIR2 R4 | `{OPENMRS_BASE_URL}/ws/fhir2/R4/` | `pipeline/fhir_write.py`, `review_ui/app.py` | HTTP Basic (`OPENMRS_USER:OPENMRS_PASSWORD`) | FHIR Bundle, Patient, Encounter, Condition resources |
| Edge TTS (Microsoft) | Internal SDK — `edge_tts.Communicate()` | `corpus/generate_corpus.py`, `corpus/benchmark_whisper.py` | None (public) | MP3 audio stream |

---

## 4. Decisions table

| Decision | Alternatives considered | Why this choice | Recorded in DEVLOG step |
|---|---|---|---|
| Language: Hindi | Swahili (sw-KE) | Hindi avg WER 18.3% < 20% threshold; Swahili 21.8% marginal. Measured on Whisper large-v3 with 6 matched TTS clips. | Step 3 |
| Whisper model: large-v3 | medium, large-v2 | large-v3 has best Hindi accuracy. fp16=True fits in 4GB VRAM on RTX 3050. | Step 3 |
| Framework: FastAPI | Flask, Django | Whisper inference blocks for 30-60s. Flask/Django are synchronous — all other requests block during inference. FastAPI runs on async event loop with concurrent request handling. | Step 12 |
| Silence removal: silero-vad | None (raw audio) | Clinic recordings have long pauses between speaker turns. VAD removes silence before Whisper processes, reducing WER on real audio. | Step 4 |
| Code mapping: runtime API lookup | Pre-built local SQLite/CSV | Hardcoded codes in patient records is a clinical error risk. APIs guarantee current codes. Partial results (one system failing) acceptable; fail-silent per term. | Step 9 |
| Entity extraction: Claude API | NER models, spaCy | Hindi clinical NER models are not publicly available. Claude handles code-switching (Hindi+English) natively and returns structured JSON directly. | Step 7 |
| Evaluation metric: Jaccard token overlap ≥ 0.5 | Exact string match, ROUGE | Exact match too strict for Hindi (different orderings of same words). Jaccard at 0.5 threshold balances precision and recall for clinical entity lists. | Step 13 |
| Frontend: single HTML file | React, Vue | No build step required. One file is simpler to audit, deploy, and review. CDN dependencies (Pico CSS + Google Fonts) are the only external dependencies. | Step 12 |

---

## 5. Evaluation results

Results below are populated after running `evaluation/wer.py` and `evaluation/entity_f1.py` on the generated corpus.

### WER results (`evaluation/results/wer_results.json`)

Run `python evaluation/wer.py` after generating corpus clips to populate this section with real numbers.

Benchmark WER (from `corpus/benchmark_results.json`, TTS audio, Whisper large-v3):

| Clip | Scenario | Hindi WER |
|---|---|---|
| 01 | Fever and headache | 13.8% |
| 02 | Chest pain and breathlessness | 19.4% |
| 03 | Diabetes follow-up | 24.1% |
| 04 | Cough and TB concern | 11.1% |
| 05 | Stomach pain and vomiting | 19.4% |
| 06 | Joint pain | 21.9% |
| **Avg** | | **18.3%** |

Clinical usability threshold: WER < 20% → **PASS**

### Entity F1 results (`evaluation/results/entity_f1.json`)

Run `python evaluation/entity_f1.py` after generating corpus clips to populate with real numbers.

---

## 6. Known limitations

1. **TTS-only corpus** — All benchmark and evaluation audio is synthetic (Edge TTS). Real clinic audio will have higher WER (estimated 5-15% higher) due to background noise, regional accents, and natural disfluencies.

2. **Code-switching normalization** — Whisper large-v3 with `language="hi"` generally preserves English medical terms (blood pressure, sugar) in their English form. Some terms may be transcribed in Devanagari, making code mapping harder.

3. **SNOMED API availability** — The public SNOMED browser at `browser.ihtsdotools.org` has no SLA. If it is down, SNOMED codes are omitted from Conditions (text-only fallback). This does not break the FHIR write.

4. **ICD-10 API coverage** — `icd10api.com` covers ICD-10-CM codes. Hindi medical terms need to be in English for lookup. Claude extraction produces English terms but code-switched input may reduce accuracy.

5. **No streaming** — Whisper processes the entire audio file before returning. For long consultations (>10 minutes), UI shows a spinner for the full duration. Real-time streaming would require a different ASR service.

6. **Single-user server** — The current setup loads one Whisper model. Concurrent transcription requests will queue. Production deployment would need a job queue (e.g., Celery + Redis).

7. **OpenMRS patient UUID required** — The doctor must search and select a patient before approving. If the patient is not registered in OpenMRS, the encounter cannot be saved.
