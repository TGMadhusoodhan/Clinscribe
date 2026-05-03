# ClinScribe

Hindi AI clinical scribe for OpenMRS. A doctor speaks in Hindi; ClinScribe transcribes, translates to English, extracts structured clinical entities, maps SNOMED-CT and ICD-10 codes, and writes FHIR R4 records to OpenMRS — after explicit doctor review and approval.

**Course:** CS 595 Medical Informatics AI — Illinois Institute of Technology  
**Language:** Hindi (hi-IN) — chosen after benchmarking Whisper large-v3 on both Hindi and Swahili  
**WER (Hindi, synthetic corpus):** 18.3% average — below the 20% clinical usability threshold

---

## What it does

1. Doctor uploads a Hindi audio recording (MP3/WAV) of a patient consultation
2. ClinScribe transcribes the audio locally using faster-whisper large-v3 (INT8, CPU)
3. The transcript is translated to English
4. Claude (or a local Qwen model in offline mode) extracts structured clinical entities: chief complaint, symptoms, diagnoses, medications, treatment plan, red flags
5. Diagnoses are mapped to ICD-10 and SNOMED-CT codes via public APIs
6. The doctor reviews and edits all extracted data in a web UI, including vitals and clinical notes
7. The doctor explicitly checks a confirmation checkbox and clicks **Approve** — only then is data written to OpenMRS
8. An FHIR R4 Encounter, Condition records, vitals (Observations), and medications are posted to the local OpenMRS instance

---

## Pipeline

```
Audio (Hindi MP3/WAV)
      ↓
[silero-vad]              removes silence — reduces WER on clinic recordings
      ↓
[faster-whisper large-v3] local inference, INT8 on CPU — audio never leaves the machine (HIPAA)
      ↓  Hindi transcript with word-level timestamps
[Translation]             Online: Claude API  |  Offline: Ollama/qwen2.5:7b
      ↓  English transcript (translated per-segment — no truncation on long consultations)
[Entity Extraction]       Online: Claude API  |  Offline: qwen2.5:7b via Ollama (single-pass)
      ↓  {chief_complaint, symptoms, diagnosis, medications, treatment_plan, red_flags}
[ICD-10 + SNOMED APIs]    runtime lookup — no hardcoded codes
      ↓  [{term, icd10, snomed, confidence}]
[Doctor Review UI]        editable, with hallucination flags, manual diagnosis input, vitals, notes
      ↓  after explicit checkbox confirmation + doctor approval only
[OpenMRS FHIR R4]         POST Encounter → POST Condition(s) + Vitals + Medications via REST v1
```

---

## Two modes

| | Online | Offline |
|---|---|---|
| **Translation** | Claude API | Ollama/qwen2.5:7b |
| **Entity extraction** | Claude API | qwen2.5:7b via Ollama (single-pass) |
| **Internet required** | Yes (Anthropic API) | No (fully local) |
| **Setup** | Just an API key | Ollama + model pull |
| **Use case** | Default / clinic with internet | Rural deployment, air-gapped |

Toggle between modes using the switch in the top-right of the UI.

---

## Project structure

```
clinscribe/
├── pipeline/
│   ├── transcribe.py              faster-whisper large-v3 (INT8, CPU) + silero-vad; online Claude translation
│   ├── translate_offline.py       Offline translation via Ollama/qwen2.5:7b with Hindi medical glossary
│   ├── extract.py                 Online entity extraction via Claude API (with prompt caching)
│   ├── extract_offline.py         Offline single-pass extraction via Ollama (qwen2.5:7b, ~3s)
│   ├── map_codes.py               ICD-10 (local simple-icd-10-cm + rapidfuzz) + SNOMED-CT (API) lookup
│   └── fhir_write.py              OpenMRS FHIR R4 Encounter + REST v1 Conditions + Vitals + Medications
├── review_ui/
│   ├── app.py                     FastAPI backend — /transcribe, /approve, /patients, /health
│   └── static/index.html          Single-file review UI (no build step)
├── data/
│   └── hindi_medical_glossary.csv 80+ Hindi→English medical term pairs (Devanagari-transliterated English)
├── corpus/
│   ├── generate_corpus.py         Generates 10 Hindi clinical dialogue clips via Edge TTS
│   └── benchmark_whisper.py       Hindi vs Swahili WER benchmark (corpus clips)
├── evaluation/
│   ├── wer.py                     WER evaluation on corpus clips
│   └── entity_f1.py               Entity extraction F1 evaluation
├── tests/
│   ├── conftest.py
│   ├── test_transcribe.py
│   ├── test_extract.py
│   └── test_fhir_write.py
├── logs/
│   ├── llm_calls.jsonl            Every LLM API call logged (phase, tokens, latency, success)
│   └── unmapped_terms.txt         Terms with no ICD-10 or SNOMED match
├── benchmark_audio/               12 MP3 clips — 6 Hindi, 6 Swahili
├── benchmark_modes.py             Online vs offline entity completeness benchmark
├── benchmark_whisper_compare.py   faster-whisper vs openai-whisper speed/quality benchmark
├── benchmark_translation_compare.py  MarianMT vs Ollama/Qwen vs Claude translation quality benchmark
├── Modelfile                      Custom Ollama model definition (qwen2.5:7b + context/GPU config)
├── ARCHITECTURE.md                System architecture notes
├── DEVLOG.md                      Development log
├── config.py                      Loads all env vars from .env; single import for all modules
├── docker-compose.yml             OpenMRS + MySQL
├── enable_basic_auth.sql          One-time OpenMRS auth fix for REST API access
├── requirements.txt
├── .env.example
└── ERROR_LOG.md                   All runtime errors appended here automatically
```

---

## Setup

### Prerequisites

- Python 3.10+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for OpenMRS)
- [Ollama](https://ollama.com/download) (for offline mode — handles both translation AND extraction)
- FFmpeg — `winget install ffmpeg` on Windows, `brew install ffmpeg` on macOS
- GPU recommended (CUDA 11.8+) — CPU works but transcription is slower
- Microsoft C++ Build Tools (Windows only, required for some packages)

---

### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd clinscribe
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For PyTorch with CUDA 11.8 (GPU):
```bash
pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

For CPU-only:
```bash
pip install torch torchaudio
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required for online mode
ANTHROPIC_API_KEY=sk-ant-...

# OpenMRS REST API (backend, port 8080 via Docker)
OPENMRS_BASE_URL=http://localhost:8080/openmrs
OPENMRS_USER=admin
OPENMRS_PASSWORD=Admin123

# OpenMRS SPA URL (web UI, port 80 via Docker gateway)
OPENMRS_SPA_URL=http://localhost/openmrs/spa

# Whisper
WHISPER_MODEL=large-v3
WHISPER_LANGUAGE=hi

# Claude model
ANTHROPIC_MODEL=claude-sonnet-4-6

# Offline mode — Ollama server (used for both translation AND extraction)
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen2.5:7b
```

> **Note:** `OPENMRS_BASE_URL` points to the backend container (port 8080). `OPENMRS_SPA_URL` points to the gateway (port 80). Both are set automatically by docker-compose; you only need to change them if you run OpenMRS outside Docker.

### 4. Start OpenMRS

```bash
docker compose up -d
```

Wait ~2 minutes for startup. Visit `http://localhost/openmrs/spa` to confirm it is running.

Enable Basic Auth for REST API access (one-time, first run only):
```bash
docker compose exec db mysql -u openmrs -popenmrs openmrs < enable_basic_auth.sql
```

### 5. Run ClinScribe

```bash
uvicorn review_ui.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`

> **Note:** Do not use `--reload` — watchfiles monitors the entire directory including `.venv`, causing constant restarts.

---

## Offline mode setup (optional)

Offline mode uses Ollama for **both translation and entity extraction** — one model (qwen2.5:7b) handles the entire AI pipeline with no internet access.

### Ollama setup

> **Why Ollama, not vLLM?** vLLM requires `uvloop` which is Linux-only and will not start on Windows. Ollama exposes the same OpenAI-compatible API and works on all platforms.

1. Install Ollama from [ollama.com/download](https://ollama.com/download)

2. Pull the model (~4.5GB):
   ```bash
   ollama pull qwen2.5:7b
   ```

3. Set these in `.env`:
   ```env
   OLLAMA_BASE_URL=http://localhost:11434/v1
   OLLAMA_MODEL=qwen2.5:7b
   ```

4. Start Ollama before running ClinScribe:
   ```bash
   ollama serve
   ```

Then start ClinScribe normally and flip the **Offline** toggle in the UI header.

### Hindi medical glossary

Offline translation pre-processes Devanagari-transliterated English before sending text to Ollama. Common medical terms written in Hindi script that a 7B model won't recognise from context (e.g. "ओरस" → "ORS", "फूट पॉइस्निंग" → "food poisoning") are substituted from `data/hindi_medical_glossary.csv` before translation. The glossary is a plain CSV — add rows to extend it without touching code.

### Custom Ollama model (optional)

The `Modelfile` in the project root sets GPU layer count and context window for ClinScribe:

```bash
ollama create clinscribe-llm -f Modelfile
```

Then set `OLLAMA_MODEL=clinscribe-llm` in `.env`. The base model is still `qwen2.5:7b` — this only tunes inference parameters.

### Pinning Ollama to prevent auto-updates

Ollama auto-updates silently on Windows and macOS. An update can change API behavior and break the offline pipeline without warning. Disable it immediately after install:

**Windows** — open `%APPDATA%\Ollama\ollama.json` (create if missing) and add:
```json
{
  "noAutoUpdate": true
}
```

Or via the system tray: right-click the Ollama icon → **Settings** → uncheck **Automatically update Ollama**.

**macOS** — open `~/.ollama/ollama.json` and add the same `"noAutoUpdate": true` line.

To pin the exact model weights you pulled, record the current digest:
```bash
ollama show qwen2.5:7b --modelinfo
```
Save the `sha256:...` digest. To restore the exact same model later:
```bash
ollama pull qwen2.5:7b@sha256:<your-digest-here>
```

### Verifying Ollama is using your GPU

Ollama automatically uses the GPU if CUDA drivers are installed. To confirm:

```bash
ollama ps
```

Look for `100% GPU` in the `PROCESSOR` column. If it shows `100% CPU`, check that your CUDA drivers are installed.

```bash
nvidia-smi
```

`qwen2.5:7b` uses approximately 4.4–4.7GB of VRAM. If your GPU has less than 5GB free, Ollama will fall back to CPU automatically.

> **Note on VRAM allocation:** Whisper runs on CPU (INT8 via faster-whisper) so the full GPU is available for Ollama. On a 6GB GPU, Ollama gets the entire VRAM budget. Whisper and Ollama are sequential in the pipeline — Whisper finishes before Ollama starts — so there is no simultaneous VRAM contention.

---

## Using the UI

1. **Select mode** — Online (Claude API) or Offline (local Ollama) using the toggle in the header
2. **Select patient** — Search by name; pulls from OpenMRS patient registry
3. **Upload audio** — MP3 or WAV recording of the Hindi consultation
4. **Review** — Edit any extracted field; manually add diagnoses using the input below the diagnosis list; fill in vitals (BP, HR, temperature, weight, height, SpO2, respiratory rate) and clinical notes
5. **Confirm** — Check the "I have reviewed the AI-extracted data" checkbox to unlock the Approve button
6. **Approve** — Click "Approve & Save to OpenMRS" — this is the only step that writes to OpenMRS

Switching to a different patient clears all extracted data from the previous session — no cross-patient data contamination.

The approval gate is intentional. AI output never enters patient records without a doctor's explicit sign-off.

---

## Running tests

```bash
# Fast structural tests — no GPU, no API key needed
python -m pytest tests/test_fhir_write.py -v

# Skip slow Whisper inference tests
python -m pytest tests/ -v -m "not slow"

# All tests (requires GPU + ANTHROPIC_API_KEY)
python -m pytest tests/ -v
```

---

## Running evaluations

```bash
# Generate corpus clips first (requires internet for Edge TTS)
python corpus/generate_corpus.py

# WER evaluation (requires GPU)
python evaluation/wer.py

# Entity F1 evaluation (requires GPU + ANTHROPIC_API_KEY)
python evaluation/entity_f1.py
```

---

## Benchmark results

### Whisper: Hindi vs Swahili

Whisper large-v3 on 6 matched synthetic clips per language (Edge TTS):

| Scenario | Hindi WER | Swahili WER | Winner |
|---|---|---|---|
| Fever and headache | 13.8% | 30.8% | Hindi |
| Chest pain | 19.4% | 10.7% | Swahili |
| Diabetes follow-up | 24.1% | 16.0% | Swahili |
| Cough and TB concern | 11.1% | 26.7% | Hindi |
| Stomach pain | 19.4% | 26.7% | Hindi |
| Joint pain | 21.9% | 20.0% | Swahili |
| **Average** | **18.3%** | **21.8%** | **Hindi** |

Hindi passes the 20% clinical usability threshold. Swahili is marginal. Hindi was selected as the project language.

> TTS audio is cleaner than real clinic audio. Expect 5–15% higher WER in production.

### faster-whisper vs openai-whisper (large-v3, INT8, CPU)

Benchmarked on all 6 Hindi clinical clips. Both models produce essentially identical transcripts — differences are limited to minor spelling variants (e.g. `दर्ध` vs `दर्द`) where faster-whisper is marginally more accurate.

| Scenario | openai-whisper | faster-whisper | Speedup |
|---|---|---|---|
| Fever and headache | 43.3s | 33.3s | 1.3x |
| Chest pain and breathlessness | 45.5s | 35.5s | 1.3x |
| Diabetes follow-up | 44.7s | 35.3s | 1.3x |
| Cough and TB concern | 47.1s | 37.0s | 1.3x |
| Stomach pain and vomiting | 45.7s | 35.9s | 1.3x |
| Joint pain | 46.3s | 36.0s | 1.3x |
| **Total** | **272.5s** | **213.0s** | **1.3x** |

ClinScribe uses faster-whisper (INT8 on CPU). The 1.3x speedup is consistent — roughly 10 seconds saved per consultation.

### Translation quality: MarianMT vs qwen2.5:7b vs Claude

Benchmarked on all 6 Hindi clinical clips. Quality measured by clinical keyword coverage and translation completeness (output-to-input word ratio).

| Scenario | MarianMT keywords | Qwen2.5:7b keywords | Claude keywords | Qwen word-ratio | Claude word-ratio |
|---|---|---|---|---|---|
| Fever and headache | 4/22 | 4/22 | 4/22 | 1.23 | 1.03 |
| Chest pain | 2/22 | 5/22 | 5/22 | 0.95 | 0.86 |
| Diabetes follow-up | 2/22 | 3/22 | 3/22 | 1.10 | 1.03 |
| Cough and TB concern | 6/22 | 7/22 | 7/22 | 1.14 | 1.11 |
| Stomach pain | 0/22 | 4/22 | 0/22 | 1.06 | — |
| Joint pain | 1/22 | 4/22 | 4/22 | 1.16 | 1.09 |
| **Total** | **15/132** | **27/132** | **23/132** | — | — |

> Word-ratio < 1.0 means the translation is shorter than the source — content is being dropped. MarianMT consistently under-translates (ratio 0.37–0.83 across clips). Qwen and Claude produce complete translations.

**Key examples showing the quality gap:**

Chest pain clip:
- **MarianMT:** *"I'm getting pain in the chest and I'm struggling to take my mother-in-law"* (hallucinated)
- **Qwen:** *"I am experiencing chest pain and difficulty breathing. This has been going on for two hours. I have a history of blood pressure issues."* (accurate)
- **Claude:** *"I am having chest pain and difficulty breathing. This has been happening for two hours. I already have blood pressure disease."* (accurate)

MarianMT speed advantage (~0.5s/clip) does not compensate for translation failures that cause downstream extraction to miss diagnoses entirely. ClinScribe uses Qwen2.5:7b for offline translation and Claude for online.

---

## Mode evaluation: Online vs Offline

Benchmarked on all 6 Hindi clinical audio clips using the same Whisper transcription. Latency excludes Whisper (identical for both modes).

> **Note:** The entity completeness and accuracy numbers below were measured with an earlier offline setup using MarianMT translation (Helsinki-NLP/opus-mt-hi-en) + 5-step Qwen2.5-3B-AWQ extraction. The current offline mode uses qwen2.5:7b (via Ollama) for both translation and extraction. Translation quality improved substantially (see benchmark above), and entity extraction accuracy is expected to improve accordingly since extraction runs on the English translation. Updated entity benchmarks for the current setup are pending.

### Entity completeness (earlier offline setup — MarianMT + Qwen2.5-3B)

Completeness = fraction of 8 required fields (chief complaint, HPI, symptoms, duration, medications, diagnosis, treatment plan, red flags) that are non-empty.

| Scenario | Online (Claude) | Offline (MarianMT + Qwen2.5-3B) |
|---|---|---|
| Fever and headache | 88% | 62% |
| Chest pain and breathlessness | 75% | 50% |
| Diabetes follow-up | 88% | 88% |
| Cough and TB concern | 88% | 50% |
| Stomach pain and vomiting | 88% | 62% |
| Joint pain | 75% | 50% |
| **Average** | **84%** | **60%** |

### Latency (translate + extract, excluding Whisper)

| Stage | Online | Offline (current) |
|---|---|---|
| Translation | 2.2s | ~5s |
| Extraction | 4.9s | ~3s (single-pass) |
| **Total** | **7.1s** | **~8s** |

### Entity extraction accuracy (online vs earlier offline)

| Metric | Online | Offline (MarianMT + Qwen2.5-3B) |
|---|---|---|
| Diagnoses extracted (clips with ≥1) | 3 / 6 | 0 / 6 |
| Medications correctly identified | 3 / 6 | 1 / 6 (wrong name) |
| Red flags raised on serious cases | Yes (5 flags on chest pain) | No |
| Hallucination detection active | Yes | Yes |
| Chief complaint accurate | 6 / 6 | 2 / 6 |

Online mode also correctly flagged two items as `potentially_hallucinated` in the TB clip (hemoptysis and suspected TB) — the doctor mentioned them as concerns, not confirmed diagnoses. This is the hallucination grounding check working as intended.

---

## Decision: Which mode to use

**Online mode (Claude) is the clear choice** for any deployment with internet access.

| | Online | Offline |
|---|---|---|
| Translation accuracy | Excellent — handles medical code-switching natively | Good — qwen2.5:7b handles mixed Hindi/English and has full context for ambiguous words like "कल" (yesterday vs tomorrow) |
| Entity completeness | 84% average | Improved with current Ollama pipeline; doctor reviews anyway |
| Diagnoses detected | Reliable | Reasonable; doctor must verify |
| Red flag detection | Reliable | Reasonable |
| Total latency | ~7s | ~8s (translate + extract) |
| Internet required | Yes | No |
| Cost | ~$0.005 per consultation | Free after setup |
| **Recommendation** | **Use this** | **Fallback only** |

**When to use offline mode:** Only when the clinic has no internet — for example, a rural health post with no connectivity. In that case, offline mode still transcribes correctly (Whisper is local in both modes) and produces a partial extraction the doctor can review and manually correct before approving.

---

## Key design decisions

| Decision | Why |
|---|---|
| Whisper runs locally only | Patient audio is PHI — sending to an external API is a HIPAA violation |
| faster-whisper with INT8 | 1.3x faster than openai-whisper on CPU with equivalent transcript quality (benchmarked); GPU reserved for Ollama |
| Whisper on CPU, Ollama on GPU | 6GB VRAM is fully used by qwen2.5:7b (~4.4GB). Whisper INT8 on CPU keeps GPU free. Pipeline is sequential so there is no VRAM contention. |
| Offline translation via Ollama, not MarianMT | MarianMT (Helsinki-NLP/opus-mt-hi-en) produced hallucinated translations on clinical text — e.g. "chest pain" became "struggling to take my mother-in-law", making downstream entity extraction fail entirely. qwen2.5:7b already runs for extraction, so using it for translation adds no memory cost and gives contextual quality. |
| Hindi medical glossary (external CSV) | Devanagari-transliterated English ("ओरस", "फूट पॉइस्निंग") is invisible to a 7B model. Pre-substituting known terms from a CSV before the LLM call fixes this without hardcoding in Python. |
| Single-pass offline extraction | Multi-step agentic extraction with a 7B model via Ollama is slow (~14s) because Ollama serializes requests. A single comprehensive prompt takes ~3s. The doctor reviews the output anyway, so minor recall differences are caught at the review stage. |
| Prompt caching on Claude extraction | System prompt is fixed per request — marking it `cache_control: ephemeral` saves input tokens and ~200ms after the first call within a 5-minute window. |
| Doctor must confirm + click Approve | A `confirm()` dialog can be bypassed by keyboard Enter. An explicit checkbox + separate button requires intentional action — unreviewed AI output in patient records is a patient safety risk. |
| State reset on patient switch | Without resetting, switching patients while transcription results are displayed could write Patient A's data to Patient B's record. `_resetTranscriptionState()` clears all state and UI on patient change. |
| No hardcoded ICD-10 / SNOMED codes | Wrong codes in patient records = clinical error risk; runtime lookup guarantees current codes |
| Medications as Text Observations | OpenMRS FHIR MedicationRequest returns 501 on standard deployments. A Text Observation with concept "Medication noted" is consistent with how clinical notes are saved and works on all instances. |
| Conditions written via REST v1, not FHIR | OpenMRS SPA Conditions widget reads from REST v1 — FHIR Conditions with no concept UUID show as blank |
| FastAPI not Flask | Whisper inference blocks 30–60s; Flask is synchronous and would block all other requests |
| Single HTML file frontend | No build step; one file to audit, deploy, and review |
| Extract from English translation, not Hindi | Diagnoses and symptoms need to be in English for OpenMRS concept matching and doctor review |
| Claude semantic concept matching | OpenMRS demo DB is sparse; Claude picks the closest existing concept before creating a new one |

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the review UI |
| `GET` | `/health` | OpenMRS connectivity + Whisper load status |
| `GET` | `/patients?q=<name>` | Patient name search (proxies OpenMRS REST) |
| `POST` | `/transcribe` | Full pipeline: transcribe → translate → extract → map codes |
| `POST` | `/approve` | Writes approved data to OpenMRS (Encounter + Conditions + Vitals + Medications) |

### POST /transcribe

```
Form fields:
  audio  — MP3 or WAV file
  mode   — "online" (default) or "offline"

Response:
  segments          — list of transcript segments with English translations
  entities          — extracted clinical entities (JSON)
  mapped_codes      — ICD-10 and SNOMED codes per diagnosis
  processing_time_seconds — per-stage timing breakdown
  mode              — echo of the requested mode
```

### POST /approve

```json
{
  "patient_uuid": "...",
  "edited_entities": {
    "chief_complaint": "...",
    "symptoms": [],
    "diagnosis": [],
    "medications_mentioned": [],
    "treatment_plan": "...",
    "red_flags": [],
    "duration": "...",
    "raw_language": "hi"
  },
  "mapped_codes": [],
  "vitals": {
    "bp_systolic": 120,
    "bp_diastolic": 80,
    "heart_rate": 72,
    "temperature": 37.0,
    "weight_kg": 65,
    "height_cm": 170,
    "spo2": 98,
    "respiratory_rate": 16
  },
  "clinical_notes": "Patient presents with..."
}
```

Response includes `encounter_uuid`, `conditions_saved`, `vitals_saved`, `medications_saved`, and a `success` / `partial` status.

---

## Logs

| File | What is logged |
|---|---|
| `logs/llm_calls.jsonl` | Every LLM call: timestamp, phase, model, tokens, latency, success |
| `logs/unmapped_terms.txt` | Terms with no ICD-10 or SNOMED match |
| `ERROR_LOG.md` | Full tracebacks for every runtime error, appended automatically |

---

## Known limitations

1. **Synthetic corpus only** — Benchmark audio is Edge TTS. Real clinic audio has background noise, accents, and disfluencies that will increase WER by an estimated 5–15%.

2. **No real-time streaming** — Whisper processes the full audio file before returning. Long consultations (>10 min) will show a spinner for the full duration.

3. **Single-user server** — One Whisper model instance. Concurrent requests queue. Production would need a job queue (Celery + Redis).

4. **SNOMED API availability** — `browser.ihtsdotools.org` is a public server with no SLA. If it is down, SNOMED codes are omitted and the fallback is text-only Conditions.

5. **OpenMRS concept coverage** — The OpenMRS demo database has limited diagnosis concepts. Unknown diagnoses are created as new concepts automatically, but they won't have linked SNOMED/ICD-10 coding in the OpenMRS concept dictionary.

6. **Patient must exist in OpenMRS** — The patient must be registered in OpenMRS before a consultation can be saved. The UI patient search will return no results for unregistered patients.

7. **Offline single-pass extraction trade-off** — The single-pass Ollama extraction (~3s) has lower recall on complex cases compared to a multi-step agentic approach (~14s). The doctor's review step is the intended safety net for any missed fields.
