# ClinScribe

Hindi AI clinical scribe for OpenMRS. A doctor speaks in Hindi; ClinScribe transcribes, translates to English, extracts structured clinical entities, maps SNOMED-CT and ICD-10 codes, and writes FHIR R4 records to OpenMRS — after explicit doctor review and approval.

**Course:** CS 595 Medical Informatics AI — Illinois Institute of Technology  
**Language:** Hindi (hi-IN) — chosen after benchmarking Whisper large-v3 on both Hindi and Swahili  
**WER (Hindi, synthetic corpus):** 18.3% average — below the 20% clinical usability threshold

---

## What it does

1. Doctor uploads a Hindi audio recording (MP3/WAV) of a patient consultation
2. ClinScribe transcribes the audio locally using Whisper large-v3
3. The transcript is translated to English
4. Claude (or a local Qwen model in offline mode) extracts structured clinical entities: chief complaint, symptoms, diagnoses, medications, treatment plan, red flags
5. Diagnoses are mapped to ICD-10 and SNOMED-CT codes via public APIs
6. The doctor reviews and edits all extracted data in a web UI
7. The doctor explicitly clicks **Approve** — only then is data written to OpenMRS
8. An FHIR R4 Encounter and Condition records are posted to the local OpenMRS instance

---

## Pipeline

```
Audio (Hindi MP3/WAV)
      ↓
[silero-vad]              removes silence — reduces WER on clinic recordings
      ↓
[Whisper large-v3]        local inference — audio never leaves the machine (HIPAA)
      ↓  Hindi transcript with word-level timestamps
[Translation]             Online: Claude API  |  Offline: Helsinki-NLP/opus-mt-hi-en
      ↓  English transcript
[Entity Extraction]       Online: Claude API  |  Offline: Qwen2.5:3B via Ollama
      ↓  {chief_complaint, symptoms, diagnosis, medications, treatment_plan, red_flags}
[ICD-10 + SNOMED APIs]   runtime lookup — no hardcoded codes
      ↓  [{term, icd10, snomed, confidence}]
[Doctor Review UI]        editable, with hallucination flags and manual diagnosis input
      ↓  after explicit doctor approval only
[OpenMRS FHIR R4]        POST Encounter → POST Condition(s) via REST v1
```

---

## Two modes

| | Online | Offline |
|---|---|---|
| **Translation** | Claude API | Helsinki-NLP/opus-mt-hi-en (MarianMT) |
| **Entity extraction** | Claude API | Qwen2.5:3B via Ollama (local) |
| **Internet required** | Yes (Anthropic API) | No (fully local) |
| **Setup** | Just an API key | Ollama + model pull |
| **Use case** | Default / clinic with internet | Rural deployment, air-gapped |

Toggle between modes using the switch in the top-right of the UI.

---

## Project structure

```
clinscribe/
├── pipeline/
│   ├── transcribe.py         Whisper large-v3 + silero-vad; online Claude translation
│   ├── translate_offline.py  Offline translation via Helsinki-NLP/opus-mt-hi-en
│   ├── extract.py            Online entity extraction via Claude API
│   ├── extract_offline.py    Offline entity extraction via vLLM (Qwen2.5)
│   ├── map_codes.py          ICD-10 (icd10api.com) + SNOMED-CT (ihtsdotools.org) lookup
│   └── fhir_write.py         OpenMRS FHIR R4 Encounter + REST v1 Condition write-back
├── review_ui/
│   ├── app.py                FastAPI backend — /transcribe, /approve, /patients, /health
│   └── static/index.html     Single-file review UI (no build step)
├── corpus/
│   ├── generate_corpus.py    Generates 10 Hindi clinical dialogue clips via Edge TTS
│   └── benchmark_whisper.py  Hindi vs Swahili WER benchmark
├── evaluation/
│   ├── wer.py                WER evaluation on corpus clips
│   └── entity_f1.py          Entity extraction F1 evaluation
├── tests/
│   ├── conftest.py
│   ├── test_transcribe.py
│   ├── test_extract.py
│   └── test_fhir_write.py
├── logs/
│   ├── llm_calls.jsonl       Every LLM API call logged (phase, tokens, latency, success)
│   └── unmapped_terms.txt    Terms with no ICD-10 or SNOMED match
├── benchmark_audio/          12 MP3 clips — 6 Hindi, 6 Swahili
├── config.py                 Loads all env vars from .env; single import for all modules
├── docker-compose.yml        OpenMRS + MySQL
├── enable_basic_auth.sql     One-time OpenMRS auth fix for REST API access
├── requirements.txt
├── .env.example
└── ERROR_LOG.md              All runtime errors appended here automatically
```

---

## Setup

### Prerequisites

- Python 3.10+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for OpenMRS)
- [Ollama](https://ollama.com/download) (for offline mode entity extraction)
- FFmpeg — `winget install ffmpeg` on Windows, `brew install ffmpeg` on macOS
- GPU recommended (CUDA 11.8+) — CPU works but transcription is slow
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

# OpenMRS (defaults work with docker-compose)
OPENMRS_BASE_URL=http://localhost:8080/openmrs
OPENMRS_USER=admin
OPENMRS_PASSWORD=Admin123

# Whisper
WHISPER_MODEL=large-v3
WHISPER_LANGUAGE=hi

# Claude model
ANTHROPIC_MODEL=claude-sonnet-4-6

# Offline mode — Ollama server
VLLM_BASE_URL=http://localhost:11434/v1
VLLM_MODEL=qwen2.5:3b
```

### 4. Start OpenMRS

```bash
docker compose up -d
```

Wait ~2 minutes for startup. Visit `http://localhost/openmrs` to confirm it is running.

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

Offline mode requires two extra components:

### Translation model (auto-downloads on first use)

`Helsinki-NLP/opus-mt-hi-en` downloads automatically from HuggingFace (~300MB) on first offline run. No account needed.

### Ollama for entity extraction

Offline entity extraction runs via [Ollama](https://ollama.com/download), which works natively on Windows, macOS, and Linux.

> **Why Ollama, not vLLM?** vLLM requires `uvloop` which is Linux-only and will not start on Windows. Ollama exposes the same OpenAI-compatible API and works on all platforms.

1. Install Ollama from [ollama.com/download](https://ollama.com/download)

2. Pull the model (~2GB):
   ```bash
   ollama pull qwen2.5:3b
   ```

3. Set these in `.env`:
   ```env
   VLLM_BASE_URL=http://localhost:11434/v1
   VLLM_MODEL=qwen2.5:3b
   ```

4. Start Ollama before running ClinScribe:
   ```bash
   ollama serve
   ```

Then start ClinScribe normally and flip the **Offline** toggle in the UI header.

---

## Using the UI

1. **Select mode** — Online (Claude API) or Offline (local models) using the toggle in the header
2. **Select patient** — Search by name; pulls from OpenMRS patient registry
3. **Upload audio** — MP3 or WAV recording of the Hindi consultation
4. **Review** — Edit any extracted field; manually add diagnoses using the input below the diagnosis list
5. **Approve** — Click "Approve & Save to OpenMRS" — this is the only step that writes to OpenMRS

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

---

## Key design decisions

| Decision | Why |
|---|---|
| Whisper runs locally only | Patient audio is PHI — sending to an external API is a HIPAA violation |
| Doctor must click Approve | Unreviewed AI output in patient records = patient safety risk |
| No hardcoded ICD-10 / SNOMED codes | Wrong codes in patient records = clinical error risk; runtime lookup guarantees current codes |
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
| `POST` | `/approve` | Writes approved data to OpenMRS (Encounter + Conditions) |

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
    "symptoms": [...],
    "diagnosis": [...],
    "medications_mentioned": [...],
    "treatment_plan": "...",
    "red_flags": [...],
    "duration": "...",
    "raw_language": "hi"
  },
  "mapped_codes": [...]
}
```

---

## Logs

| File | What is logged |
|---|---|
| `logs/llm_calls.jsonl` | Every LLM call: timestamp, phase, model, tokens, latency, success |
| `logs/unmapped_terms.txt` | Terms with no ICD-10 or SNOMED match |
| `ERROR_LOG.md` | Full tracebacks for every runtime error, appended automatically |

---

## Mode evaluation: Online vs Offline

Benchmarked on all 6 Hindi clinical audio clips using the same Whisper transcription. Latency excludes Whisper (identical for both modes). Offline mode uses MarianMT (translation) + Qwen2.5:3B via Ollama (extraction).

### Entity completeness

Completeness = fraction of 8 required fields (chief complaint, HPI, symptoms, duration, medications, diagnosis, treatment plan, red flags) that are non-empty.

| Scenario | Online (Claude) | Offline (MarianMT + Qwen 3B) |
|---|---|---|
| Fever and headache | 88% | 62% |
| Chest pain and breathlessness | 75% | 50% |
| Diabetes follow-up | 88% | 88% |
| Cough and TB concern | 88% | 50% |
| Stomach pain and vomiting | 88% | 62% |
| Joint pain | 75% | 50% |
| **Average** | **84%** | **60%** |

### Latency (translate + extract, excluding Whisper)

| Stage | Online | Offline |
|---|---|---|
| Translation | 2.2s | 4.4s |
| Extraction | 4.9s | 18.6s |
| **Total** | **7.1s** | **23.0s** |

### Translation quality

This is where the gap is most visible. The offline MarianMT model fails on medical Hindi because clinical speech mixes Hindi grammar with English medical terms (code-switching), which MarianMT was not trained for.

| Clip | Online (Claude) | Offline (MarianMT) |
|---|---|---|
| Chest pain | *"I am having chest pain and difficulty breathing"* | *"I'm growing up in the chest and suffering from taking in my mother-in-law"* |
| Diabetes | *"Fasting blood sugar came out to be 180. Are you taking Metformin?"* | *"Fasting blood pressure has come 180. Are you taking Metzin?"* |
| TB concern | *"Sometimes there is also blood in the sputum. Have you come in contact with any TB patient?"* | *"For two weeks there's been blood at any given time. If you come into contact with a TB patient, I've got T-B."* |
| Joint pain | *"There is pain in both knees... Arthritis can occur with age."* | *"Both knees are under the weight of a lot of pain... Arthur might have been diagnosed"* |

The offline translations are partially or entirely wrong on 4 of 6 clips. Because the extraction runs on the translated text, these errors cascade — wrong translation → wrong entities.

### Entity extraction accuracy

| Metric | Online | Offline |
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
| Translation accuracy | Excellent — handles medical code-switching natively | Poor — MarianMT fails on Hindi+English mixed speech |
| Entity completeness | 84% average | 60% average |
| Diagnoses detected | 3/6 clips | 0/6 clips |
| Red flag detection | Reliable | Unreliable |
| Total latency | 7.1s | 23.0s |
| Internet required | Yes | No |
| Cost | ~$0.005 per consultation | Free after setup |
| **Recommendation** | **Use this** | **Fallback only** |

**When to use offline mode:** Only when the clinic has no internet — for example, a rural health post with no connectivity. In that case, offline mode still transcribes correctly (Whisper is local in both modes) and produces a partial extraction the doctor can review and manually correct before approving. The 3.2x slower latency and lower extraction quality are acceptable tradeoffs when internet is genuinely unavailable.

**Root cause of offline gap:** The bottleneck is translation, not extraction. MarianMT (`Helsinki-NLP/opus-mt-hi-en`) is a 300MB model trained on general Hindi text — it was not trained on clinical speech that mixes Hindi and English medical terminology. A larger offline translation model (e.g. NLLB-1.3B) would close most of the gap, at the cost of higher memory and latency.

---

## Known limitations

1. **Synthetic corpus only** — Benchmark audio is Edge TTS. Real clinic audio has background noise, accents, and disfluencies that will increase WER by an estimated 5–15%.

2. **No real-time streaming** — Whisper processes the full audio file before returning. Long consultations (>10 min) will show a spinner for the full duration.

3. **Single-user server** — One Whisper model instance. Concurrent requests queue. Production would need a job queue (Celery + Redis).

4. **SNOMED API availability** — `browser.ihtsdotools.org` is a public server with no SLA. If it is down, SNOMED codes are omitted and the fallback is text-only Conditions.

5. **OpenMRS concept coverage** — The OpenMRS demo database has limited diagnosis concepts. Unknown diagnoses are created as new concepts automatically, but they won't have linked SNOMED/ICD-10 coding.

6. **Patient must exist in OpenMRS** — The patient must be registered in OpenMRS before a consultation can be saved. The UI patient search will return no results for unregistered patients.

7. **Offline translation quality** — `Helsinki-NLP/opus-mt-hi-en` is a compact MarianMT model. Translation quality is lower than Claude, which may reduce entity extraction accuracy in offline mode.
