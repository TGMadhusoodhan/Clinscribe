# ClinScribe

Hindi AI clinical scribe for OpenMRS. A doctor speaks in Hindi; ClinScribe transcribes, extracts clinical entities, maps SNOMED-CT and ICD-10 codes, and writes structured FHIR R4 records to OpenMRS — after explicit doctor review and approval.

**Course:** CS 595 Medical Informatics AI — Illinois Institute of Technology  
**Language:** Hindi (hi-IN) — chosen after benchmarking Whisper large-v3 on both Hindi and Swahili  
**WER (Hindi, synthetic corpus):** 18.3% average — below 20% clinical usability threshold

---

## Architecture

```
Audio (Hindi MP3/WAV)
      ↓
[silero-vad]  removes silence
      ↓
[Whisper large-v3]  local inference — audio never leaves the machine
      ↓  Hindi transcript (UTF-8)
[Claude API]  entity extraction → JSON (chief_complaint, symptoms, diagnosis, ...)
      ↓  structured entities
[ICD-10 + SNOMED APIs]  code lookup at runtime — no hardcoded codes
      ↓  mapped codes
[Doctor review UI]  editable transcript + clinical summary + approve button
      ↓  after explicit approval only
[OpenMRS FHIR R4]  POST Encounter → POST Condition(s)
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo>
cd clinscribe
python -m venv .venv
.venv/Scripts/activate    # Windows
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY
```

Minimum required:
```
ANTHROPIC_API_KEY=your_key_here
```

Optional (defaults shown):
```
OPENMRS_BASE_URL=http://localhost:8080/openmrs
OPENMRS_USER=admin
OPENMRS_PASSWORD=Admin123
WHISPER_MODEL=large-v3
WHISPER_LANGUAGE=hi
ANTHROPIC_MODEL=claude-sonnet-4-6
```

### 3. Start OpenMRS (Docker)

```bash
docker compose up -d
```

Wait ~2 minutes for the backend to initialise, then open http://localhost/openmrs to confirm it is running.

OpenMRS 3 disables HTTP Basic auth by default. Enable it once with the helper SQL:

```bash
docker compose exec db mysql -u openmrs -popenmrs openmrs < enable_basic_auth.sql
```

### 4. Generate corpus clips (optional — for evaluation)

```bash
python corpus/generate_corpus.py
```

### 5. Run the review UI

```bash
uvicorn review_ui.app:app --reload
```

Open http://localhost:8000

---

## Running tests

```bash
python -m pytest tests/test_fhir_write.py -v          # fast structural tests (no GPU, no API)
python -m pytest tests/ -v -m "not slow"               # skip Whisper inference tests
python -m pytest tests/ -v                             # all tests (requires GPU + Anthropic key)
```

---

## Running evaluations

```bash
# WER evaluation (requires corpus clips + GPU)
python evaluation/wer.py

# Entity F1 evaluation (requires corpus clips + GPU + Anthropic API key)
python evaluation/entity_f1.py
```

Results are saved to `evaluation/results/`.

---

## Benchmark results (Whisper large-v3, synthetic TTS corpus)

| Scenario               | Hindi WER | Swahili WER | Winner  |
|------------------------|-----------|-------------|---------|
| Fever and headache     | 13.8%     | 30.8%       | Hindi   |
| Chest pain             | 19.4%     | 10.7%       | Swahili |
| Diabetes follow-up     | 24.1%     | 16.0%       | Swahili |
| Cough and TB concern   | 11.1%     | 26.7%       | Hindi   |
| Stomach pain           | 19.4%     | 26.7%       | Hindi   |
| Joint pain             | 21.9%     | 20.0%       | Swahili |
| **Average**            | **18.3%** | **21.8%**   | **Hindi** |

Hindi is below the 20% clinical usability threshold. Swahili is marginal.  
Note: TTS audio is cleaner than real clinic audio. Expect 5–15% higher WER in production.

---

## Key constraints

| Constraint | Why |
|---|---|
| Whisper runs locally only | Patient audio is PHI — sending to external APIs is a HIPAA violation |
| All FHIR writes go to local OpenMRS | Writing to wrong system = data integrity failure |
| Doctor must click Approve | Unreviewed AI output in patient records = patient safety risk |
| Hindi text stored as UTF-8 | Transliteration makes records unsearchable |
| No hardcoded SNOMED/ICD-10 codes | Wrong codes in patient records = clinical error risk |

---

## Project structure

```
clinscribe/
├── pipeline/
│   ├── transcribe.py      Whisper + silero-vad
│   ├── extract.py         Claude entity extraction
│   ├── map_codes.py       SNOMED + ICD-10 API lookup
│   └── fhir_write.py      OpenMRS FHIR R4 write-back
├── review_ui/
│   ├── app.py             FastAPI backend
│   └── static/index.html  Single-file review UI
├── corpus/
│   ├── generate_corpus.py 10 clinical dialogue clips
│   └── benchmark_whisper.py Hindi vs Swahili WER benchmark
├── evaluation/
│   ├── wer.py             Whisper WER on corpus
│   └── entity_f1.py       Entity extraction F1
├── tests/
│   ├── test_transcribe.py
│   ├── test_extract.py
│   └── test_fhir_write.py
├── logs/
│   ├── llm_calls.jsonl    Every Anthropic API call logged
│   └── unmapped_terms.txt Terms with no SNOMED/ICD-10 match
├── config.py
├── requirements.txt
└── .env.example
```
