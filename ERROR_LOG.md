# ClinScribe — Error Log

No errors recorded yet. Every error, exception, test failure, and API problem
will be appended here immediately when encountered.

Format for each entry:

## ERROR — {datetime}
**Step:** Step N — filename
**What I was doing:** one sentence
**Error type:** ImportError / HTTPError / AssertionError / etc.
**Full error message:**
```
paste full traceback here
```
**What I tried:** describe each fix attempt
**Resolution:** FIXED / BLOCKED / WORKAROUND
**Impact on project:** none / delayed / blocked
---

## ERROR — 2026-04-19T18:10:32.296946+00:00
**Step:** Step 12 — /transcribe
**What I was doing:** Running pipeline stage: transcribe
**Error type:** RuntimeError
**Full error message:**
```
Traceback (most recent call last):
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\torchaudio\_torchcodec.py", line 82, in load_with_torchcodec
    from torchcodec.decoders import AudioDecoder
ModuleNotFoundError: No module named 'torchcodec'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\tgmad/.cache\torch\hub\snakers4_silero-vad_master\src\silero_vad\utils_vad.py", line 148, in read_audio
    wav, sr = torchaudio.load(path)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\torchaudio\__init__.py", line 86, in load
    return load_with_torchcodec(
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\torchaudio\_torchcodec.py", line 84, in load_with_torchcodec
    raise ImportError(
ImportError: TorchCodec is required for load_with_torchcodec. Please install torchcodec to use this function.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\tgmad/.cache\torch\hub\snakers4_silero-vad_master\src\silero_vad\utils_vad.py", line 151, in read_audio
    from torchcodec.decoders import AudioDecoder
ModuleNotFoundError: No module named 'torchcodec'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\review_ui\app.py", line 159, in transcribe_audio
    transcript_result = transcribe(tmp_path)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\transcribe.py", line 83, in transcribe
    audio_np = _remove_silence(audio_path)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\transcribe.py", line 57, in _remove_silence
    wav = _read_audio(audio_path, sampling_rate=16000)
  File "C:\Users\tgmad/.cache\torch\hub\snakers4_silero-vad_master\src\silero_vad\utils_vad.py", line 156, in read_audio
    raise RuntimeError(
RuntimeError: torchaudio version 2.11.0+cpu requires torchcodec for audio I/O. Install torchcodec or pin torchaudio < 2.9

```
**Resolution:** BLOCKED — needs investigation
---

## ERROR — 2026-04-19T18:13:31.099282+00:00
**Step:** Step 12 — /transcribe
**What I was doing:** Running pipeline stage: transcribe
**Error type:** RuntimeError
**Full error message:**
```
Traceback (most recent call last):
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\torchaudio\_torchcodec.py", line 82, in load_with_torchcodec
    from torchcodec.decoders import AudioDecoder
ModuleNotFoundError: No module named 'torchcodec'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\tgmad/.cache\torch\hub\snakers4_silero-vad_master\src\silero_vad\utils_vad.py", line 148, in read_audio
    wav, sr = torchaudio.load(path)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\torchaudio\__init__.py", line 86, in load
    return load_with_torchcodec(
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\torchaudio\_torchcodec.py", line 84, in load_with_torchcodec
    raise ImportError(
ImportError: TorchCodec is required for load_with_torchcodec. Please install torchcodec to use this function.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\tgmad/.cache\torch\hub\snakers4_silero-vad_master\src\silero_vad\utils_vad.py", line 151, in read_audio
    from torchcodec.decoders import AudioDecoder
ModuleNotFoundError: No module named 'torchcodec'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\review_ui\app.py", line 230, in transcribe_audio
    transcript_result = transcribe(tmp_path)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\transcribe.py", line 83, in transcribe
    audio_np = _remove_silence(audio_path)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\transcribe.py", line 57, in _remove_silence
    wav = _read_audio(audio_path, sampling_rate=16000)
  File "C:\Users\tgmad/.cache\torch\hub\snakers4_silero-vad_master\src\silero_vad\utils_vad.py", line 156, in read_audio
    raise RuntimeError(
RuntimeError: torchaudio version 2.11.0+cpu requires torchcodec for audio I/O. Install torchcodec or pin torchaudio < 2.9

```
**Resolution:** BLOCKED — needs investigation
---

## ERROR — 2026-04-19T18:16:43.830438+00:00
**Step:** Step 12 — /transcribe
**What I was doing:** Running pipeline stage: transcribe
**Error type:** RuntimeError
**Full error message:**
```
Traceback (most recent call last):
  File "C:\Users\tgmad/.cache\torch\hub\snakers4_silero-vad_master\src\silero_vad\utils_vad.py", line 143, in read_audio
    wav, sr = torchaudio.sox_effects.apply_effects_file(path, effects=effects)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\torchaudio\_internal\module_utils.py", line 71, in wrapped
    return func(*args, **kwargs)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\torchaudio\sox_effects\sox_effects.py", line 275, in apply_effects_file
    return sox_ext.apply_effects_file(path, effects, normalize, channels_first, format)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\torchaudio\_extension\utils.py", line 121, in __getattr__
    self._import_once()
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\torchaudio\_extension\utils.py", line 135, in _import_once
    self.module = self.import_func()
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\torchaudio\_extension\utils.py", line 85, in _init_sox
    ext = _import_sox_ext()
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\torchaudio\_extension\utils.py", line 66, in _import_sox_ext
    raise RuntimeError("sox extension is not supported on Windows")
RuntimeError: sox extension is not supported on Windows

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\review_ui\app.py", line 230, in transcribe_audio
    transcript_result = transcribe(tmp_path)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\transcribe.py", line 83, in transcribe
    audio_np = _remove_silence(audio_path)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\transcribe.py", line 57, in _remove_silence
    wav = _read_audio(audio_path, sampling_rate=16000)
  File "C:\Users\tgmad/.cache\torch\hub\snakers4_silero-vad_master\src\silero_vad\utils_vad.py", line 145, in read_audio
    wav, sr = torchaudio.load(path)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\torchaudio\_backend\utils.py", line 221, in load
    backend = dispatcher(uri, format, backend)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\torchaudio\_backend\utils.py", line 117, in dispatcher
    raise RuntimeError(f"Couldn't find appropriate backend to handle uri {uri} and format {format}.")
RuntimeError: Couldn't find appropriate backend to handle uri C:\Users\tgmad\AppData\Local\Temp\tmpaav6gvgt.mp3 and format None.

```
**Resolution:** BLOCKED — needs investigation
---

## ERROR — 2026-04-19T18:20:43.110497+00:00
**Step:** Step 12 — /transcribe
**What I was doing:** Running pipeline stage: transcribe
**Error type:** RuntimeError
**Full error message:**
```
Traceback (most recent call last):
  File "C:\Users\tgmad/.cache\torch\hub\snakers4_silero-vad_master\src\silero_vad\utils_vad.py", line 143, in read_audio
    wav, sr = torchaudio.sox_effects.apply_effects_file(path, effects=effects)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\torchaudio\_internal\module_utils.py", line 71, in wrapped
    return func(*args, **kwargs)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\torchaudio\sox_effects\sox_effects.py", line 275, in apply_effects_file
    return sox_ext.apply_effects_file(path, effects, normalize, channels_first, format)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\torchaudio\_extension\utils.py", line 121, in __getattr__
    self._import_once()
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\torchaudio\_extension\utils.py", line 135, in _import_once
    self.module = self.import_func()
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\torchaudio\_extension\utils.py", line 85, in _init_sox
    ext = _import_sox_ext()
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\torchaudio\_extension\utils.py", line 66, in _import_sox_ext
    raise RuntimeError("sox extension is not supported on Windows")
RuntimeError: sox extension is not supported on Windows

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\review_ui\app.py", line 245, in transcribe_audio
    transcript_result = transcribe(tmp_path)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\transcribe.py", line 83, in transcribe
    audio_np = _remove_silence(audio_path)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\transcribe.py", line 57, in _remove_silence
    wav = _read_audio(audio_path, sampling_rate=16000)
  File "C:\Users\tgmad/.cache\torch\hub\snakers4_silero-vad_master\src\silero_vad\utils_vad.py", line 145, in read_audio
    wav, sr = torchaudio.load(path)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\torchaudio\_backend\utils.py", line 221, in load
    backend = dispatcher(uri, format, backend)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\torchaudio\_backend\utils.py", line 117, in dispatcher
    raise RuntimeError(f"Couldn't find appropriate backend to handle uri {uri} and format {format}.")
RuntimeError: Couldn't find appropriate backend to handle uri C:\Users\tgmad\AppData\Local\Temp\tmpkm7n6vg_.wav and format None.

```
**Resolution:** BLOCKED — needs investigation
---

## ERROR — 2026-04-19T19:04:47.452033+00:00
**Step:** Step 12 — /approve
**What I was doing:** Writing to OpenMRS FHIR
**Error type:** HTTPError
**Full error message:**
```
Traceback (most recent call last):
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\review_ui\app.py", line 304, in approve_and_write
    result = write_encounter_and_conditions(
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\fhir_write.py", line 198, in write_encounter_and_conditions
    enc_resp = _post_fhir("Encounter", encounter_payload)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\fhir_write.py", line 172, in _post_fhir
    resp.raise_for_status()
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\requests\models.py", line 1028, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 400 Client Error:  for url: http://localhost:80/openmrs/ws/fhir2/R4/Encounter

```
**Resolution:** BLOCKED — needs investigation
---

## ERROR — 2026-04-19T19:11:44.992212+00:00
**Step:** Step 12 — /approve
**What I was doing:** Writing to OpenMRS FHIR
**Error type:** HTTPError
**Full error message:**
```
Traceback (most recent call last):
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\review_ui\app.py", line 304, in approve_and_write
    result = write_encounter_and_conditions(
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\fhir_write.py", line 198, in write_encounter_and_conditions
    enc_resp = _post_fhir("Encounter", encounter_payload)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\fhir_write.py", line 172, in _post_fhir
    resp.raise_for_status()
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\requests\models.py", line 1028, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 400 Client Error:  for url: http://localhost:80/openmrs/ws/fhir2/R4/Encounter

```
**Resolution:** BLOCKED — needs investigation
---

## ERROR — 2026-04-19T19:16:56.721680+00:00
**Step:** Step 12 — /approve
**What I was doing:** Writing to OpenMRS FHIR
**Error type:** HTTPError
**Full error message:**
```
Traceback (most recent call last):
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\review_ui\app.py", line 304, in approve_and_write
    result = write_encounter_and_conditions(
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\fhir_write.py", line 199, in write_encounter_and_conditions
    enc_resp = _post_fhir("Encounter", encounter_payload, session=session)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\fhir_write.py", line 172, in _post_fhir
    resp.raise_for_status()
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\requests\models.py", line 1028, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 400 Client Error:  for url: http://localhost:80/openmrs/ws/fhir2/R4/Encounter

```
**Resolution:** BLOCKED — needs investigation
---

## ERROR — 2026-04-19T19:21:31.755656+00:00
**Step:** Step 12 — /approve
**What I was doing:** Writing to OpenMRS FHIR
**Error type:** HTTPError
**Full error message:**
```
Traceback (most recent call last):
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\review_ui\app.py", line 304, in approve_and_write
    result = write_encounter_and_conditions(
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\fhir_write.py", line 199, in write_encounter_and_conditions
    enc_resp = _post_fhir("Encounter", encounter_payload, session=session)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\fhir_write.py", line 172, in _post_fhir
    resp.raise_for_status()
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\requests\models.py", line 1028, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 400 Client Error:  for url: http://localhost:80/openmrs/ws/fhir2/R4/Encounter

```
**Resolution:** BLOCKED — needs investigation
---

## ERROR — 2026-04-19T19:29:41.650180+00:00
**Step:** Step 12 — /approve
**What I was doing:** Writing to OpenMRS FHIR
**Error type:** InvalidFHIRError
**Full error message:**
```
Traceback (most recent call last):
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\review_ui\app.py", line 304, in approve_and_write
    result = write_encounter_and_conditions(
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\fhir_write.py", line 201, in write_encounter_and_conditions
    enc_resp = _post_fhir("Encounter", encounter_payload, session=session)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\fhir_write.py", line 170, in _post_fhir
    raise InvalidFHIRError(f"OpenMRS bad request for {resource_type}: {resp.text}")
pipeline.fhir_write.InvalidFHIRError: OpenMRS bad request for Encounter: {"resourceType":"OperationOutcome","text":{"status":"generated","div":"<div xmlns=\"http://www.w3.org/1999/xhtml\"><h1>Operation Outcome</h1><table border=\"0\"><tr><td style=\"font-weight: bold;\">ERROR</td><td>[]</td><td><pre>Invalid type of request</pre></td>\n\t\t\t</tr>\n\t\t</table>\n\t</div>"},"issue":[{"severity":"error","code":"processing","diagnostics":"Invalid type of request"}]}

```
**Resolution:** BLOCKED — needs investigation
---

## ERROR — 2026-04-19T19:37:03.950457+00:00
**Step:** Step 12 — /approve
**What I was doing:** Writing to OpenMRS FHIR
**Error type:** InvalidFHIRError
**Full error message:**
```
Traceback (most recent call last):
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\review_ui\app.py", line 304, in approve_and_write
    result = write_encounter_and_conditions(
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\fhir_write.py", line 200, in write_encounter_and_conditions
    enc_resp = _post_fhir("Encounter", encounter_payload, session=session)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\fhir_write.py", line 169, in _post_fhir
    raise InvalidFHIRError(f"OpenMRS bad request for {resource_type}: {resp.text}")
pipeline.fhir_write.InvalidFHIRError: OpenMRS bad request for Encounter: {"resourceType":"OperationOutcome","text":{"status":"generated","div":"<div xmlns=\"http://www.w3.org/1999/xhtml\"><h1>Operation Outcome</h1><table border=\"0\"><tr><td style=\"font-weight: bold;\">ERROR</td><td>[]</td><td><pre>Invalid type of request</pre></td>\n\t\t\t</tr>\n\t\t</table>\n\t</div>"},"issue":[{"severity":"error","code":"processing","diagnostics":"Invalid type of request"}]}

```
**Resolution:** BLOCKED — needs investigation
---

## ERROR — 2026-04-19T20:41:29.625801+00:00
**Step:** Step 12 — /approve
**What I was doing:** Writing to OpenMRS FHIR
**Error type:** InvalidFHIRError
**Full error message:**
```
Traceback (most recent call last):
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\review_ui\app.py", line 316, in approve_and_write
    result = write_encounter_and_conditions(
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\fhir_write.py", line 200, in write_encounter_and_conditions
    enc_resp = _post_fhir("Encounter", encounter_payload, session=session)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\fhir_write.py", line 169, in _post_fhir
    raise InvalidFHIRError(f"OpenMRS bad request for {resource_type}: {resp.text}")
pipeline.fhir_write.InvalidFHIRError: OpenMRS bad request for Encounter: {"resourceType":"OperationOutcome","text":{"status":"generated","div":"<div xmlns=\"http://www.w3.org/1999/xhtml\"><h1>Operation Outcome</h1><table border=\"0\"><tr><td style=\"font-weight: bold;\">ERROR</td><td>[]</td><td><pre>Invalid type of request</pre></td>\n\t\t\t</tr>\n\t\t</table>\n\t</div>"},"issue":[{"severity":"error","code":"processing","diagnostics":"Invalid type of request"}]}

```
**Resolution:** BLOCKED — needs investigation
---

## ERROR — 2026-04-19T20:53:28.344376+00:00
**Step:** Step 12 — /transcribe
**What I was doing:** Running pipeline stage: transcribe
**Error type:** JSONDecodeError
**Full error message:**
```
Traceback (most recent call last):
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\review_ui\app.py", line 262, in transcribe_audio
    translated_segments = translate_segments(list(transcript_result["segments"]))
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\transcribe.py", line 167, in translate_segments
    translations = json.loads(raw)
  File "C:\Users\tgmad\AppData\Local\Programs\Python\Python310\lib\json\__init__.py", line 346, in loads
    return _default_decoder.decode(s)
  File "C:\Users\tgmad\AppData\Local\Programs\Python\Python310\lib\json\decoder.py", line 337, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
  File "C:\Users\tgmad\AppData\Local\Programs\Python\Python310\lib\json\decoder.py", line 355, in raw_decode
    raise JSONDecodeError("Expecting value", s, err.value) from None
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)

```
**Resolution:** FIXED — Added regex fallback in translate_segments; strips markdown fences correctly; falls back to extracting JSON array with re.search if json.loads fails.
---

## SESSION SUMMARY — 2026-04-19
**What was fixed this session:**
- watchfiles constant reload: caused by .venv inside project dir being monitored. Fix: run `uvicorn review_ui.app:app` without `--reload`, or use `--reload-exclude ".venv"`
- `/transcribe` JSONDecodeError: Claude returning empty string after markdown fence stripping. Fixed regex in `translate_segments` + added fallback JSON extraction.
- `/approve` Encounter "Invalid type of request": OpenMRS FHIR requires `type` field with encounter type UUID. Fixed in `build_encounter_payload` — added `type: Consultation (dd528487-82a5-4082-9c72-ed246bd49591)`.
- Condition name blank in OpenMRS SPA: FHIR Condition API stores with `condition: null`; SPA only shows concept display name. Switched to REST v1 `/ws/rest/v1/condition` with concept lookup.
- Condition concept field wrong name: was sending `"concept"` but OpenMRS REST condition uses `"condition"` as the field name.
- Hindi diagnosis names: entities now extracted from English translation (not Hindi full_text), so diagnoses/symptoms are in English.
- Concept not found for "Arthritis": OpenMRS demo DB lacks many diagnosis concepts. Created: Arthritis, Osteoarthritis, Rheumatoid arthritis, Anemia, Skin rash, Knee pain. Common ones (Fever, Cough, Diabetes, Hypertension etc.) already existed.
- Concept search returning Drug concepts: added class filter (Finding/Symptom/Diagnosis) in `_lookup_concept`; also added term simplification to strip modifiers like "Suspected", "(age-related)".

**Current state:** Full pipeline working end-to-end. Conditions appear named in OpenMRS SPA Conditions widget.
---

## ERROR — 2026-04-27T18:29:49.996221+00:00
**Step:** Step 12 — /transcribe
**What I was doing:** Running pipeline stage: transcribe
**Error type:** ModuleNotFoundError
**Full error message:**
```
Traceback (most recent call last):
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\review_ui\app.py", line 275, in transcribe_audio
    translated_segments = translate_segments_offline(list(transcript_result["segments"]))
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\translate_offline.py", line 44, in translate_segments_offline
    ip, tokenizer, model, device = _load_models()
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\translate_offline.py", line 21, in _load_models
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
ModuleNotFoundError: No module named 'transformers'

```
**Resolution:** BLOCKED — needs investigation
---

## ERROR — 2026-04-27T18:39:34.330848+00:00
**Step:** Step 12 — /transcribe
**What I was doing:** Running pipeline stage: transcribe
**Error type:** ModuleNotFoundError
**Full error message:**
```
Traceback (most recent call last):
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\review_ui\app.py", line 275, in transcribe_audio
    translated_segments = translate_segments_offline(list(transcript_result["segments"]))
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\translate_offline.py", line 44, in translate_segments_offline
    ip, tokenizer, model, device = _load_models()
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\translate_offline.py", line 22, in _load_models
    from IndicTransToolkit.processor import IndicProcessor
ModuleNotFoundError: No module named 'IndicTransToolkit'

```
**Resolution:** BLOCKED — needs investigation
---

## ERROR — 2026-04-27T19:01:33.954227+00:00
**Step:** Step 12 — /transcribe
**What I was doing:** Running pipeline stage: transcribe
**Error type:** OSError
**Full error message:**
```
Traceback (most recent call last):
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\huggingface_hub\utils\_http.py", line 761, in hf_raise_for_status
    response.raise_for_status()
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\httpx\_models.py", line 829, in raise_for_status
    raise HTTPStatusError(message, request=request, response=self)
httpx.HTTPStatusError: Client error '401 Unauthorized' for url 'https://huggingface.co/ai4bharat/indictrans2-hi-en-dist-200M/resolve/main/config.json'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/401

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\transformers\utils\hub.py", line 422, in cached_files
    hf_hub_download(
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\huggingface_hub\utils\_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\huggingface_hub\file_download.py", line 997, in hf_hub_download
    return _hf_hub_download_to_cache_dir(
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\huggingface_hub\file_download.py", line 1148, in _hf_hub_download_to_cache_dir
    _raise_on_head_call_error(head_call_error, force_download, local_files_only)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\huggingface_hub\file_download.py", line 1782, in _raise_on_head_call_error
    raise head_call_error
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\huggingface_hub\file_download.py", line 1669, in _get_metadata_or_catch_error
    metadata = get_hf_file_metadata(
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\huggingface_hub\utils\_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\huggingface_hub\file_download.py", line 1591, in get_hf_file_metadata
    response = _httpx_follow_relative_redirects_with_backoff(
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\huggingface_hub\utils\_http.py", line 692, in _httpx_follow_relative_redirects_with_backoff
    hf_raise_for_status(response)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\huggingface_hub\utils\_http.py", line 835, in hf_raise_for_status
    raise _format(RepositoryNotFoundError, message, response, repo_type=repo_type, repo_id=repo_id) from e
huggingface_hub.errors.RepositoryNotFoundError: 401 Client Error. (Request ID: Root=1-69efb28e-796b7a2a60a489fa7f5bdc8f;0dee8be0-92ac-4d23-8deb-22c1654ae910)

Repository Not Found for url: https://huggingface.co/ai4bharat/indictrans2-hi-en-dist-200M/resolve/main/config.json.
Please make sure you specified the correct `repo_id` and `repo_type`.
If you are trying to access a private or gated repo, make sure you are authenticated and your token has the required permissions.
For more details, see https://huggingface.co/docs/huggingface_hub/authentication
Invalid username or password.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\transformers\models\auto\tokenization_auto.py", line 686, in from_pretrained
    config = AutoConfig.from_pretrained(
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\transformers\models\auto\configuration_auto.py", line 374, in from_pretrained
    config_dict, unused_kwargs = PreTrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\transformers\configuration_utils.py", line 673, in get_config_dict
    config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\transformers\configuration_utils.py", line 728, in _get_config_dict
    resolved_config_file = cached_file(
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\transformers\utils\hub.py", line 278, in cached_file
    file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\transformers\utils\hub.py", line 454, in cached_files
    raise OSError(
OSError: ai4bharat/indictrans2-hi-en-dist-200M is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `hf auth login` or by passing `token=<your_token>`

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\huggingface_hub\utils\_http.py", line 761, in hf_raise_for_status
    response.raise_for_status()
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\httpx\_models.py", line 829, in raise_for_status
    raise HTTPStatusError(message, request=request, response=self)
httpx.HTTPStatusError: Client error '401 Unauthorized' for url 'https://huggingface.co/ai4bharat/indictrans2-hi-en-dist-200M/resolve/main/config.json'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/401

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\transformers\utils\hub.py", line 422, in cached_files
    hf_hub_download(
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\huggingface_hub\utils\_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\huggingface_hub\file_download.py", line 997, in hf_hub_download
    return _hf_hub_download_to_cache_dir(
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\huggingface_hub\file_download.py", line 1148, in _hf_hub_download_to_cache_dir
    _raise_on_head_call_error(head_call_error, force_download, local_files_only)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\huggingface_hub\file_download.py", line 1782, in _raise_on_head_call_error
    raise head_call_error
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\huggingface_hub\file_download.py", line 1669, in _get_metadata_or_catch_error
    metadata = get_hf_file_metadata(
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\huggingface_hub\utils\_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\huggingface_hub\file_download.py", line 1591, in get_hf_file_metadata
    response = _httpx_follow_relative_redirects_with_backoff(
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\huggingface_hub\utils\_http.py", line 692, in _httpx_follow_relative_redirects_with_backoff
    hf_raise_for_status(response)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\huggingface_hub\utils\_http.py", line 835, in hf_raise_for_status
    raise _format(RepositoryNotFoundError, message, response, repo_type=repo_type, repo_id=repo_id) from e
huggingface_hub.errors.RepositoryNotFoundError: 401 Client Error. (Request ID: Root=1-69efb28e-74f5e6d54381f72d2d788958;4a54576a-f941-4a46-8d19-aecaa11c4c48)

Repository Not Found for url: https://huggingface.co/ai4bharat/indictrans2-hi-en-dist-200M/resolve/main/config.json.
Please make sure you specified the correct `repo_id` and `repo_type`.
If you are trying to access a private or gated repo, make sure you are authenticated and your token has the required permissions.
For more details, see https://huggingface.co/docs/huggingface_hub/authentication
Invalid username or password.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\review_ui\app.py", line 275, in transcribe_audio
    translated_segments = translate_segments_offline(list(transcript_result["segments"]))
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\translate_offline.py", line 44, in translate_segments_offline
    ip, tokenizer, model, device = _load_models()
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\translate_offline.py", line 26, in _load_models
    tokenizer = AutoTokenizer.from_pretrained(_INDICTRANS_MODEL, trust_remote_code=True)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\transformers\models\auto\tokenization_auto.py", line 690, in from_pretrained
    config = PreTrainedConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\transformers\configuration_utils.py", line 632, in from_pretrained
    config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\transformers\configuration_utils.py", line 673, in get_config_dict
    config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\transformers\configuration_utils.py", line 728, in _get_config_dict
    resolved_config_file = cached_file(
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\transformers\utils\hub.py", line 278, in cached_file
    file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\transformers\utils\hub.py", line 454, in cached_files
    raise OSError(
OSError: ai4bharat/indictrans2-hi-en-dist-200M is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `hf auth login` or by passing `token=<your_token>`

```
**Resolution:** BLOCKED — needs investigation
---

## ERROR — 2026-04-28T16:29:40.991966+00:00
**Step:** Step 12 — /transcribe
**What I was doing:** Running pipeline stage: transcribe
**Error type:** APIConnectionError
**Full error message:**
```
Traceback (most recent call last):
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\httpx\_transports\default.py", line 101, in map_httpcore_exceptions
    yield
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\httpx\_transports\default.py", line 250, in handle_request
    resp = self._pool.handle_request(req)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\httpcore\_sync\connection_pool.py", line 256, in handle_request
    raise exc from None
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\httpcore\_sync\connection_pool.py", line 236, in handle_request
    response = connection.handle_request(
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\httpcore\_sync\connection.py", line 101, in handle_request
    raise exc
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\httpcore\_sync\connection.py", line 78, in handle_request
    stream = self._connect(request)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\httpcore\_sync\connection.py", line 124, in _connect
    stream = self._network_backend.connect_tcp(**kwargs)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\httpcore\_backends\sync.py", line 207, in connect_tcp
    with map_exceptions(exc_map):
  File "C:\Users\tgmad\AppData\Local\Programs\Python\Python310\lib\contextlib.py", line 153, in __exit__
    self.gen.throw(typ, value, traceback)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\httpcore\_exceptions.py", line 14, in map_exceptions
    raise to_exc(exc) from exc
httpcore.ConnectError: [WinError 10061] No connection could be made because the target machine actively refused it

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\openai\_base_client.py", line 1019, in request
    response = self._send_request(
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\openai\_client.py", line 400, in _send_request
    return self._send_with_auth_retry(request, stream=stream, **kwargs)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\openai\_client.py", line 378, in _send_with_auth_retry
    response = super()._send_request(request, stream=stream, **kwargs)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\openai\_base_client.py", line 947, in _send_request
    return self._client.send(request, stream=stream, **kwargs)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\httpx\_client.py", line 914, in send
    response = self._send_handling_auth(
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\httpx\_client.py", line 942, in _send_handling_auth
    response = self._send_handling_redirects(
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\httpx\_client.py", line 979, in _send_handling_redirects
    response = self._send_single_request(request)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\httpx\_client.py", line 1014, in _send_single_request
    response = transport.handle_request(request)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\httpx\_transports\default.py", line 249, in handle_request
    with map_httpcore_exceptions():
  File "C:\Users\tgmad\AppData\Local\Programs\Python\Python310\lib\contextlib.py", line 153, in __exit__
    self.gen.throw(typ, value, traceback)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\httpx\_transports\default.py", line 118, in map_httpcore_exceptions
    raise mapped_exc(message) from exc
httpx.ConnectError: [WinError 10061] No connection could be made because the target machine actively refused it

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\review_ui\app.py", line 289, in transcribe_audio
    entities = extract_offline(english_full_text)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\extract_offline.py", line 153, in extract_offline
    data = _call_vllm(transcript)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\pipeline\extract_offline.py", line 119, in _call_vllm
    response = client.chat.completions.create(
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\openai\_utils\_utils.py", line 287, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\openai\resources\chat\completions\completions.py", line 1211, in create
    return self._post(
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\openai\_base_client.py", line 1314, in post
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
  File "C:\Users\tgmad\OneDrive\Desktop\clinscribe\.venv\lib\site-packages\openai\_base_client.py", line 1054, in request
    raise APIConnectionError(request=request) from err
openai.APIConnectionError: Connection error.

```
**Resolution:** BLOCKED — needs investigation
---
