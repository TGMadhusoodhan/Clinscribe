"""
ClinScribe corpus generator.
Generates 10 Hindi clinical dialogue clips using Edge TTS and writes ground-truth JSON.
Verifies voices exist before use — substitutes if missing.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import edge_tts

_CLIPS_DIR = Path(__file__).parent / "clips"

# Hindi TTS voices — verified present in edge-tts as of 2026-04.
# run `edge-tts --list-voices | grep hi-IN` to re-verify.
_DOCTOR_VOICE = "hi-IN-MadhurNeural"
_PATIENT_VOICE = "hi-IN-SwaraNeural"

# ── Clinical dialogue scripts ────────────────────────────────────────────────
# Each entry: list of (speaker: "D"/"P", text: str)
# D = Doctor (MadhurNeural), P = Patient (SwaraNeural)
# Natural code-switching: patients say "blood pressure", "sugar", "fever" in English.

SCRIPTS = [
    {
        "id": 1,
        "slug": "fever_headache",
        "scenario": "Fever and headache",
        "expected_entities": {
            "chief_complaint": "बुखार और सिर में दर्द",
            "symptoms": ["बुखार", "सिर दर्द"],
            "duration": "तीन दिन",
            "medications_mentioned": ["पैरासिटामोल"],
            "diagnosis": ["viral fever"],
            "treatment_plan": "पैरासिटामोल और आराम",
        },
        "lines": [
            ("P", "डॉक्टर साहब, मुझे तीन दिनों से बुखार है।"),
            ("D", "कितना temperature है?"),
            ("P", "कल रात 102 degree था। सिर में भी बहुत दर्द है।"),
            ("D", "कोई दवाई ली क्या?"),
            ("P", "नहीं, कोई दवाई नहीं ली अभी तक।"),
            ("D", "ठीक है। यह viral fever लग रहा है। पानी खूब पिएं और पैरासिटामोल लें।"),
            ("P", "ठीक है डॉक्टर। कितने दिन में ठीक होगा?"),
            ("D", "दो से तीन दिन में ठीक हो जाएगा। अगर बुखार नहीं उतरा तो वापस आना।"),
        ],
    },
    {
        "id": 2,
        "slug": "chest_pain",
        "scenario": "Chest pain with hypertension history",
        "expected_entities": {
            "chief_complaint": "सीने में दर्द और सांस लेने में तकलीफ",
            "symptoms": ["सीने में दर्द", "सांस की तकलीफ"],
            "duration": "दो घंटे",
            "medications_mentioned": ["blood pressure की दवाई"],
            "diagnosis": ["chest pain", "hypertension"],
            "treatment_plan": "ECG और आगे जांच",
        },
        "lines": [
            ("P", "डॉक्टर, मुझे सीने में दर्द हो रहा है।"),
            ("D", "कब से है?"),
            ("P", "दो घंटे से। सांस लेने में भी तकलीफ है।"),
            ("D", "क्या पहले से कोई बीमारी है?"),
            ("P", "हाँ, blood pressure की बीमारी है। दवाई भी लेता हूँ।"),
            ("D", "यह बहुत गंभीर लक्षण हैं। अभी ECG करते हैं।"),
            ("P", "कोई खतरा तो नहीं है ना?"),
            ("D", "घबराएं नहीं, हम जांच करके बताएंगे।"),
        ],
    },
    {
        "id": 3,
        "slug": "diabetes_followup",
        "scenario": "Diabetes follow-up with blood sugar review",
        "expected_entities": {
            "chief_complaint": "sugar की जांच और follow-up",
            "symptoms": ["थकान"],
            "duration": "तीन महीने से दवाई",
            "medications_mentioned": ["metformin"],
            "diagnosis": ["diabetes mellitus", "hyperglycemia"],
            "treatment_plan": "खाने में चीनी कम करें, metformin जारी रखें",
        },
        "lines": [
            ("P", "डॉक्टर, मेरी sugar की जांच करवाई।"),
            ("D", "क्या आया fasting blood sugar?"),
            ("P", "180 आई है। थोड़ी थकान भी होती है।"),
            ("D", "क्या आप metformin ले रहे हैं?"),
            ("P", "हाँ, रोज़ सुबह लेता हूँ।"),
            ("D", "खाने में चीनी और चावल कम करें। metformin जारी रखें।"),
            ("P", "Exercise करना चाहिए क्या?"),
            ("D", "हाँ, रोज़ आधा घंटा walking करें।"),
        ],
    },
    {
        "id": 4,
        "slug": "cough_tb_concern",
        "scenario": "Persistent cough with TB exposure history",
        "expected_entities": {
            "chief_complaint": "दो हफ्ते से खांसी",
            "symptoms": ["खांसी", "बलगम में खून"],
            "duration": "दो हफ्ते",
            "medications_mentioned": [],
            "diagnosis": ["suspected tuberculosis", "hemoptysis"],
            "treatment_plan": "chest X-ray और sputum test",
        },
        "lines": [
            ("P", "डॉक्टर, दो हफ्तों से खांसी है।"),
            ("D", "क्या बलगम आता है?"),
            ("P", "हाँ, कभी-कभी बलगम में खून भी आता है।"),
            ("D", "क्या आप किसी TB के मरीज़ के संपर्क में आए?"),
            ("P", "हाँ, मेरे पड़ोसी को TB है।"),
            ("D", "सीने का X-ray और sputum test करना होगा।"),
            ("P", "क्या मुझे TB हो सकता है?"),
            ("D", "जांच के बाद पता चलेगा। घबराएं नहीं।"),
        ],
    },
    {
        "id": 5,
        "slug": "stomach_pain",
        "scenario": "Stomach pain, vomiting and diarrhea after outside food",
        "expected_entities": {
            "chief_complaint": "पेट दर्द और उल्टी",
            "symptoms": ["पेट दर्द", "उल्टी", "दस्त"],
            "duration": "कल रात से",
            "medications_mentioned": ["ORS"],
            "diagnosis": ["food poisoning", "gastroenteritis"],
            "treatment_plan": "ORS और आराम",
        },
        "lines": [
            ("P", "डॉक्टर, पेट में बहुत दर्द है और उल्टी भी हो रही है।"),
            ("D", "कब से है?"),
            ("P", "कल रात से। दस्त भी हैं।"),
            ("D", "क्या बाहर का खाना खाया था?"),
            ("P", "हाँ, कल शाम को बाहर खाया था।"),
            ("D", "यह food poisoning लग रही है। ORS पीते रहें।"),
            ("P", "कोई दवाई लेनी है क्या?"),
            ("D", "अभी सिर्फ पानी और ORS। कल सुबह आना।"),
        ],
    },
    {
        "id": 6,
        "slug": "skin_rash",
        "scenario": "Skin rash with itching",
        "expected_entities": {
            "chief_complaint": "त्वचा पर दाने और खुजली",
            "symptoms": ["त्वचा पर दाने", "खुजली", "जलन"],
            "duration": "चार दिन",
            "medications_mentioned": ["antihistamine"],
            "diagnosis": ["allergic reaction", "urticaria"],
            "treatment_plan": "antihistamine और cream",
        },
        "lines": [
            ("P", "डॉक्टर, त्वचा पर बहुत दाने हो गए हैं।"),
            ("D", "कब से है? खुजली है?"),
            ("P", "चार दिन से। बहुत खुजली और जलन है।"),
            ("D", "क्या कोई नई दवाई ली या कुछ नया खाया?"),
            ("P", "नहीं, कुछ नया नहीं।"),
            ("D", "यह allergic reaction लग रहा है। antihistamine देता हूँ।"),
        ],
    },
    {
        "id": 7,
        "slug": "joint_pain",
        "scenario": "Bilateral knee pain with difficulty climbing stairs",
        "expected_entities": {
            "chief_complaint": "घुटनों में दर्द",
            "symptoms": ["घुटनों में दर्द", "सीढ़ी चढ़ने में तकलीफ"],
            "duration": "तीन महीने",
            "medications_mentioned": [],
            "diagnosis": ["arthritis", "osteoarthritis"],
            "treatment_plan": "X-ray और blood test",
        },
        "lines": [
            ("P", "डॉक्टर, दोनों घुटनों में बहुत दर्द है।"),
            ("D", "कितने समय से है?"),
            ("P", "तीन महीने से। सीढ़ियाँ चढ़ने में बहुत तकलीफ होती है।"),
            ("D", "सुबह उठते समय जकड़न होती है?"),
            ("P", "हाँ, सुबह ज़्यादा दर्द होता है।"),
            ("D", "उम्र के साथ arthritis हो सकता है। X-ray और blood test करेंगे।"),
        ],
    },
    {
        "id": 8,
        "slug": "pregnancy_checkup",
        "scenario": "Pregnancy check-up at 6 months",
        "expected_entities": {
            "chief_complaint": "गर्भावस्था जांच",
            "symptoms": ["पैरों में सूजन", "थकान"],
            "duration": "छह महीने की गर्भावस्था",
            "medications_mentioned": ["iron tablets", "folic acid"],
            "diagnosis": ["normal pregnancy", "edema"],
            "treatment_plan": "iron tablets और folic acid जारी",
        },
        "lines": [
            ("P", "डॉक्टर, छह महीने की गर्भावस्था है। जांच करवानी है।"),
            ("D", "कोई तकलीफ है?"),
            ("P", "पैरों में थोड़ी सूजन है और थकान रहती है।"),
            ("D", "blood pressure और weight check करते हैं।"),
            ("P", "ठीक है।"),
            ("D", "सब normal है। iron tablets और folic acid जारी रखें।"),
        ],
    },
    {
        "id": 9,
        "slug": "child_fever",
        "scenario": "Child with high fever and cold",
        "expected_entities": {
            "chief_complaint": "बच्चे को बुखार और जुकाम",
            "symptoms": ["बुखार", "जुकाम", "खांसी"],
            "duration": "दो दिन",
            "medications_mentioned": ["paracetamol syrup"],
            "diagnosis": ["upper respiratory infection", "viral fever"],
            "treatment_plan": "paracetamol syrup और आराम",
        },
        "lines": [
            ("P", "डॉक्टर, मेरे बच्चे को दो दिन से बुखार है।"),
            ("D", "कितना बुखार है?"),
            ("P", "101 degree। जुकाम और खांसी भी है।"),
            ("D", "कोई दवाई दी?"),
            ("P", "नहीं, घर पर कोई दवाई नहीं है।"),
            ("D", "paracetamol syrup दो। तीन दिन में ठीक हो जाएगा।"),
        ],
    },
    {
        "id": 10,
        "slug": "eye_irritation",
        "scenario": "Eye redness and discharge",
        "expected_entities": {
            "chief_complaint": "आँखों में लालिमा और जलन",
            "symptoms": ["आँखों में लालिमा", "जलन", "पानी आना"],
            "duration": "तीन दिन",
            "medications_mentioned": ["eye drops"],
            "diagnosis": ["conjunctivitis"],
            "treatment_plan": "antibiotic eye drops",
        },
        "lines": [
            ("P", "डॉक्टर, आँखें बहुत लाल हो गई हैं।"),
            ("D", "कब से है? जलन है?"),
            ("P", "तीन दिन से। बहुत जलन होती है और पानी आता है।"),
            ("D", "क्या किसी infected person के साथ थे?"),
            ("P", "हाँ, मेरे दोस्त की आँखें भी लाल थीं।"),
            ("D", "यह conjunctivitis है। antibiotic eye drops दे रहा हूँ।"),
        ],
    },
]


async def _generate_clip(script: dict, voice_map: dict) -> str:
    """
    What it does: Generates a single multi-turn audio clip by concatenating TTS lines.
    Inputs: script — dict with "lines" list and "slug"; voice_map — {"D": voice, "P": voice}
    Outputs: str path to saved MP3 file
    Dependencies: edge_tts; pydub for concatenation
    Side effects: Writes MP3 to corpus/clips/
    Failure modes: edge_tts.exceptions.NoAudioReceived if voice unavailable
    """
    from pydub import AudioSegment
    import tempfile

    _CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _CLIPS_DIR / f"clip_{script['id']:02d}_{script['slug']}.mp3"

    if out_path.exists():
        print(f"  Skipping (exists): {out_path.name}")
        return str(out_path)

    print(f"  Generating clip {script['id']:02d}: {script['scenario']}...")
    segments_audio = []

    with tempfile.TemporaryDirectory() as tmp:
        for idx, (speaker, text) in enumerate(script["lines"]):
            voice = voice_map[speaker]
            tmp_path = os.path.join(tmp, f"line_{idx:03d}.mp3")
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(tmp_path)
            seg = AudioSegment.from_mp3(tmp_path)
            # 400ms silence between speaker turns — natural pause in dialogue
            segments_audio.append(seg + AudioSegment.silent(duration=400))

    combined = sum(segments_audio, AudioSegment.empty())
    combined.export(str(out_path), format="mp3")
    print(f"    Saved: {out_path.name} ({combined.duration_seconds:.1f}s)")
    return str(out_path)


def _write_ground_truth(script: dict, voice_map: dict) -> str:
    """
    What it does: Writes a ground truth JSON file for a clip.
    Inputs: script — dict with all clip metadata; voice_map — {"D": voice, "P": voice}
    Outputs: str path to saved JSON file
    Dependencies: None
    Side effects: Writes JSON to corpus/clips/
    Failure modes: OSError if clips/ not writable
    """
    out_path = _CLIPS_DIR / f"clip_{script['id']:02d}_{script['slug']}_ground_truth.json"
    full_text = " ".join(text for _, text in script["lines"])
    word_count = len(full_text.split())

    data = {
        "clip_id": script["id"],
        "scenario": script["scenario"],
        "reference_transcript": full_text,
        "expected_entities": script["expected_entities"],
        "tts_voices_used": {
            "doctor": voice_map["D"],
            "patient": voice_map["P"],
        },
        "script_length_words": word_count,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return str(out_path)


async def _verify_voices() -> dict:
    """
    What it does: Lists available Edge TTS voices and verifies Hindi voices are present.
    Inputs: None
    Outputs: dict {"D": doctor_voice, "P": patient_voice} with verified or substitute voices
    Dependencies: edge_tts
    Side effects: Prints substitution warnings if voices missing
    Failure modes: edge_tts network error
    """
    voices = await edge_tts.list_voices()
    available = {v["ShortName"] for v in voices}

    doctor = _DOCTOR_VOICE
    patient = _PATIENT_VOICE

    if _DOCTOR_VOICE not in available:
        # Find any available Hindi male voice
        fallback = next((v["ShortName"] for v in voices if "hi-IN" in v["ShortName"] and v.get("Gender") == "Male"), None)
        if fallback:
            print(f"  WARNING: {_DOCTOR_VOICE} not available, using {fallback}")
            doctor = fallback
        else:
            raise RuntimeError(f"No Hindi male voice available. Available: {[v for v in available if 'hi' in v.lower()]}")

    if _PATIENT_VOICE not in available:
        fallback = next((v["ShortName"] for v in voices if "hi-IN" in v["ShortName"] and v.get("Gender") == "Female"), None)
        if fallback:
            print(f"  WARNING: {_PATIENT_VOICE} not available, using {fallback}")
            patient = fallback
        else:
            raise RuntimeError(f"No Hindi female voice available.")

    return {"D": doctor, "P": patient}


async def generate_all():
    """
    What it does: Generates all 10 clinical corpus clips and their ground truth JSON files.
    Inputs: None
    Outputs: None (files written to corpus/clips/)
    Dependencies: edge_tts; pydub; corpus/clips/ directory writable
    Side effects: Writes 20 files (10 MP3 + 10 JSON) to corpus/clips/
    Failure modes: RuntimeError if no Hindi voices available; OSError if disk full
    """
    print("Verifying Edge TTS voice availability...")
    voice_map = await _verify_voices()
    print(f"  Doctor voice : {voice_map['D']}")
    print(f"  Patient voice: {voice_map['P']}\n")

    tasks = [_generate_clip(s, voice_map) for s in SCRIPTS]
    await asyncio.gather(*tasks)

    print("\nWriting ground truth JSON files...")
    for script in SCRIPTS:
        path = _write_ground_truth(script, voice_map)
        print(f"  Wrote: {Path(path).name}")

    # Verify all 20 files exist
    mp3_files = list(_CLIPS_DIR.glob("clip_*.mp3"))
    json_files = list(_CLIPS_DIR.glob("clip_*_ground_truth.json"))
    print(f"\nVerification: {len(mp3_files)} MP3 files, {len(json_files)} JSON files")
    if len(mp3_files) < 10 or len(json_files) < 10:
        print("WARNING: Expected 10 MP3 + 10 JSON. Some files may be missing.")
    else:
        print("All 20 files verified.")

    print("\nFile list:")
    for f in sorted(_CLIPS_DIR.iterdir()):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name} ({size_kb} KB)")


if __name__ == "__main__":
    asyncio.run(generate_all())
