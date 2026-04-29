import asyncio
import json
import os
import time
from faster_whisper import WhisperModel
from jiwer import wer
import edge_tts

# ─────────────────────────────────────────────────────────
# CORPUS — 6 matched clinical scenarios in both languages
# Each scenario has: the TTS text, reference transcript,
# and expected clinical entities for context
# ─────────────────────────────────────────────────────────

CORPUS = [
    {
        "id": 1,
        "scenario": "Fever and headache",
        "hindi": {
            "tts_text": "मुझे तीन दिनों से बुखार है और सिर में दर्द भी है। कोई दवाई नहीं ली अभी तक। डॉक्टर, तापमान 102 डिग्री है। पानी पीते रहें और पैरासिटामोल लें।",
            "reference": "मुझे तीन दिनों से बुखार है और सिर में दर्द भी है कोई दवाई नहीं ली अभी तक डॉक्टर तापमान 102 डिग्री है पानी पीते रहें और पैरासिटामोल लें",
            "voice": "hi-IN-SwaraNeural",
        },
        "swahili": {
            "tts_text": "Nimekuwa na homa kwa siku tatu na maumivu ya kichwa pia. Sijachukua dawa yoyote bado. Daktari, joto ni digrii 102. Endelea kunywa maji na uchukue paracetamol.",
            "reference": "nimekuwa na homa kwa siku tatu na maumivu ya kichwa pia sijachukua dawa yoyote bado daktari joto ni digrii 102 endelea kunywa maji na uchukue paracetamol",
            "voice": "sw-KE-ZuriNeural",
        },
    },
    {
        "id": 2,
        "scenario": "Chest pain and breathlessness",
        "hindi": {
            "tts_text": "मुझे सीने में दर्द हो रहा है और सांस लेने में तकलीफ है। यह दो घंटे से हो रहा है। पहले से blood pressure की बीमारी है। यह बहुत गंभीर लक्षण हैं, अभी ECG करते हैं।",
            "reference": "मुझे सीने में दर्द हो रहा है और सांस लेने में तकलीफ है यह दो घंटे से हो रहा है पहले से blood pressure की बीमारी है यह बहुत गंभीर लक्षण हैं अभी ECG करते हैं",
            "voice": "hi-IN-MadhurNeural",
        },
        "swahili": {
            "tts_text": "Nina maumivu ya kifua na ugumu wa kupumua. Hii imekuwa ikitokea kwa masaa mawili. Nina shinikizo la damu tayari. Hizi ni dalili nzito sana, tutafanya ECG sasa hivi.",
            "reference": "nina maumivu ya kifua na ugumu wa kupumua hii imekuwa ikitokea kwa masaa mawili nina shinikizo la damu tayari hizi ni dalili nzito sana tutafanya ECG sasa hivi",
            "voice": "sw-KE-RafikiNeural",
        },
    },
    {
        "id": 3,
        "scenario": "Diabetes follow-up",
        "hindi": {
            "tts_text": "मेरी sugar की जांच करवाई। fasting blood sugar 180 आई है। क्या आप metformin ले रहे हैं? हाँ, रोज़ सुबह लेता हूँ। खाने में चीनी और चावल कम करें।",
            "reference": "मेरी sugar की जांच करवाई fasting blood sugar 180 आई है क्या आप metformin ले रहे हैं हाँ रोज़ सुबह लेता हूँ खाने में चीनी और चावल कम करें",
            "voice": "hi-IN-SwaraNeural",
        },
        "swahili": {
            "tts_text": "Nimefanya uchunguzi wa sukari yangu. Sukari ya damu ya kufunga ilikuwa 180. Je, unachukua metformin? Ndiyo, kila asubuhi. Punguza sukari na mchele katika chakula chako.",
            "reference": "nimefanya uchunguzi wa sukari yangu sukari ya damu ya kufunga ilikuwa 180 je unachukua metformin ndiyo kila asubuhi punguza sukari na mchele katika chakula chako",
            "voice": "sw-KE-ZuriNeural",
        },
    },
    {
        "id": 4,
        "scenario": "Cough and TB concern",
        "hindi": {
            "tts_text": "दो हफ्तों से खांसी है। कभी-कभी बलगम में खून भी आता है। क्या आप किसी TB के मरीज़ के संपर्क में आए? हाँ, मेरे पड़ोसी को TB है। सीने का X-ray और sputum test करना होगा।",
            "reference": "दो हफ्तों से खांसी है कभी-कभी बलगम में खून भी आता है क्या आप किसी TB के मरीज़ के संपर्क में आए हाँ मेरे पड़ोसी को TB है सीने का X-ray और sputum test करना होगा",
            "voice": "hi-IN-MadhurNeural",
        },
        "swahili": {
            "tts_text": "Nina kikohozi kwa wiki mbili. Wakati mwingine kuna damu katika makohozi. Je, umegusana na mgonjwa wa TB? Ndiyo, jirani yangu ana TB. Tutahitaji X-ray ya kifua na mtihani wa makohozi.",
            "reference": "nina kikohozi kwa wiki mbili wakati mwingine kuna damu katika makohozi je umegusana na mgonjwa wa TB ndiyo jirani yangu ana TB tutahitaji X-ray ya kifua na mtihani wa makohozi",
            "voice": "sw-KE-RafikiNeural",
        },
    },
    {
        "id": 5,
        "scenario": "Stomach pain and vomiting",
        "hindi": {
            "tts_text": "पेट में बहुत दर्द है और उल्टी भी हो रही है। कल रात से दस्त भी हैं। क्या बाहर का खाना खाया था? हाँ, कल शाम को। यह food poisoning लग रही है। ORS पीते रहें।",
            "reference": "पेट में बहुत दर्द है और उल्टी भी हो रही है कल रात से दस्त भी हैं क्या बाहर का खाना खाया था हाँ कल शाम को यह food poisoning लग रही है ORS पीते रहें",
            "voice": "hi-IN-SwaraNeural",
        },
        "swahili": {
            "tts_text": "Nina maumivu makali ya tumbo na pia natapika. Nimekuwa na kuhara tangu usiku wa jana. Je, ulikula chakula nje? Ndiyo, jana jioni. Inaonekana ni sumu ya chakula. Endelea kunywa ORS.",
            "reference": "nina maumivu makali ya tumbo na pia natapika nimekuwa na kuhara tangu usiku wa jana je ulikula chakula nje ndiyo jana jioni inaonekana ni sumu ya chakula endelea kunywa ORS",
            "voice": "sw-KE-ZuriNeural",
        },
    },
    {
        "id": 6,
        "scenario": "Joint pain",
        "hindi": {
            "tts_text": "दोनों घुटनों में दर्द है। सीढ़ियाँ चढ़ने में बहुत तकलीफ होती है। यह कितने समय से है? तीन महीने से। उम्र के साथ arthritis हो सकता है। X-ray और blood test करेंगे।",
            "reference": "दोनों घुटनों में दर्द है सीढ़ियाँ चढ़ने में बहुत तकलीफ होती है यह कितने समय से है तीन महीने से उम्र के साथ arthritis हो सकता है X-ray और blood test करेंगे",
            "voice": "hi-IN-MadhurNeural",
        },
        "swahili": {
            "tts_text": "Nina maumivu katika magoti yote mawili. Ni vigumu sana kupanda ngazi. Hii imekuwa kwa muda gani? Kwa miezi mitatu. Arthritis inaweza kutokea na umri. Tutafanya X-ray na vipimo vya damu.",
            "reference": "nina maumivu katika magoti yote mawili ni vigumu sana kupanda ngazi hii imekuwa kwa muda gani kwa miezi mitatu arthritis inaweza kutokea na umri tutafanya X-ray na vipimo vya damu",
            "voice": "sw-KE-RafikiNeural",
        },
    },
]


# ─────────────────────────────────────────────────────────
# STEP 1 — Generate audio clips using Edge TTS
# ─────────────────────────────────────────────────────────

async def generate_audio(text: str, voice: str, output_path: str):
    """Generate a single audio clip using Edge TTS."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


async def generate_all_audio():
    """Generate all Hindi and Swahili clips."""
    os.makedirs("benchmark_audio", exist_ok=True)
    tasks = []

    for item in CORPUS:
        for lang in ["hindi", "swahili"]:
            path = f"benchmark_audio/{lang}_{item['id']}.mp3"
            if not os.path.exists(path):
                tasks.append(
                    generate_audio(item[lang]["tts_text"], item[lang]["voice"], path)
                )
                print(f"  Queued: {lang} clip {item['id']} — {item['scenario']}")
            else:
                print(f"  Skipping (exists): {path}")

    if tasks:
        print(f"\nGenerating {len(tasks)} audio clips...")
        await asyncio.gather(*tasks)
        print("Audio generation complete.\n")
    else:
        print("All audio clips already exist, skipping generation.\n")


# ─────────────────────────────────────────────────────────
# STEP 2 — Transcribe with Whisper large-v3
# ─────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """
    Normalize text for WER comparison.
    Strips punctuation and lowercases — both reference and hypothesis
    must be normalized the same way for fair comparison.
    """
    import re
    text = text.lower().strip()
    # Remove punctuation but preserve Devanagari characters
    text = re.sub(r'[।,\.!?;:\-\(\)\[\]"\']+', '', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def transcribe_clip(model, audio_path: str, language: str) -> dict:
    """
    Transcribe a single audio clip and return text + latency.
    Language is hardcoded per clip — 'hi' for Hindi, 'sw' for Swahili.
    """
    start = time.time()

    # Clinical domain prompt biases the decoder toward medical vocabulary
    # Different prompt per language since they use different terminology
    prompts = {
        "hi": "यह एक डॉक्टर और मरीज़ के बीच की बातचीत है। मरीज़ के लक्षण, दवाइयाँ, और निदान।",
        "sw": "Hii ni mazungumzo kati ya daktari na mgonjwa. Dalili za mgonjwa, dawa, na utambuzi.",
    }

    raw_segments, _ = model.transcribe(
        audio_path,
        language=language,
        initial_prompt=prompts[language],
        word_timestamps=False,
        beam_size=5,
    )
    text = " ".join(seg.text.strip() for seg in raw_segments)

    latency = time.time() - start
    return {
        "text": text,
        "latency_seconds": round(latency, 2),
    }


# ─────────────────────────────────────────────────────────
# STEP 3 — Run full benchmark
# ─────────────────────────────────────────────────────────

def run_benchmark():
    print("=" * 60)
    print("ClinScribe — Whisper Benchmark: Hindi vs Swahili")
    print("Model: large-v3  |  Task: transcribe (no translation)")
    print("=" * 60)

    print("\nLoading Whisper large-v3 (int8)...")
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    print("Model loaded.\n")

    results = []
    hindi_wers = []
    swahili_wers = []
    hindi_latencies = []
    swahili_latencies = []

    for item in CORPUS:
        scenario_results = {"id": item["id"], "scenario": item["scenario"]}

        for lang, lang_code in [("hindi", "hi"), ("swahili", "sw")]:
            audio_path = f"benchmark_audio/{lang}_{item['id']}.mp3"
            reference = normalize(item[lang]["reference"])

            print(f"Transcribing [{lang.upper()}] Clip {item['id']}: {item['scenario']}...")
            output = transcribe_clip(model, audio_path, lang_code)

            hypothesis = normalize(output["text"])
            clip_wer = round(wer(reference, hypothesis) * 100, 1)

            print(f"  Reference : {reference[:80]}...")
            print(f"  Hypothesis: {hypothesis[:80]}...")
            print(f"  WER: {clip_wer}%  |  Latency: {output['latency_seconds']}s\n")

            scenario_results[lang] = {
                "reference": reference,
                "hypothesis": hypothesis,
                "wer_percent": clip_wer,
                "latency_seconds": output["latency_seconds"],
            }

            if lang == "hindi":
                hindi_wers.append(clip_wer)
                hindi_latencies.append(output["latency_seconds"])
            else:
                swahili_wers.append(clip_wer)
                swahili_latencies.append(output["latency_seconds"])

        results.append(scenario_results)

    # ─────────────────────────────────────────────────────
    # STEP 4 — Compute summary statistics
    # ─────────────────────────────────────────────────────

    avg_hindi_wer = round(sum(hindi_wers) / len(hindi_wers), 1)
    avg_swahili_wer = round(sum(swahili_wers) / len(swahili_wers), 1)
    avg_hindi_lat = round(sum(hindi_latencies) / len(hindi_latencies), 1)
    avg_swahili_lat = round(sum(swahili_latencies) / len(swahili_latencies), 1)

    summary = {
        "model": "large-v3",
        "clips_per_language": len(CORPUS),
        "hindi": {
            "wer_per_clip": hindi_wers,
            "average_wer_percent": avg_hindi_wer,
            "average_latency_seconds": avg_hindi_lat,
            "clinical_usability": "YES" if avg_hindi_wer < 20 else "MARGINAL" if avg_hindi_wer < 35 else "NO",
        },
        "swahili": {
            "wer_per_clip": swahili_wers,
            "average_wer_percent": avg_swahili_wer,
            "average_latency_seconds": avg_swahili_lat,
            "clinical_usability": "YES" if avg_swahili_wer < 20 else "MARGINAL" if avg_swahili_wer < 35 else "NO",
        },
        "verdict": "Hindi" if avg_hindi_wer < avg_swahili_wer else "Swahili",
        "wer_difference_percent": round(abs(avg_hindi_wer - avg_swahili_wer), 1),
    }

    full_results = {"summary": summary, "clips": results}

    # Save JSON
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)

    # ─────────────────────────────────────────────────────
    # STEP 5 — Print human-readable report
    # ─────────────────────────────────────────────────────

    report = []
    report.append("\n" + "=" * 60)
    report.append("BENCHMARK RESULTS SUMMARY")
    report.append("Whisper large-v3 | Clinical Hindi vs Swahili")
    report.append("=" * 60)
    report.append(f"\n{'Scenario':<30} {'Hindi WER':>12} {'Swahili WER':>14} {'Winner':>10}")
    report.append("-" * 70)

    for item, hw, sw in zip(CORPUS, hindi_wers, swahili_wers):
        winner = "Hindi" if hw < sw else "Swahili" if sw < hw else "Tie"
        report.append(f"{item['scenario']:<30} {hw:>10.1f}%  {sw:>12.1f}%  {winner:>10}")

    report.append("-" * 70)
    report.append(f"{'AVERAGE':<30} {avg_hindi_wer:>10.1f}%  {avg_swahili_wer:>12.1f}%")
    report.append(f"{'Avg latency':<30} {avg_hindi_lat:>9.1f}s  {avg_swahili_lat:>11.1f}s")

    report.append("\n" + "=" * 60)
    report.append("CLINICAL USABILITY THRESHOLD: WER < 20%")
    report.append("=" * 60)
    report.append(f"Hindi   → {avg_hindi_wer}% WER  →  {summary['hindi']['clinical_usability']}")
    report.append(f"Swahili → {avg_swahili_wer}% WER  →  {summary['swahili']['clinical_usability']}")

    report.append("\n" + "=" * 60)
    report.append(f"RECOMMENDATION: Use {summary['verdict']}")
    report.append(f"WER gap: {summary['wer_difference_percent']} percentage points")
    report.append("=" * 60)

    report.append("\nINTERPRETATION:")
    report.append("  <10% WER  → Excellent (comparable to human transcription)")
    report.append("  10-20% WER → Good (usable for clinical notes with review)")
    report.append("  20-35% WER → Marginal (significant errors, risky for clinical use)")
    report.append("  >35% WER  → Poor (not suitable for clinical documentation)")

    report.append("\nNOTE: These results are on synthetic TTS audio (Edge TTS).")
    report.append("Real clinical audio will have higher WER due to:")
    report.append("  - Background noise in clinics")
    report.append("  - Natural speech disfluencies (um, uh, pauses)")
    report.append("  - Stronger regional accents")
    report.append("  - Code-switching mid-sentence")
    report.append("Expect real-world WER to be 5-15% higher than these numbers.")

    full_report = "\n".join(report)
    print(full_report)

    with open("benchmark_report.txt", "w", encoding="utf-8") as f:
        f.write(full_report)

    print(f"\nResults saved to:")
    print(f"  benchmark_results.json  (full data)")
    print(f"  benchmark_report.txt    (human-readable report)")

    return full_results


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Step 1: Generating audio clips with Edge TTS...")
    asyncio.run(generate_all_audio())

    print("Step 2: Running Whisper transcription benchmark...")
    run_benchmark()