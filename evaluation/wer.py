"""
ClinScribe WER evaluation.
Runs Whisper on all 10 corpus clips and computes Word Error Rate against ground truth.
"""

import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


def normalize(text: str) -> str:
    """
    What it does: Normalizes Hindi/English text for fair WER comparison.
    Inputs: text — raw transcript string
    Outputs: normalized lowercase string with punctuation and danda removed
    Dependencies: None
    Side effects: None
    Failure modes: None
    """
    text = text.lower().strip()
    # Remove danda (।), standard punctuation — these are not spoken words
    text = re.sub(r'[।,\.!?;:\-\(\)\[\]"\']+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def run_wer_evaluation():
    """
    What it does: Transcribes all 10 corpus clips and computes per-clip and average WER.
    Inputs: None (reads from corpus/clips/)
    Outputs: Saves evaluation/results/wer_results.json and prints table
    Dependencies: Whisper (loaded from pipeline.transcribe); jiwer; corpus clips must exist
    Side effects: Writes evaluation/results/wer_results.json
    Failure modes: FileNotFoundError if clips missing; RuntimeError on GPU OOM
    """
    from jiwer import wer as compute_wer
    from pipeline.transcribe import transcribe

    clips_dir = _ROOT / "corpus" / "clips"
    results_dir = _ROOT / "evaluation" / "results"
    results_dir.mkdir(exist_ok=True)

    gt_files = sorted(clips_dir.glob("clip_*_ground_truth.json"))
    if not gt_files:
        print("ERROR: No ground truth files found in corpus/clips/")
        print("Run: python corpus/generate_corpus.py first")
        sys.exit(1)

    print("=" * 70)
    print("ClinScribe WER Evaluation — Whisper large-v3 on Hindi corpus")
    print("=" * 70)
    print(f"\n{'Clip':<6} {'Scenario':<35} {'WER':>6}  {'Status'}")
    print("-" * 70)

    per_clip = []
    wers = []

    for gt_path in gt_files:
        with open(gt_path, encoding="utf-8") as f:
            gt = json.load(f)

        clip_id = gt["clip_id"]
        scenario = gt["scenario"]
        reference = normalize(gt["reference_transcript"])

        mp3_name = gt_path.name.replace("_ground_truth.json", ".mp3")
        mp3_path = clips_dir / mp3_name
        if not mp3_path.exists():
            print(f"  {clip_id:02d}   {scenario:<35} SKIP (no MP3)")
            continue

        try:
            result = transcribe(str(mp3_path))
            hypothesis = normalize(result["full_text"])
            clip_wer = round(compute_wer(reference, hypothesis), 4)
            wers.append(clip_wer)
            status = "PASS" if clip_wer < 0.20 else "FAIL"
            print(f"  {clip_id:02d}   {scenario:<35} {clip_wer*100:5.1f}%  {status}")
            per_clip.append({
                "clip_id": clip_id,
                "scenario": scenario,
                "wer": clip_wer,
                "wer_percent": round(clip_wer * 100, 1),
                "reference": reference,
                "hypothesis": hypothesis,
            })
        except Exception as e:
            print(f"  {clip_id:02d}   {scenario:<35} ERROR: {e}")

    if not wers:
        print("\nNo results computed.")
        return

    avg_wer = sum(wers) / len(wers)
    print("-" * 70)
    print(f"  {'AVERAGE':<40} {avg_wer*100:5.1f}%")
    print()
    print(f"Clinical usability threshold: WER < 20%")
    print(f"Result: {'PASS' if avg_wer < 0.20 else 'FAIL'} (avg WER = {avg_wer*100:.1f}%)")
    print("=" * 70)

    output = {
        "run_datetime": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "model": "large-v3",
        "language": "hi",
        "clips_evaluated": len(per_clip),
        "average_wer": round(avg_wer, 4),
        "average_wer_percent": round(avg_wer * 100, 1),
        "clinical_usability_pass": avg_wer < 0.20,
        "per_clip": per_clip,
    }

    out_path = results_dir / "wer_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    run_wer_evaluation()
