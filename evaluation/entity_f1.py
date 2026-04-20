"""
ClinScribe entity extraction F1 evaluation.
Runs full pipeline on all 10 corpus clips and computes precision/recall/F1
for symptoms, diagnoses, and medications using Jaccard token overlap.
"""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


def jaccard_match(predicted: str, reference: str, threshold: float = 0.5) -> bool:
    """
    What it does: Computes Jaccard token overlap between two strings.
    Inputs: predicted/reference — str; threshold — minimum overlap to count as match
    Outputs: True if |intersection| / |union| >= threshold
    Dependencies: None
    Side effects: None
    Failure modes: None
    """
    pred_tokens = set(predicted.lower().split())
    ref_tokens = set(reference.lower().split())
    if not pred_tokens and not ref_tokens:
        return True
    if not pred_tokens or not ref_tokens:
        return False
    intersection = len(pred_tokens & ref_tokens)
    union = len(pred_tokens | ref_tokens)
    return (intersection / union) >= threshold


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict:
    """
    What it does: Computes precision, recall, and F1 from TP/FP/FN counts.
    Inputs: tp/fp/fn — int counts
    Outputs: dict with precision, recall, f1 as floats rounded to 3 decimal places
    Dependencies: None
    Side effects: None
    Failure modes: None
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp, "fp": fp, "fn": fn,
    }


def evaluate_field(predicted_list: list, reference_list: list) -> dict:
    """
    What it does: Computes TP/FP/FN for one entity field using Jaccard matching.
    Inputs: predicted_list/reference_list — lists of str entity values
    Outputs: dict from precision_recall_f1()
    Dependencies: jaccard_match()
    Side effects: None
    Failure modes: None
    """
    # Coerce to plain strings (handle hallucination-flagged dicts)
    def to_str(item):
        return item.get("text", str(item)) if isinstance(item, dict) else str(item)

    preds = [to_str(x) for x in (predicted_list or [])]
    refs = [to_str(x) for x in (reference_list or [])]

    matched_refs = set()
    tp = 0
    for pred in preds:
        for i, ref in enumerate(refs):
            if i not in matched_refs and jaccard_match(pred, ref):
                tp += 1
                matched_refs.add(i)
                break

    fp = len(preds) - tp
    fn = len(refs) - len(matched_refs)
    return precision_recall_f1(tp, fp, fn)


def run_entity_f1_evaluation():
    """
    What it does: Runs transcribe+extract on all 10 corpus clips and evaluates entity F1.
    Inputs: None (reads from corpus/clips/)
    Outputs: Saves evaluation/results/entity_f1.json and prints table
    Dependencies: pipeline.transcribe, pipeline.extract; corpus clips must exist
    Side effects: Writes evaluation/results/entity_f1.json; LLM API calls
    Failure modes: FileNotFoundError if clips missing; anthropic.APIError on key failure
    """
    from pipeline.transcribe import transcribe
    from pipeline.extract import extract

    clips_dir = _ROOT / "corpus" / "clips"
    results_dir = _ROOT / "evaluation" / "results"
    results_dir.mkdir(exist_ok=True)

    gt_files = sorted(clips_dir.glob("clip_*_ground_truth.json"))
    if not gt_files:
        print("ERROR: No ground truth files found. Run corpus/generate_corpus.py first.")
        sys.exit(1)

    print("=" * 70)
    print("ClinScribe Entity Extraction F1 Evaluation")
    print("=" * 70)

    all_symptoms_metrics = []
    all_diag_metrics = []
    all_meds_metrics = []
    per_clip_results = []

    for gt_path in gt_files:
        with open(gt_path, encoding="utf-8") as f:
            gt = json.load(f)

        clip_id = gt["clip_id"]
        scenario = gt["scenario"]
        expected = gt["expected_entities"]

        mp3_name = gt_path.name.replace("_ground_truth.json", ".mp3")
        mp3_path = clips_dir / mp3_name
        if not mp3_path.exists():
            print(f"  Clip {clip_id:02d} {scenario}: SKIP (no MP3)")
            continue

        print(f"\n  Clip {clip_id:02d}: {scenario}")
        try:
            transcription = transcribe(str(mp3_path))
            entities = extract(transcription["full_text"])
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        symp_m = evaluate_field(entities.get("symptoms", []), expected.get("symptoms", []))
        diag_m = evaluate_field(entities.get("diagnosis", []), expected.get("diagnosis", []))
        meds_m = evaluate_field(entities.get("medications_mentioned", []), expected.get("medications_mentioned", []))

        print(f"    Symptoms  — P={symp_m['precision']:.2f} R={symp_m['recall']:.2f} F1={symp_m['f1']:.2f}")
        print(f"    Diagnoses — P={diag_m['precision']:.2f} R={diag_m['recall']:.2f} F1={diag_m['f1']:.2f}")
        print(f"    Meds      — P={meds_m['precision']:.2f} R={meds_m['recall']:.2f} F1={meds_m['f1']:.2f}")

        all_symptoms_metrics.append(symp_m)
        all_diag_metrics.append(diag_m)
        all_meds_metrics.append(meds_m)

        per_clip_results.append({
            "clip_id": clip_id,
            "scenario": scenario,
            "symptoms": symp_m,
            "diagnosis": diag_m,
            "medications": meds_m,
            "predicted_entities": {
                "symptoms": entities.get("symptoms", []),
                "diagnosis": entities.get("diagnosis", []),
                "medications_mentioned": entities.get("medications_mentioned", []),
            },
            "expected_entities": {
                "symptoms": expected.get("symptoms", []),
                "diagnosis": expected.get("diagnosis", []),
                "medications_mentioned": expected.get("medications_mentioned", []),
            },
        })

    def avg_metric(metrics: list, key: str) -> float:
        return round(sum(m[key] for m in metrics) / len(metrics), 3) if metrics else 0.0

    print("\n" + "=" * 70)
    print("OVERALL AVERAGES")
    print("-" * 70)
    print(f"  {'Category':<15} {'Precision':>10} {'Recall':>8} {'F1':>6}")
    print(f"  {'Symptoms':<15} {avg_metric(all_symptoms_metrics, 'precision'):>10.3f} {avg_metric(all_symptoms_metrics, 'recall'):>8.3f} {avg_metric(all_symptoms_metrics, 'f1'):>6.3f}")
    print(f"  {'Diagnoses':<15} {avg_metric(all_diag_metrics, 'precision'):>10.3f} {avg_metric(all_diag_metrics, 'recall'):>8.3f} {avg_metric(all_diag_metrics, 'f1'):>6.3f}")
    print(f"  {'Medications':<15} {avg_metric(all_meds_metrics, 'precision'):>10.3f} {avg_metric(all_meds_metrics, 'recall'):>8.3f} {avg_metric(all_meds_metrics, 'f1'):>6.3f}")
    print("=" * 70)

    output = {
        "run_datetime": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "clips_evaluated": len(per_clip_results),
        "averages": {
            "symptoms": {k: avg_metric(all_symptoms_metrics, k) for k in ("precision", "recall", "f1")},
            "diagnosis": {k: avg_metric(all_diag_metrics, k) for k in ("precision", "recall", "f1")},
            "medications": {k: avg_metric(all_meds_metrics, k) for k in ("precision", "recall", "f1")},
        },
        "per_clip": per_clip_results,
    }

    out_path = results_dir / "entity_f1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    run_entity_f1_evaluation()
