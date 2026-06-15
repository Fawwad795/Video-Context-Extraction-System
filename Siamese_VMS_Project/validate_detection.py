"""Validate detector output against Whisper transcripts (chunk level).

Ground truth: each live chunk is transcribed with whisper-tiny; a chunk is
a true positive target iff the keyword appears as a word in its transcript.
Predictions: chunks with at least one detection in logs/detections_<kw>.json
(produced by detector.py).

Prints a per-chunk table and precision / recall / F1.
"""

import argparse
import json
import os
import re
import warnings

from scoring import PROJECT_ROOT, list_chunk_audios

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser(description="Validate detections vs Whisper transcripts.")
    ap.add_argument("--keyword", default=None, help="defaults to selected_keyword.txt")
    args = ap.parse_args()

    keyword = args.keyword
    if keyword is None:
        kw_file = os.path.join(PROJECT_ROOT, "selected_keyword.txt")
        if not os.path.exists(kw_file):
            print("No --keyword given and selected_keyword.txt not found.")
            return
        keyword = open(kw_file).read().strip()
    keyword = keyword.lower()

    json_path = os.path.join(PROJECT_ROOT, "logs", f"detections_{keyword}.json")
    if not os.path.exists(json_path):
        print(f"{json_path} not found - run detector.py first.")
        return
    with open(json_path) as f:
        results = json.load(f)
    predicted = {c["file"]: bool(c.get("detections")) for c in results["chunks"]}
    best_score = {c["file"]: c.get("best_score") for c in results["chunks"]}

    print("Loading whisper-tiny for ground-truth transcription...")
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

    audio_files = list_chunk_audios()
    tp = fp = tn = fn = 0
    print(f"\n{'chunk':<14}{'truth':<7}{'pred':<7}{'best':<8}transcript")
    print("-" * 80)
    for audio_file in audio_files:
        name = os.path.basename(audio_file)
        text = asr(audio_file)["text"].lower()
        tokens = set(re.findall(r"[a-z']+", text))
        truth = keyword in tokens
        pred = predicted.get(name, False)
        if truth and pred:
            tp += 1
        elif truth and not pred:
            fn += 1
        elif not truth and pred:
            fp += 1
        else:
            tn += 1
        score = best_score.get(name)
        score_str = f"{score:.2f}" if score is not None else "-"
        snippet = (text[:45] + "...") if len(text) > 48 else text
        print(f"{name:<14}{str(truth):<7}{str(pred):<7}{score_str:<8}{snippet}")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print("-" * 80)
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"precision={precision:.2f}  recall={recall:.2f}  F1={f1:.2f}")
    if tp + fn == 0:
        print(f"NOTE: Whisper found no chunk containing '{keyword}' - "
              f"metrics only reflect false-alarm behaviour on this batch.")


if __name__ == "__main__":
    main()
