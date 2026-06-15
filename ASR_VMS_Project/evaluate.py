import os
import json
import librosa
from stream_pipeline import StreamPipeline
from keyword_matcher import KeywordMatcher

def evaluate_predictions(gt_intervals, predictions, time_tolerance=0.5):
    tp, fp, fn = 0, 0, 0
    matched_gts = set()
    
    for p in predictions:
        t = p["start"] # we can use start or center
        matched = False
        for i, (start, end) in enumerate(gt_intervals):
            if start - time_tolerance <= t <= end + time_tolerance:
                if i not in matched_gts:
                    tp += 1
                    matched_gts.add(i)
                    matched = True
                    break
        if not matched:
            fp += 1
            
    fn = len(gt_intervals) - len(matched_gts)
    return tp, fp, fn

def run_evaluation():
    print("ASR-Based Detector Evaluation")
    # This expects ground truth json from the previous project
    base_path = r"d:\Video Context Extraction System\Siamese_VMS_Project"
    gt_path = os.path.join(base_path, "eval_ground_truth.json")
    
    if not os.path.exists(gt_path):
        print(f"Ground truth not found at {gt_path}")
        return
        
    with open(gt_path, "r") as f:
        ground_truth = json.load(f)
        
    test_words = ["thank", "sport", "play", "passion", "level"]
    
    pipeline = StreamPipeline(test_words)
    
    # We will just evaluate F1 using the current CONF_THRESHOLD
    # A full sweep like Siamese is less necessary because we are word-based now
    
    results = {}
    
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for i in range(15):
        chunk_name = f"eval_{i}.wav"
        chunk_path = os.path.join(base_path, "eval_audios", chunk_name)
        if not os.path.exists(chunk_path):
            continue
            
        print(f"Evaluating {chunk_name}...")
        
        # We bypass the vad and orchestrator for pure metric eval
        words = pipeline.asr.transcribe_words(chunk_path)
        
        for word in test_words:
            # GT
            chunk_gt = [item for item in ground_truth.get(chunk_name, []) if item["word"] == word]
            gt_intervals = [(item["start"], item["end"]) for item in chunk_gt]
            
            # Predict
            matcher = KeywordMatcher([word], fuzzy_ratio=85.0, conf_threshold=0.5)
            preds = matcher.find(words)
            
            tp, fp, fn = evaluate_predictions(gt_intervals, preds)
            
            if word not in results:
                results[word] = {"tp": 0, "fp": 0, "fn": 0}
            results[word]["tp"] += tp
            results[word]["fp"] += fp
            results[word]["fn"] += fn
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print("\n--- FINAL EVALUATION RESULTS ---")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1 Score:  {f1:.2f}")
    
    for w, r in results.items():
        wtp, wfp, wfn = r["tp"], r["fp"], r["fn"]
        wp = wtp / (wtp + wfp) if (wtp + wfp) > 0 else 1.0
        wr = wtp / (wtp + wfn) if (wtp + wfn) > 0 else 0.0
        wf1 = 2 * (wp * wr) / (wp + wr) if (wp + wr) > 0 else 0.0
        print(f"Word '{w}': F1={wf1:.2f} (TP:{wtp}, FP:{wfp}, FN:{wfn})")

if __name__ == "__main__":
    run_evaluation()
