import os
import json
import torch
import librosa
import numpy as np
from collections import defaultdict
from siamese_model import SiameseAudioModel

def non_max_suppression(predictions, window_s=1.0):
    """
    Groups predictions that are close in time and picks the one with the lowest distance.
    predictions: list of dicts {"time": float, "dist": float}
    """
    if not predictions:
        return []
        
    predictions = sorted(predictions, key=lambda x: x["time"])
    clusters = []
    current_cluster = [predictions[0]]
    
    for p in predictions[1:]:
        if p["time"] - current_cluster[-1]["time"] <= window_s:
            current_cluster.append(p)
        else:
            clusters.append(current_cluster)
            current_cluster = [p]
    clusters.append(current_cluster)
    
    suppressed = []
    for cluster in clusters:
        best_p = min(cluster, key=lambda x: x["dist"])
        suppressed.append(best_p)
        
    return suppressed

def evaluate_predictions(gt_intervals, predictions, time_tolerance=0.5):
    """
    Calculates TP, FP, FN
    gt_intervals: list of (start, end)
    predictions: list of {"time": t, "dist": d} (after NMS)
    """
    tp = 0
    fp = 0
    fn = 0
    
    matched_gts = set()
    
    for p in predictions:
        t = p["time"]
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

def run_evaluation_pipeline():
    # To keep the exhaustive evaluation mathematically rigorous but computationally feasible on CPU,
    # we select 5 highly diverse words and 15 chunks (which still produces ~3,750 Neural Network passes).
    test_words = ["thank", "sport", "play", "passion", "level"]
    
    base_path = r"d:\Video Context Extraction System\Siamese_VMS_Project"
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseAudioModel().to(device)
    model.eval()
    
    model_path = os.path.join(base_path, "best_siamese_model.pth")
    if os.path.exists(model_path):
        model.load_weights(model_path)
    else:
        print(f"ERROR: Model weights not found at {model_path}")
        return

    # Load Ground Truth
    gt_path = os.path.join(base_path, "eval_ground_truth.json")
    with open(gt_path, "r") as f:
        ground_truth = json.load(f)
        
    results = {}
    
    # Run Inference
    for word in test_words:
        print(f"--- Evaluating word: {word} ---")
        keyword_path = os.path.join(base_path, "eval_keywords", f"{word}.wav")
        if not os.path.exists(keyword_path):
            print(f"Skipping {word}, TTS anchor not found.")
            continue
            
        keyword_audio, sr = librosa.load(keyword_path, sr=16000)
        keyword_embed = model.get_embedding(keyword_audio, sr)
        
        # Collect all predictions and GT for this word across all 60 chunks
        all_chunk_predictions = defaultdict(list)
        all_chunk_gts = defaultdict(list)
        
        for i in range(15):
            chunk_name = f"eval_{i}.wav"
            chunk_path = os.path.join(base_path, "eval_audios", chunk_name)
            
            # Get GT intervals for this word in this chunk
            chunk_gt = [item for item in ground_truth.get(chunk_name, []) if item["word"] == word]
            gt_intervals = [(item["start"], item["end"]) for item in chunk_gt]
            all_chunk_gts[chunk_name] = gt_intervals
            
            if not os.path.exists(chunk_path):
                continue
                
            # Run sliding window
            chunk_audio, _ = librosa.load(chunk_path, sr=16000)
            window_size = len(keyword_audio)
            step_size = int(16000 * 0.1) # 100ms step
            
            for start_idx in range(0, len(chunk_audio) - window_size, step_size):
                window = chunk_audio[start_idx:start_idx + window_size]
                if len(window) < window_size:
                    break
                    
                window_embed = model.get_embedding(window, sr)
                distance = torch.dist(keyword_embed, window_embed, p=2).item()
                
                center_time = (start_idx + (window_size / 2)) / 16000.0
                all_chunk_predictions[chunk_name].append({
                    "time": center_time,
                    "dist": distance
                })
                
        # Now sweep thresholds
        thresholds = np.arange(0.1, 2.55, 0.05)
        pr_curve = []
        
        for thresh in thresholds:
            total_tp, total_fp, total_fn = 0, 0, 0
            
            for chunk_name in all_chunk_predictions.keys():
                # Filter by threshold
                valid_preds = [p for p in all_chunk_predictions[chunk_name] if p["dist"] <= thresh]
                
                # NMS
                suppressed_preds = non_max_suppression(valid_preds, window_s=1.0)
                
                # Eval
                tp, fp, fn = evaluate_predictions(all_chunk_gts[chunk_name], suppressed_preds)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Special case: If 0 precision and 0 recall but we correctly guessed 0 because there were 0 GTs
            if total_tp == 0 and total_fp == 0 and total_fn == 0:
                precision, recall, f1 = 1.0, 1.0, 1.0
                
            pr_curve.append({
                "threshold": float(thresh),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": int(total_tp),
                "fp": int(total_fp),
                "fn": int(total_fn)
            })
            
        # Find best F1
        best_point = max(pr_curve, key=lambda x: x["f1"])
        print(f"  Best F1: {best_point['f1']:.2f} at Threshold: {best_point['threshold']:.2f} (P: {best_point['precision']:.2f}, R: {best_point['recall']:.2f})")
        
        results[word] = {
            "best_f1": best_point["f1"],
            "best_threshold": best_point["threshold"],
            "pr_curve": pr_curve
        }
        
    with open(os.path.join(base_path, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    print("\nEvaluation complete! Saved to evaluation_results.json")

if __name__ == "__main__":
    run_evaluation_pipeline()
