import os
import json
import matplotlib.pyplot as plt
import numpy as np

def visualize_results():
    base_path = r"d:\Video Context Extraction System\Siamese_VMS_Project"
    results_file = os.path.join(base_path, "evaluation_results.json")
    
    if not os.path.exists(results_file):
        print("Results file not found. Run eval_pipeline.py first.")
        return
        
    with open(results_file, "r") as f:
        results = json.load(f)
        
    # Sort words by length to see if length affects performance
    words = sorted(list(results.keys()), key=lambda x: len(x))
    f1_scores = [results[w]["best_f1"] for w in words]
    best_thresholds = [results[w]["best_threshold"] for w in words]

    # --- Plot 1: F1 Scores Bar Chart ---
    plt.figure(figsize=(12, 6))
    bars = plt.bar(words, f1_scores, color='skyblue')
    plt.axhline(y=0.8, color='g', linestyle='--', label='80% F1 Target')
    plt.axhline(y=0.5, color='r', linestyle='--', label='50% F1 Target')
    plt.title("Peak F1-Score per Keyword (Sorted by Word Length)")
    plt.xlabel("Keyword")
    plt.ylabel("Peak F1-Score")
    plt.ylim(0, 1.1)
    
    # Add threshold labels above bars
    for bar, t in zip(bars, best_thresholds):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"Thresh:\n{t:.2f}", ha='center', va='bottom', fontsize=9)
        
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, "f1_scores_bar.png"))
    plt.close()
    
    # --- Plot 2: PR Curves ---
    plt.figure(figsize=(10, 8))
    
    # Plot a few diverse words to avoid clutter (shortest, medium, longest)
    sample_words = [words[0], words[len(words)//2], words[-1]]
    colors = ['r', 'g', 'b']
    
    for word, color in zip(sample_words, colors):
        pr_curve = results[word]["pr_curve"]
        # Filter valid precision/recall values where there were actual GT matches
        valid_points = [p for p in pr_curve if (p["tp"] + p["fp"]) > 0 or (p["tp"] + p["fn"]) > 0]
        if not valid_points:
            continue
            
        recalls = [p["recall"] for p in valid_points]
        precisions = [p["precision"] for p in valid_points]
        
        # Sort by recall for plotting
        sort_idx = np.argsort(recalls)
        recalls = np.array(recalls)[sort_idx]
        precisions = np.array(precisions)[sort_idx]
        
        plt.plot(recalls, precisions, label=f"'{word}'", color=color, linewidth=2, marker='o', markersize=4)

    plt.title("Precision-Recall Curve (Selected Words)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, "pr_curves.png"))
    plt.close()
    
    print("Visualizations saved: f1_scores_bar.png, pr_curves.png")

if __name__ == "__main__":
    visualize_results()
