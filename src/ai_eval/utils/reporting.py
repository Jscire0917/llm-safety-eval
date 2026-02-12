import matplotlib
matplotlib.use('agg') 
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any, Dict

# This function recursively converts objects to JSON-serializable formats, particularly handling NumPy types which are common in evaluation metrics. It ensures that all data can be saved to a JSON file without serialization issues.
def convert_for_json(obj: Any) -> Any:
    """
    Recursively convert NumPy types and other non-serializable objects to JSON-friendly formats.
    """
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return tuple(convert_for_json(item) for item in obj)
    if isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    return obj
# This function saves the evaluation results to a JSON file. It ensures that all data types are converted to JSON-serializable formats, particularly handling NumPy types which are common in evaluation metrics.
# The results are saved in a specified output directory with a given filename.
def save_results(results: Dict[str, Any], output_dir: str = "evaluation_results", filename: str = "eval_results.json"):
    Path(output_dir).mkdir(exist_ok=True)
    path = Path(output_dir) / filename

    # Convert all NumPy types before saving
    clean_results = convert_for_json(results)

    with open(path, 'w') as f:
        json.dump(clean_results, f, indent=2)

    print(f"Results saved to {path}")
    
# This function generates bar charts for the evaluation scores, including confidence intervals where available. It creates two plots: one with the full scale and another zoomed in for small scores (like toxicity).
# The function handles cases where some scores may not have confidence intervals and ensures that the visualizations are informative and clear.
def generate_visual_report(results: Dict[str, Any], output_dir: str = "evaluation_results"):
    Path(output_dir).mkdir(exist_ok=True)

    print(list(results.keys()))

    df_rows = []

    for eval_name, data in results.items():
        if not isinstance(data, dict):
            continue

        print(f"{eval_name}")
        print(f"  Data keys: {list(data.keys())}")

        row = {"Evaluator": eval_name.replace("Evaluator", ""), "Score": None, "CI Low": None, "CI High": None, "Type": None}
        
        if "bias_score" in data:
            row["Score"] = data["bias_score"]
            ci = data.get("bias_ci_95", [None, None])
            row["CI Low"] = ci[0] if len(ci) > 0 else None
            row["CI High"] = ci[1] if len(ci) > 1 else None
            row["Type"] = "Bias"

        if "toxicity_score" in data:
            row["Score"] = data["toxicity_score"]
            ci = data.get("toxicity_ci_95", [None, None])
            row["CI Low"] = ci[0] if len(ci) > 0 else None
            row["CI High"] = ci[1] if len(ci) > 1 else None
            row["Type"] = "Toxicity"

        if "hallucination_score" in data:
            row["Score"] = data["hallucination_score"]
            row["CI Low"] = None
            row["CI High"] = None
            row["Type"] = "Hallucination"

        if row["Score"] is not None:
            df_rows.append(row)

    if not df_rows:
        print("[VISUAL] No plottable scores found after processing")
        return

    df = pd.DataFrame(df_rows)
    print(df.to_string())

    # Dynamic y-limit: zoom in if max score is small
    max_score = df["Score"].max()
    ylim_upper = max(max_score * 1.2, 0.01) if max_score > 0 else 1.0

# Main plot (full scale)
    plt.figure(figsize=(10, 6))
    yerr = None
    if "CI Low" in df.columns and "CI High" in df.columns:
        yerr_low = df["Score"] - df["CI Low"].fillna(df["Score"])
        yerr_high = df["CI High"].fillna(df["Score"]) - df["Score"]
        yerr = [yerr_low, yerr_high]

# Add value labels on top of bars
    for i, v in enumerate(df["Score"]):
        plt.text(i, v + 0.005 * ylim_upper, f"{v:.6f}", ha='center', va='bottom', fontsize=9)
    plt.bar(df["Evaluator"], df["Score"], yerr=yerr, capsize=5, alpha=0.7, color='skyblue', error_kw={'ecolor':'black', 'lw':1.5})
    plt.title("LLM Evaluation Scores (with 95% CI where available)")
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, ylim_upper)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "scores_bar_full.png")
    plt.close()

# Zoomed plot for tiny scores
    plt.figure(figsize=(10, 6))
    plt.bar(df["Evaluator"], df["Score"], yerr=yerr, capsize=5, alpha=0.7, color='lightgreen')
    plt.title("Zoomed View: Small Scores")
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 0.002)  # zoom on toxicity range
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "scores_bar_zoomed.png")
    plt.close()

    print(f"Visualizations saved: {output_dir}/scores_bar_full.png and scores_bar_zoomed.png")