import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS_DIR = "./outputs/results"
OUTPUT_DIR  = "./outputs/day12_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = {
    "resnet50": {"name": "ResNet50",   "color": "#2E75B6"},
    "deit":     {"name": "DeiT-Small", "color": "#ED7D31"},
    "rad_dino": {"name": "RAD-DINO",   "color": "#70AD47"},
}
METRICS = ["accuracy", "f1", "precision", "recall", "auc", "mcc"]
METRIC_LABELS = {
    "accuracy":  "Accuracy",
    "f1":  "F1-Score",
    "precision": "Precision",
    "recall":  "Recall",
    "auc": "AUC",
    "mcc": "MCC",
}


#  LOAD RESULTS 
def load_results():
   
    results = {}

    for model_key in MODELS:
        seed42_path  = os.path.join(RESULTS_DIR, f"{model_key}_seed42.json")
        seed123_path = os.path.join(RESULTS_DIR, f"{model_key}_seed123.json")

        missing = [p for p in [seed42_path, seed123_path] if not os.path.exists(p)]
        if missing:
            print(f" Missing files for {model_key}: {missing}")
            continue

        r42  = json.load(open(seed42_path))
        r123 = json.load(open(seed123_path))

        model_results = {"name": MODELS[model_key]["name"]}
        for m in METRICS:
            v1 = r42["test_metrics"][m]
            v2 = r123["test_metrics"][m]
            model_results[m] = ((v1 + v2) / 2, abs(v1 - v2) / 2)  

        model_results["train_min"] = (r42["train_minutes"] + r123["train_minutes"]) / 2
        model_results["inf_ms"] = (r42["inference_ms"]  + r123["inference_ms"])  / 2
        model_results["params_M"]  = r42["params_M"]
        results[model_key] = model_results

    return results


# CLASSIFICATION METRICS 
def print_classification_table(results):
    print(" Classification Metrics (Mean ± Std, 2 seeds)")
    header = f"{'Model':<14}" + "".join(f"  {METRIC_LABELS[m]:<14}" for m in METRICS)
    print(header)

    rows = []
    for key, r in results.items():
        row = {"Model": r["name"]}
        line = f"{r['name']:<14}"
        for m in METRICS:
            mean, std = r[m]
            cell = f"{mean:.3f}±{std:.3f}"
            line += f"  {cell:<14}"
            row[METRIC_LABELS[m]] = f"{mean:.3f} ± {std:.3f}"
        print(line)
        rows.append(row)


    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "table1_classification.csv")
    df.to_csv(path, index=False)
    print(f"\n  Saved: {path}")
    return df


# efficiency
def print_efficiency_table(results):
    print(f"{'Model':<14}  {'Params (M)':<14}  {'Train (min)':<14}  {'Infer (ms/img)'}")

    rows = []
    for key, r in results.items():
        params = r.get("params_M", 0)
        t_min  = r.get("train_minutes") or r.get("train_min", 0)
        i_ms   = r.get("inference_ms")  or r.get("inf_ms",   0)

        print(f"{r['name']:<14}  {params:<14.1f}  {t_min:<14.1f}  {i_ms:.1f}")
        rows.append({"Model": r["name"],
                     "Parameters (M)": params,
                     "Training Time (min)": round(t_min, 1),
                     "Inference (ms/img)": round(i_ms, 1)})


    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "table2_efficiency.csv")
    df.to_csv(path, index=False)
    print(f" Saved: {path}")
    return df


# BAR CHART COMPARISON 
def plot_bar_comparison(results):
    key_metrics = ["f1", "auc", "recall", "mcc"]
    labels = [METRIC_LABELS[m] for m in key_metrics]
    x = np.arange(len(key_metrics))
    width = 0.25
    model_keys  = list(results.keys())

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, key in enumerate(model_keys):
        r = results[key]
        means  = [r[m][0] for m in key_metrics]
        stds = [r[m][1] for m in key_metrics]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, means, width,
                      label=r["name"],
                      color=MODELS[key]["color"],
                      yerr=stds, capsize=4,
                      error_kw={"elinewidth": 1.5})
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{mean:.3f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison: F1, AUC, Recall, MCC\n(Mean ± Std, 2 seeds, RSNA test set)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.35)
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "figure1_bar_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f" Saved: {path}")


# RADAR CHART 
def plot_radar(results):
    radar_metrics = ["accuracy", "f1", "precision", "recall", "auc", "mcc"]
    labels = [METRIC_LABELS[m] for m in radar_metrics]
    N = len(labels)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]   # close the loop

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=7)

    for key, r in results.items():
        values = [r[m][0] for m in radar_metrics] + [r[radar_metrics[0]][0]]
        ax.plot(angles, values, "o-", linewidth=2,
                color=MODELS[key]["color"], label=r["name"])
        ax.fill(angles, values, alpha=0.1, color=MODELS[key]["color"])

    ax.set_title("Model Performance Profile\n(higher = better on all axes)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "figure2_radar.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f" Saved: {path}")


# SUMMARY 
def print_summary(results):
    print("\n  Best model by AUC:")
    best_key = max(results, key=lambda k: results[k]["auc"][0])
    best_r = results[best_key]
    print(f" {best_r['name']}  AUC={best_r['auc'][0]:.3f} ± {best_r['auc'][1]:.3f}")

    print("\n Best model by Recall (clinical priority):")
    best_rec = max(results, key=lambda k: results[k]["recall"][0])
    best_rr = results[best_rec]
    print(f" {best_rr['name']}  Recall={best_rr['recall'][0]:.3f} ± {best_rr['recall'][1]:.3f}")

    print("\n Fastest training:")
    fastest = min(results, key=lambda k: results[k]["train_min"])
    print(f" {results[fastest]['name']}  {results[fastest]['train_min']:.1f} min")



# main
if __name__ == "__main__":
    print("Building Master Results Table")
    

    results = load_results()

    if not results:
        print("\n No results found")
        for key in MODELS:
            for seed in [42, 123]:
                print(f" {key}_seed{seed}.json")
    else:
        print(f"\n Loaded results for: {', '.join(r['name'] for r in results.values())}")
        print_classification_table(results)
        print_efficiency_table(results)
        plot_bar_comparison(results)
        plot_radar(results)
        print_summary(results)
