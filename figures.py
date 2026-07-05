
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc
from PIL import Image
 
RESULTS_DIR = "./outputs/results"
OUTPUT_DIR  = "./outputs/paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
MODELS = {
    "resnet50": {"name": "ResNet50",   "color": "#2E75B6", "ls": "-"},
    "deit":     {"name": "DeiT-Small", "color": "#ED7D31", "ls": "--"},
    "rad_dino": {"name": "RAD-DINO",   "color": "#70AD47", "ls": "-."},
}
 
 
def load_summaries():
    summaries = {}
    for key in MODELS:
        path = os.path.join(RESULTS_DIR, f"{key}_summary.json")
        if os.path.exists(path):
            summaries[key] = json.load(open(path))
        else:
            print(f"  ⚠  Missing: {path}")
    return summaries
 
 
# METRICS BAR CHART 
def plot_metrics_bar(summaries):
    metrics = ["auc", "f1", "recall", "mcc"]
    labels = ["AUC", "F1-Score", "Recall", "MCC"]
    x  = np.arange(len(metrics))
    width  = 0.25
    model_keys  = list(summaries.keys())
 
    fig, ax = plt.subplots(figsize=(10, 5))
 
    for i, key in enumerate(model_keys):
        r     = summaries[key]
        means, stds = [], []
        for m in metrics:
            val = r["mean_std"][m]        
            mean, std = [float(v.strip()) for v in val.split("±")]
            means.append(mean); stds.append(std)
 
        offset = (i - 1) * width
        bars = ax.bar(x + offset, means, width,
                      label=MODELS[key]["name"],
                      color=MODELS[key]["color"],
                      yerr=stds, capsize=4,
                      error_kw={"elinewidth": 1.5, "ecolor": "black"},
                      edgecolor="white", linewidth=0.5)
 
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + std + 0.008,
                    f"{mean:.3f}", ha="center", va="bottom",
                    fontsize=7, fontweight="bold")
 
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0.5, 1.10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Classification Performance Comparison\n(Mean ± Std, 2 seeds, RSNA test set)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)
    ax.axhline(0.9, color="gray", ls=":", alpha=0.5, lw=1)
 
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_metrics_bar.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {path}")
 
 
#  EFFICIENCY SCATTER
def plot_efficiency(summaries):
    fig, ax = plt.subplots(figsize=(8, 5))
 
    for key, r in summaries.items():
        auc_val = float(r["mean_std"]["auc"].split("±")[0])
        t_min = r.get("train_minutes", 0)
        params  = r.get("head_params_M", r.get("params_M", 0))
        if isinstance(params, str):
            params = float(params)
 
        size = max(params * 40, 50)
 
        ax.scatter(t_min, auc_val,
                   s=size, color=MODELS[key]["color"],
                   label=f"{MODELS[key]['name']} ({params:.1f}M params trained)",
                   zorder=5, edgecolors="black", linewidth=0.8)
        ax.annotate(MODELS[key]["name"],
                    (t_min, auc_val),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=10, fontweight="bold",
                    color=MODELS[key]["color"])
 
    ax.set_xlabel("Training Time (minutes)", fontsize=12)
    ax.set_ylabel("AUC (Mean, 2 seeds)", fontsize=12)
    ax.set_title("AUC vs Training Time\n(bubble size = trainable parameters)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0.88, 1.00)
    ax.grid(alpha=0.35, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)
 
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_efficiency_scatter.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {path}")
 
 
# CONFUSION MATRICES SIDE BY SIDE 
def plot_confusion_matrices():
    cm_paths = {
        "ResNet50": "./outputs/resnet50_seed42/confusion_matrix.png",
        "DeiT-Small": "./outputs/deit_seed42/confusion_matrix.png",
        "RAD-DINO": "./outputs/rad_dino/seed42/confusion_matrix.png",
    }
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Confusion Matrices — All Models (Seed 42, Test Set)",
                 fontsize=13, fontweight="bold")
 
    for ax, (name, path) in zip(axes, cm_paths.items()):
        if os.path.exists(path):
            img = np.array(Image.open(path))
            ax.imshow(img); ax.axis("off")
            ax.set_title(name, fontsize=12, fontweight="bold", pad=10)
        else:
            ax.text(0.5, 0.5, f"Missing:\n{path}", ha="center", va="center",
                    fontsize=9, color="red")
            ax.axis("off")
 
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_confusion_matrices.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {path}")
 
 
# SUMMARY TABLE AS FIGURE 
def plot_results_table(summaries):
    metrics_display = ["Accuracy", "F1-Score", "Precision", "Recall", "AUC", "MCC"]
    metrics_keys= ["accuracy", "f1", "precision", "recall", "auc", "mcc"]
 
    rows = []
    for key, r in summaries.items():
        row = [MODELS[key]["name"]]
        for m in metrics_keys:
            row.append(r["mean_std"][m])
        rows.append(row)
 
    fig, ax = plt.subplots(figsize=(14, 2.5 + len(rows) * 0.6))
    ax.axis("off")
 
    col_labels = ["Model"] + metrics_display
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)
 
    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#1F4E79")
        table[0, j].set_text_props(color="white", fontweight="bold")
 
    # Style data rows
    row_colors = ["#D6E4F0", "#FFFFFF", "#D9EAD3"]
    for i, key in enumerate(summaries.keys()):
        for j in range(len(col_labels)):
            table[i+1, j].set_facecolor(row_colors[i % len(row_colors)])
 
    ax.set_title("Table: Classification Results (Mean ± Std, 2 seeds)",
                 fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_results_table.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {path}")
 
 
#  MAIN 
if __name__ == "__main__":
   
    summaries = load_summaries()
    if not summaries:
        print("  Run all models first.")
    else:
        print(f" Loaded: {', '.join(MODELS[k]['name'] for k in summaries)}\n")
 
        print("  Figure 1: Metrics bar chart...")
        plot_metrics_bar(summaries)
 
        print("  Figure 2: Efficiency scatter...")
        plot_efficiency(summaries)
 
        print("  Figure 3: Confusion matrices...")
        plot_confusion_matrices()
 
        print("  Figure 4: Results table...")
        plot_results_table(summaries)
 
        print(f" All figures saved to: {OUTPUT_DIR}/")
