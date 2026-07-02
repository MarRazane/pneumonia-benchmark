"""
DAY 6 & 7 — Train ResNet50 (Seed 42, then Seed 123)
=====================================================
What this file does:
  - Loads ImageNet-pretrained ResNet50
  - Phase 1 (epochs 1-5):   backbone FROZEN, only train new classification head
  - Phase 2 (epochs 6-15):  backbone UNFROZEN, full fine-tuning with small LR
  - Saves best checkpoint (by val AUC)
  - Logs every epoch: loss, accuracy, F1, AUC, MCC, precision, recall
  - Plots training curves + confusion matrix after training

HOW TO USE
----------
Day 6: Run with SEED = 42 (leave default)
Day 7: Change SEED = 123, run again

Results from both seeds are saved as JSON.
After Day 7, run: python day6_7_train_resnet50.py --summarise
to compute Mean ± Std across the two seeds.

INSTALL
-------
pip install torch torchvision scikit-learn matplotlib numpy pandas tqdm
"""

import os, sys, json, time, argparse, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix, matthews_corrcoef
)

sys.path.insert(0, os.path.dirname(__file__))
from dashboard import build_dataloaders

SEED  = 42          
EPOCHS   = 15
BATCH_SIZE  = 16
LR_HEAD  = 1e-3        
LR_FULL  = 1e-4       
WEIGHT_DECAY  = 1e-4
FREEZE_EPOCHS = 5           
IMG_SIZE = 224

SPLITS_DIR  = "./data/splits"
OUTPUT_DIR  = f"./outputs/resnet50_seed{SEED}"
CKPT_PATH   = os.path.join(OUTPUT_DIR, "best_model.pth")
RESULTS_DIR = "./outputs/results"

os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#  REPRODUCIBILITY 
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


#  MODEL 
def build_resnet50():
   
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    for p in model.parameters():
        p.requires_grad = False

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, 2),
    )
    for p in model.fc.parameters():
        p.requires_grad = True

    n_total  = sum(p.numel() for p in model.parameters()) / 1e6
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  ResNet50 | Total: {n_total:.1f}M params | Trainable: {n_train:.1f}M")
    return model


def unfreeze_backbone(model, lr, weight_decay):
    for p in model.parameters():
        p.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Backbone unfrozen → Trainable: {n_train:.1f}M params | LR → {lr}")
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


#  METRICS 
def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy" : round(accuracy_score(y_true, y_pred),                    4),
        "f1" : round(f1_score(y_true, y_pred,       zero_division=0),   4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0),  4),
        "recall"   : round(recall_score(y_true, y_pred,   zero_division=0),   4),
        "auc" : round(roc_auc_score(y_true, y_prob),                     4),
        "mcc" : round(matthews_corrcoef(y_true, y_pred),                 4),
    }


#  ONE EPOCH 
def run_epoch(model, loader, criterion, optimizer, device, training=True):
    model.train() if training else model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, desc="  train" if training else "  eval ", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_labels, all_preds, all_probs)
    return avg_loss, metrics, all_labels, all_preds, all_probs


#  PLOTS 
def plot_curves(history, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"ResNet50 Training Curves (Seed {SEED})", fontsize=13, fontweight="bold")
    epochs = range(1, len(history["train_loss"]) + 1)
    pairs = [("loss","Loss"),("accuracy","Accuracy"),("f1","F1-Score"),
             ("auc","AUC"),("recall","Recall"),("mcc","MCC")]

    for ax, (key, title) in zip(axes.flat, pairs):
        ax.plot(epochs, history[f"train_{key}"], "b-o", ms=3, label="Train")
        ax.plot(epochs, history[f"val_{key}"],   "r-o", ms=3, label="Val")
        best = np.argmax(history[f"val_{key}"]) if key != "loss" else np.argmin(history[f"val_{key}"])
        ax.axvline(best + 1, color="green", ls="--", alpha=0.5,
                   label=f"Best={history[f'val_{key}'][best]:.3f}")
        ax.set_title(title, fontweight="bold"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    for ax in axes.flat:
        ax.axvline(FREEZE_EPOCHS + 0.5, color="orange", ls=":", alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.show()


def plot_confusion(y_true, y_pred, out_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Normal","Pneumonia"]); ax.set_yticklabels(["Normal","Pneumonia"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"ResNet50 Confusion Matrix — Seed {SEED}", fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black",
                    fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.show()


#  MAIN TRAINING LOOP 
def train():
    set_seed(SEED)
    print(f"  ResNet50 | Seed {SEED} | {EPOCHS} epochs | Device: {DEVICE}")
  
    train_loader, val_loader, test_loader = build_dataloaders(
        splits_dir=SPLITS_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE
    )

    model = build_resnet50().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=FREEZE_EPOCHS, eta_min=1e-6)

    history = {f"{phase}_{m}": []
               for phase in ("train","val")
               for m in ("loss","accuracy","f1","auc","recall","mcc","precision")}

    best_val_auc = 0.0
    patience_count  = 0
    PATIENCE  = 4
    total_time  = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        if epoch == FREEZE_EPOCHS + 1:
            print(f"\n[Epoch {epoch}] ── Unfreezing backbone ──")
            optimizer = unfreeze_backbone(model, LR_FULL, WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - FREEZE_EPOCHS, eta_min=1e-7)

        tr_loss, tr_m, *_  = run_epoch(model, train_loader, criterion, optimizer, DEVICE, training=True)
        vl_loss, vl_m, *_  = run_epoch(model, val_loader,   criterion, optimizer, DEVICE, training=False)
        scheduler.step()

        elapsed = time.time() - t0
        total_time += elapsed

        for m_key, val in tr_m.items(): history[f"train_{m_key}"].append(val)
        for m_key, val in vl_m.items(): history[f"val_{m_key}"].append(val)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)

        marker = ""
        if vl_m["auc"] > best_val_auc:
            best_val_auc = vl_m["auc"]
            patience_count = 0
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_metrics": vl_m}, CKPT_PATH)
            marker = "  ← saved"
        else:
            patience_count += 1

        print(f"[{epoch:02d}/{EPOCHS}] "
              f"loss {tr_loss:.3f}/{vl_loss:.3f}  "
              f"AUC {tr_m['auc']:.3f}/{vl_m['auc']:.3f}  "
              f"F1 {vl_m['f1']:.3f}  "
              f"MCC {vl_m['mcc']:.3f}  "
              f"{elapsed:.0f}s{marker}")

        if patience_count >= PATIENCE and epoch > FREEZE_EPOCHS:
            print(f"  Early stopping at epoch {epoch}")
            break

    #  Test evaluation 
    print(f"\n── Loading best checkpoint (val AUC={best_val_auc:.4f}) ──")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    _, test_m, test_labels, test_preds, test_probs = run_epoch(
        model, test_loader, criterion, optimizer, DEVICE, training=False
    )

    # Inference speed
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    with torch.no_grad():
        _ = [model(dummy) for _ in range(10)]  
        t_inf = time.time()
        for _ in range(50): model(dummy)
        inf_ms = (time.time() - t_inf) / 50 * 1000

    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"\n{'─'*50}")
    print(f"  TEST RESULTS  {SEED}")
    print(f"{'─'*50}")
    for k, v in test_m.items():
        print(f"  {k:10s}: {v:.4f}")
    print(f"  train_min  : {total_time/60:.1f} min")
    print(f"  infer_ms  : {inf_ms:.1f} ms/image")
    print(f"  params_M  : {n_params:.1f}M")
    print(f"{'─'*50}")

    # Save results JSON
    results = {
        "model": "ResNet50", "seed": SEED,
        "test_metrics": test_m,
        "train_minutes": round(total_time/60, 2),
        "inference_ms": round(inf_ms, 2),
        "params_M": round(n_params, 1),
        "best_val_auc": best_val_auc,
    }
    json_path = os.path.join(RESULTS_DIR, f"resnet50_seed{SEED}.json")
    with open(json_path, "w") as f: json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")

    plot_curves(history, OUTPUT_DIR)
    plot_confusion(test_labels, test_preds, OUTPUT_DIR)
    return results


#  SUMMARISE 
def summarise():
    files = [os.path.join(RESULTS_DIR, f"resnet50_seed{s}.json") for s in [42, 123]]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        print(f"Missing results files: {missing}")
        print("Run training for both seeds first.")
        return

    r42  = json.load(open(files[0]))
    r123 = json.load(open(files[1]))

    metrics = ["accuracy","f1","precision","recall","auc","mcc"]
    summary = {}
    for m in metrics:
        v1 = r42["test_metrics"][m]
        v2 = r123["test_metrics"][m]
        mean = (v1 + v2) / 2
        std  = abs(v1 - v2) / 2
        summary[m] = f"{mean:.3f} ± {std:.3f}"
        print(f"  {m:12s}: {summary[m]}")

    t_mean = (r42["train_minutes"] + r123["train_minutes"]) / 2
    i_mean = (r42["inference_ms"]  + r123["inference_ms"])  / 2
    print(f"\n  Train time : {t_mean:.1f} min (average)")
    print(f"  Infer time : {i_mean:.1f} ms/image")
    print(f"  Params : {r42['params_M']}M")
   

    summary_path = os.path.join(RESULTS_DIR, "resnet50_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"model": "ResNet50", "mean_std": summary,
                   "train_minutes": t_mean, "inference_ms": i_mean,
                   "params_M": r42["params_M"]}, f, indent=2)
    print(f" Saved: {summary_path}")


#  ENTRY POINT 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarise", action="store_true",
                        help="Compute Mean±Std after both seeds are done")
    args = parser.parse_args()

    if args.summarise:
        summarise()
    else:
        train()
        print("\n Done. ")