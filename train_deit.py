

import os, sys, json, time, argparse, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
#from transformers import AutoImageProcessor, DeiTForImageClassification

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix, matthews_corrcoef
)

sys.path.insert(0, os.path.dirname(__file__))
from dashboard import build_dataloaders


#  CONFIG 
SEED          = 123         
EPOCHS        = 15
BATCH_SIZE    = 8           
LR_HEAD       = 1e-3
LR_FULL       = 5e-5        
WEIGHT_DECAY  = 1e-4
FREEZE_EPOCHS = 5
IMG_SIZE      = 224

SPLITS_DIR  = "./data/splits"
OUTPUT_DIR  = f"./outputs/deit_seed{SEED}"
CKPT_PATH   = os.path.join(OUTPUT_DIR, "best_model.pth")
RESULTS_DIR = "./outputs/results"

HUGGINGFACE_MODEL = "facebook/deit-small-patch16-224"

os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#  REPRODUCIBILITY 
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


#  MODEL 
def build_deit():
 
    model = timm.create_model(
        "deit_small_patch16_224",
        pretrained=True,
        num_classes=2         
    )
 
    for name, param in model.named_parameters():
        if "head" not in name:         
            param.requires_grad = False
 
    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f" DeiT-Small | Total: {n_total:.1f}M params | Trainable: {n_train:.2f}M")
    return model
 
 
def unfreeze_backbone(model, lr, weight_decay):
    for p in model.parameters():
        p.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
 
 
#  METRICS 
def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy" : round(accuracy_score(y_true, y_pred), 4),
        "f1" : round(f1_score(y_true, y_pred, zero_division=0),  4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall" : round(recall_score(y_true, y_pred, zero_division=0),  4),
        "auc" : round(roc_auc_score(y_true, y_prob), 4),
        "mcc" : round(matthews_corrcoef(y_true, y_pred), 4),
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
            loss   = criterion(logits, labels)
 
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
 
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
 
    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return avg_loss, metrics, all_labels, all_preds, all_probs
 
 
#  PLOTS 
def plot_curves(history, out_dir, seed):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"DeiT-Small Training Curves (Seed {seed})", fontsize=13, fontweight="bold")
    epochs = range(1, len(history["train_loss"]) + 1)
    pairs  = [("loss","Loss"),("accuracy","Accuracy"),("f1","F1-Score"),
              ("auc","AUC"),("recall","Recall"),("mcc","MCC")]
 
    for ax, (key, title) in zip(axes.flat, pairs):
        ax.plot(epochs, history[f"train_{key}"], "b-o", ms=3, label="Train")
        ax.plot(epochs, history[f"val_{key}"],   "r-o", ms=3, label="Val")
        best = (np.argmin if key == "loss" else np.argmax)(history[f"val_{key}"])
        ax.axvline(best + 1, color="green", ls="--", alpha=0.5,
                   label=f"Best={history[f'val_{key}'][best]:.3f}")
        ax.axvline(FREEZE_EPOCHS + 0.5, color="orange", ls=":", alpha=0.7,
                   label="Unfreeze")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
 
    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150); plt.show()
    print(f"  Saved: {path}")
 
 
def plot_confusion(y_true, y_pred, out_dir, seed):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Oranges"); plt.colorbar(im)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Normal","Pneumonia"])
    ax.set_yticklabels(["Normal","Pneumonia"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"DeiT-Small Confusion Matrix — Seed {seed}", fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black",
                    fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150); plt.show()
    print(f"Saved: {path}")
 
 
#  MAIN TRAINING LOOP 
def train():
    set_seed(SEED)
    print(f"  DeiT-Small (timm) | Seed {SEED} | {EPOCHS} epochs | Device: {DEVICE}")
 
    train_loader, val_loader, test_loader = build_dataloaders(
        splits_dir=SPLITS_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
 
    model = build_deit().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=FREEZE_EPOCHS, eta_min=1e-6)
 
    history = {f"{ph}_{m}": []
               for ph in ("train","val")
               for m in ("loss","accuracy","f1","auc","recall","mcc","precision")}
 
    best_val_auc = 0.0
    patience_count = 0
    PATIENCE = 4
    total_time = 0.0
 
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
 
        if epoch == FREEZE_EPOCHS + 1:
            print(f"\n[Epoch {epoch}]  Unfreezing backbone ")
            optimizer = unfreeze_backbone(model, LR_FULL, WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - FREEZE_EPOCHS, eta_min=1e-7)
 
        tr_loss, tr_m, *_ = run_epoch(model, train_loader, criterion, optimizer, DEVICE, True)
        vl_loss, vl_m, *_ = run_epoch(model, val_loader,   criterion, optimizer, DEVICE, False)
        scheduler.step()
 
        elapsed = time.time() - t0
        total_time += elapsed
 
        for k, v in tr_m.items(): history[f"train_{k}"].append(v)
        for k, v in vl_m.items(): history[f"val_{k}"].append(v)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
 
        marker = ""
        if vl_m["auc"] > best_val_auc:
            best_val_auc = vl_m["auc"]
            patience_count = 0
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_metrics": vl_m}, CKPT_PATH)
            marker = "saved"
        else:
            patience_count += 1
 
        print(f"[{epoch:02d}/{EPOCHS}] "
              f"loss {tr_loss:.3f}/{vl_loss:.3f}  "
              f"AUC {tr_m['auc']:.3f}/{vl_m['auc']:.3f}  "
              f"F1 {vl_m['f1']:.3f}  "
              f"MCC {vl_m['mcc']:.3f}  "
              f"{elapsed:.0f}s{marker}")
 
        if patience_count >= PATIENCE and epoch > FREEZE_EPOCHS:
            print(f"Early stopping at epoch {epoch}")
            break
 
    #  Test evaluation 
    print(f"\n Loading best checkpoint (val AUC={best_val_auc:.4f})")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
 
    _, test_m, test_labels, test_preds, test_probs = run_epoch(
        model, test_loader, criterion, optimizer, DEVICE, False
    )
 
    # Inference speed
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    with torch.no_grad():
        for _ in range(5): model(dummy)
        t0 = time.time()
        for _ in range(20): model(dummy)
        inf_ms = (time.time() - t0) / 20 * 1000
 
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
 
    print(f" TEST RESULTS — DeiT-Small Seed {SEED}")
    for k, v in test_m.items():
        print(f"  {k:10s}: {v:.4f}")
    print(f" train_min : {total_time/60:.1f} min")
    print(f" infer_ms : {inf_ms:.1f} ms/image")
    print(f" params_M : {n_params:.1f}M")
 
    results = {
        "model": "DeiT-Small", "seed": SEED,
        "test_metrics": test_m,
        "train_minutes": round(total_time/60, 2),
        "inference_ms":  round(inf_ms, 2),
        "params_M": round(n_params, 1),
        "best_val_auc":  best_val_auc,
    }
    path = os.path.join(RESULTS_DIR, f"deit_seed{SEED}.json")
    with open(path, "w") as f: json.dump(results, f, indent=2)
    print(f"  Saved: {path}")
 
    plot_curves(history, OUTPUT_DIR, SEED)
    plot_confusion(test_labels, test_preds, OUTPUT_DIR, SEED)
    return results
 
 
#  SUMMARISE 
def summarise():
    files = [os.path.join(RESULTS_DIR, f"deit_seed{s}.json") for s in [42, 123]]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        print(f"Missing: {missing}\nRun both seeds first."); return
 
    r42  = json.load(open(files[0]))
    r123 = json.load(open(files[1]))
 
    print("\n" + "="*55)
    print("  DeiT-Small — Final Results (Mean ± Std, 2 seeds)")
    print("="*55)
    summary = {}
    for m in ["accuracy","f1","precision","recall","auc","mcc"]:
        v1, v2 = r42["test_metrics"][m], r123["test_metrics"][m]
        mean = (v1 + v2) / 2; std = abs(v1 - v2) / 2
        summary[m] = f"{mean:.3f} ± {std:.3f}"
        print(f"  {m:12s}: {summary[m]}")
 
    t_mean = (r42["train_minutes"] + r123["train_minutes"]) / 2
    i_mean = (r42["inference_ms"]  + r123["inference_ms"])  / 2
    print(f"\n  Train time : {t_mean:.1f} min")
    print(f"  Infer time : {i_mean:.1f} ms/image")
    print(f"  Params : {r42['params_M']}M")
    print("="*55)
 
    path = os.path.join(RESULTS_DIR, "deit_summary.json")
    with open(path, "w") as f:
        json.dump({"model": "DeiT-Small", "mean_std": summary,
                   "train_minutes": t_mean, "inference_ms": i_mean,
                   "params_M": r42["params_M"]}, f, indent=2)
    print(f"  Saved: {path}")
 
 
#  ENTRY POINT 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarise", action="store_true")
    args = parser.parse_args()
 
    if args.summarise:
        summarise()
    else:
        train()
        print(f"\n Done")