import os, sys, json, time, argparse, random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix, matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler

EMB_DIR = "./data/rad_dino_embeddings"
OUTPUT_DIR = "./outputs/rad_dino"
RESULTS_DIR = "./outputs/results"

MLP_HIDDEN = 256
MLP_DROPOUT = 0.3
MLP_EPOCHS = 100
MLP_LR = 1e-3
MLP_BATCH = 32

os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


#Dataset
class EmbeddingDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=768888, hidden_dim=6, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x): return self.net(x)    

#Metrics
def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy" : round(accuracy_score(y_true, y_pred),                   4),
        "f1" : round(f1_score(y_true, y_pred,       zero_division=0),  4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall"   : round(recall_score(y_true, y_pred,   zero_division=0),  4),
        "auc" : round(roc_auc_score(y_true, y_prob),                    4),
        "mcc" : round(matthews_corrcoef(y_true, y_pred),                4),
    }

#Confusion Matrix
def plot_confusion(y_true, y_pred, seed):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Greens"); plt.colorbar(im)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Normal","Pneumonia"])
    ax.set_yticklabels(["Normal","Pneumonia"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"RAD-DINO Confusion Matrix — Seed {seed}", fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black",
                    fontsize=15, fontweight="bold")
    plt.tight_layout()
    out_dir = os.path.join(OUTPUT_DIR, f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150); plt.show()
    print(f"  Saved: {path}")

#Main Training Function
def train_mlp(seed):
    set_seed(seed)
  
    # Load precomputed embeddings
    def load(split):
        e = np.load(os.path.join(EMB_DIR, f"{split}_embeddings.npy"))
        l = np.load(os.path.join(EMB_DIR, f"{split}_labels.npy"))
        return e, l
 
    X_train, y_train = load("train")
    X_val, y_val = load("val")
    X_test, y_test = load("test")
 
    print(f"  Embeddings loaded — Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
 
    # Standardise: zero mean, unit variance per dimension
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
 
    train_loader = DataLoader(EmbeddingDataset(X_train, y_train),
                              batch_size=MLP_BATCH, shuffle=True)
    val_loader = DataLoader(EmbeddingDataset(X_val,   y_val),
                              batch_size=MLP_BATCH, shuffle=False)
    test_loader = DataLoader(EmbeddingDataset(X_test,  y_test),
                              batch_size=MLP_BATCH, shuffle=False)
 
    model = MLPClassifier(768, MLP_HIDDEN, MLP_DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MLP_EPOCHS)
 
    best_val_auc = 0.0
    best_state   = None
    patience_cnt = 0
    PATIENCE = 15
    t0   = time.time()
 
    print(f"  Training MLP ({MLP_EPOCHS} epochs max, early stop patience={PATIENCE})...")
 
    for epoch in range(1, MLP_EPOCHS + 1):
        # Train
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
        scheduler.step()
 
        # Validate
        model.eval()
        vl_preds, vl_probs, vl_true = [], [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                logits = model(X_b.to(DEVICE))
                vl_probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
                vl_preds.extend(logits.argmax(1).cpu().numpy())
                vl_true.extend(y_b.numpy())
 
        val_auc = roc_auc_score(vl_true, vl_probs)
        marker = ""
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
            marker = "saved"
        else:
            patience_cnt += 1
 
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{MLP_EPOCHS}]  val AUC={val_auc:.4f}{marker}")
 
        if patience_cnt >= PATIENCE:
            print(f" Early stopping at epoch {epoch}")
            break
 
    total_time = time.time() - t0

    # test  evaluation
    model.load_state_dict(best_state)
    model.eval()
    te_preds, te_probs, te_true = [], [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            logits = model(X_b.to(DEVICE))
            te_probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            te_preds.extend(logits.argmax(1).cpu().numpy())
            te_true.extend(y_b.numpy())
 
    test_m = compute_metrics(te_true, te_preds, te_probs)
 
    # Inference speed (MLP only)
    dummy = torch.randn(1, 768).to(DEVICE)
    with torch.no_grad():
        t_inf = time.time()
        for _ in range(500): model(dummy)
        inf_ms = (time.time() - t_inf) / 500 * 1000
 
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
 
    
    for k, v in test_m.items():
        print(f"  {k:10s}: {v:.4f}")
    print(f"train_min : {total_time/60:.1f} min")
    print(f"infer_ms : {inf_ms:.3f} ms/image (MLP head only)")
    print(f"params_M  : {n_params:.3f}M (MLP head only)")
 
    # Save checkpoint
    out_dir = os.path.join(OUTPUT_DIR, f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "state_dict":  model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_std":  scaler.scale_,
    }, os.path.join(out_dir, "best_model.pth"))
 
    # Save results JSON
    results = {
        "model": "RAD-DINO",
        "seed":  seed,
        "test_metrics": test_m,
        "train_minutes":round(total_time / 60, 2),
        "inference_ms": round(inf_ms, 3),
        "params_M":  round(n_params, 3),
        "best_val_auc": best_val_auc,
        "note": "Backbone frozen (87M). Only MLP head trained.",
    }
    path = os.path.join(RESULTS_DIR, f"rad_dino_seed{seed}.json")
    with open(path, "w") as f: json.dump(results, f, indent=2)
    print(f"  Saved: {path}")
 
    plot_confusion(te_true, te_preds, seed)
    return results

def summarise():
    files = [os.path.join(RESULTS_DIR, f"rad_dino_seed{s}.json") for s in [42, 123]]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        print(f"Missing: {missing}\nRun both seeds first.")
        return
 
    r42  = json.load(open(files[0]))
    r123 = json.load(open(files[1]))
 
  
    summary = {}
    for m in ["accuracy","f1","precision","recall","auc","mcc"]:
        v1, v2 = r42["test_metrics"][m], r123["test_metrics"][m]
        mean = (v1 + v2) / 2
        std  = abs(v1 - v2) / 2
        summary[m] = f"{mean:.3f} ± {std:.3f}"
        print(f"  {m:12s}: {summary[m]}")
 
    t_mean = (r42["train_minutes"] + r123["train_minutes"]) / 2
    i_mean = (r42["inference_ms"]  + r123["inference_ms"])  / 2
    print(f"\n  Train time : {t_mean:.1f} min (MLP only)")
    print(f"  Infer time : {i_mean:.3f} ms/image")
 
    path = os.path.join(RESULTS_DIR, "rad_dino_summary.json")
    with open(path, "w") as f:
        json.dump({
            "model":  "RAD-DINO (frozen backbone)",
            "mean_std":summary,
            "train_minutes": t_mean,
            "inference_ms": i_mean,
            "backbone_params_M": 87,
            "head_params_M":  r42["params_M"],
        }, f, indent=2)
    print(f" Saved: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--summarise", action="store_true")
    args = parser.parse_args()
 
    if args.summarise:
        summarise()
    else:
        train_mlp(seed=args.seed)