import os, sys, json, time, argparse, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoProcessor

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix, matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler

# config

SEED       = 42
SPLITS_DIR = "./data/splits"
EMB_DIR    = "./data/rad_dino_embeddings"   
OUTPUT_DIR = "./outputs/rad_dino"
RESULTS_DIR= "./outputs/results"
 
HUGGINGFACE_MODEL = "microsoft/rad-dino"

MLP_HIDDEN = 256
MLP_DROPOUT = 0.3
MLP_EPOCHS  = 100
MLP_LR      = 1e-3
MLP_BATCH   = 32
 
os.makedirs(EMB_DIR,     exist_ok=True)
os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
 
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#Feature Extraction

class PNGDataset(Dataset):
    def __init__(self, df ,processor):
        self.df = df.reset_index(drop=True)
        self.processor = processor

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["png_path"]).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  
        pid_col = "patientId" if "patientId" in row.index else row.index[0]
        return pixel_values, int(row["label"]), str(row[pid_col])
    
def extract_embeddings(split_name, df, model, processor, emb_dir, batch_size=8):
    out_emb  = os.path.join(emb_dir, f"{split_name}_embeddings.npy")
    out_lbls = os.path.join(emb_dir, f"{split_name}_labels.npy")
 
    if os.path.exists(out_emb) and os.path.exists(out_lbls):
        print(f"{split_name}: embeddings already exist, loading from cache.")
        return np.load(out_emb), np.load(out_lbls)
    
    dataset = PNGDataset(df, processor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_embs, all_labels = [],[]

    model.eval()
    with torch.no_grad():
        for pixel_values, labels, _ in tqdm(loader, desc=f" Ex tracting {split_name}"):
            pixel_values = pixel_values.to(DEVICE)
            outputs = model(pixel_values = pixel_values)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embs.append(cls_emb)
            all_labels.extend(labels.numpy())

    embeddings = np.concatenate(all_embs, axis=0)
    labels_arr = np.array(all_labels)

    np.save(out_emb, embeddings)
    np.save(out_lbls, labels_arr)

    return embeddings, labels_arr

def run_extraction():
    processor = AutoProcessor.from_pretrained(HUGGINGFACE_MODEL)
    model = AutoModel.from_pretrained(HUGGINGFACE_MODEL).to(DEVICE)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"RAD-DINO (frozen) {n_params:.0f}M params Device: {DEVICE}")
    print(f" Pretrained on: 882,775 chest X-rays (MIMIC-CXR, CheXpert, NIH, PadChest)\n")

    train_df = pd.read_csv(os.path.join(SPLITS_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(SPLITS_DIR, "val.csv"))
    test_df  = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))
 
    print(" Extracting embeddings")
    t0 = time.time()

    X_train, y_train = extract_embeddings("train", train_df, model, processor, EMB_DIR)
    X_val,   y_val   = extract_embeddings("val", val_df, model, processor, EMB_DIR)
    X_test,  y_test  = extract_embeddings("test", test_df, model, processor, EMB_DIR)

    elapsed = time.time() - t0

    print(f"\n Extraction complete in {elapsed/60:.1f} min")
    print(f"Train embeddings: {X_train.shape}  labels: {y_train.shape}")
    print(f"Val embeddings: {X_val.shape} labels: {y_val.shape}")
    print(f"Test  embeddings: {X_test.shape} labels: {y_test.shape}")
    print(f"\n Embeddings saved in: {EMB_DIR}/")
 
    _visualise_embeddings(X_train, y_train)
    return X_train, y_train, X_val, y_val, X_test, y_test



def _visualise_embeddings(X, y):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X2d = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {0: "#2E75B6", 1: "#ED7D31"}
    labels_text = {0: "Normal", 1: "Pneumonia"}
    for lbl in [0, 1]:
        mask = y == lbl
        ax.scatter(X2d[mask, 0], X2d[mask, 1],
                   c=colors[lbl], label=labels_text[lbl],
                   alpha=0.5, s=15, edgecolors="none")
    ax.set_title("RAD-DINO Embeddings — PCA (2D)\nAre the classes separable?",
                 fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "embeddings_pca.png")
    plt.savefig(path, dpi=150); plt.show()
    print(f" Saved PCA plot: {path}")


if __name__ == "__main__":
    run_extraction()
