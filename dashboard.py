
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

#  CONFIG
SPLITS_DIR = "./data/splits"
IMG_SIZE   = 224
BATCH_SIZE = 16

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


#  DATASET 
class RSNADataset(Dataset):

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row["label"])
       
        img = Image.open(row["png_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


#  TRANSFORMS 
def get_transforms(split: str, img_size: int = 224):
    
    if split == "train":
        return T.Compose([
            T.Resize(img_size + 20),          # e.g. 244 → then crop to 224
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


#  WEIGHTED SAMPLER 
def make_weighted_sampler(labels):
  
    class_counts  = np.bincount(labels)                      
    class_weights = 1.0 / class_counts                  
    sample_weights = np.array([class_weights[l] for l in labels])
    return WeightedRandomSampler(
        weights = torch.from_numpy(sample_weights).float(),
        num_samples = len(sample_weights),
        replacement = True
    )


#  BUILD DATALOADERS 
def build_dataloaders(splits_dir=SPLITS_DIR, img_size=IMG_SIZE,
                      batch_size=BATCH_SIZE, num_workers=0):
    
    train_df = pd.read_csv(os.path.join(splits_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(splits_dir, "val.csv"))
    test_df  = pd.read_csv(os.path.join(splits_dir, "test.csv"))

    train_ds = RSNADataset(train_df, transform=get_transforms("train", img_size))
    val_ds  = RSNADataset(val_df,   transform=get_transforms("val",   img_size))
    test_ds  = RSNADataset(test_df,  transform=get_transforms("test",  img_size))

    train_sampler = make_weighted_sampler(train_df["label"].tolist())

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        sampler = train_sampler,   
        num_workers = num_workers,
        pin_memory  = False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle = False,
        num_workers = num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle = False,
        num_workers = num_workers,
    )
    return train_loader, val_loader, test_loader


#  VERIFICATION 
def verify_dataloaders(train_loader, val_loader, test_loader):

    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")

    imgs, labels = next(iter(train_loader))
    print(f"\n  One batch:")
    print(f" imgs shape  : {imgs.shape}    ← should be [16, 3, 224, 224]")
    print(f" labels shape: {labels.shape}  ← should be [16]")
    print(f" pixel range : [{imgs.min():.2f}, {imgs.max():.2f}]  ← normalised, not 0-255")
    print(f" label counts: {labels.tolist().count(0)} normal, {labels.tolist().count(1)} pneumonia")

    assert imgs.shape[1] == 3,   "Expected 3 channels — check .convert('RGB')"
    assert imgs.shape[2] == 224, "Expected 224px height"
    assert imgs.shape[3] == 224, "Expected 224px width"
    assert set(labels.tolist()).issubset({0, 1}), "Labels should be 0 or 1 only"
    print("\n   All checks passed.")

    # Visualise 8 samples
    _visualise_batch(imgs, labels)


def _visualise_batch(imgs, labels, n=8):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    imgs_display = (imgs[:n] * std + mean).clamp(0, 1)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    fig.suptitle("Training Batch — First 8 Samples\n(after transforms, before training)",
                 fontsize=12, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs_display[i].permute(1, 2, 0).numpy()[:, :, 0], cmap="gray")
        lbl   = int(labels[i].item())
        color = "#ED7D31" if lbl == 1 else "#2E75B6"
        ax.set_title("Pneumonia" if lbl == 1 else "Normal", color=color, fontsize=9, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    os.makedirs("./outputs/day5", exist_ok=True)
    plt.savefig("./outputs/day5/sample_batch.png", dpi=150)
    plt.show()


#  MAIN 
if __name__ == "__main__":
    train_loader, val_loader, test_loader = build_dataloaders()
    verify_dataloaders(train_loader, val_loader, test_loader)
   
    print("   DataLoaders are working")