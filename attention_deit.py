import os ,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import torch
import timm
import torch.nn as nn
import torchvision.transforms as T
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from dataload import get_transforms

CKPT_PATH  = "./outputs/deit_seed42/best_model.pth"
CASES_CSV  = "./outputs/gradcam_resnet50/selected_cases.csv"
OUTPUT_DIR = "./outputs/attention_deit"
IMG_SIZE = 224
PATCH_SIZE = 16
N_PATCHES_SIDE = IMG_SIZE // PATCH_SIZE   
 
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load model
def load_deit(ckpt_path):
    model = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=2)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  DeiT-Small loaded from {ckpt_path}")
    return model.to(DEVICE)


# Attention rollout
def get_attention_rollout(model, img_pil):
   
    transform = get_transforms("val", IMG_SIZE)
    input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
 
    raw_img = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
 
    # Storage for attention maps from each block
    attention_maps = []
 
    def hook_fn(module, input, output):
        # output shape: (batch, heads, seq_len, seq_len)
        attention_maps.append(output.detach().cpu())
 
    # Register hooks on all attention blocks
    hooks = []
    for block in model.blocks:
     
        hooks.append(block.attn.register_forward_hook(
            lambda m, i, o: attention_maps.append(o.detach().cpu())
        ))
 
    # Forward pass
    with torch.no_grad():
        _ = model(input_tensor)
 
    # Remove hooks
    for h in hooks: h.remove()
 
    # attention_maps: list of tensors (1, heads, seq_len, seq_len)
    # seq_len = 197 (1 CLS + 196 patches)
 
    # Rollout
    rollout = torch.eye(attention_maps[0].shape[-1])   
 
    for attn in attention_maps:
        # Average over heads
        attn_avg = attn[0].mean(dim=0)                 
        # Add identity 
        attn_aug = attn_avg + torch.eye(attn_avg.shape[0])
        attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True)
        # Accumulate rollout
        rollout  = torch.matmul(attn_aug, rollout)
 
    n_tokens = rollout.shape[0]        
    n_patches = n_tokens - 2             
    grid_size = int(n_patches ** 0.5)     

    cls_attn  = rollout[0, 1:-1].numpy()  
    cls_attn  = cls_attn[:grid_size * grid_size]   
    cls_attn  = cls_attn.reshape(grid_size, grid_size)
 
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
 
    attention_map = cv2.resize(cls_attn, (IMG_SIZE, IMG_SIZE))
 
    return attention_map, raw_img

#overlay attention map on image

def overlay_attention(raw_img, attention_map, alpha=0.5, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.applyColorMap((attention_map * 255).astype(np.uint8), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = (1 - alpha) * raw_img + alpha * heatmap
    return np.clip(overlay, 0, 1)

#save individual attention map

def save_individual(case_row, overlay, raw_img, attention_map, idx):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(
        f"DeiT-Small Attention Rollout — {case_row['case_type']}\n"
        f"True: {'Pneumonia' if case_row['label']==1 else 'Normal'}  |  "
        f"Predicted: {'Pneumonia' if case_row['pred']==1 else 'Normal'}  |  "
        f"Confidence: {case_row['prob']:.3f}",
        fontsize=10, fontweight="bold"
    )
    axes[0].imshow(raw_img); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(attention_map, cmap="jet")
    axes[1].set_title("Attention Map (raw)"); axes[1].axis("off")
    axes[2].imshow(overlay); axes[2].set_title("Overlay"); axes[2].axis("off")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"case{idx+1:02d}_{case_row['case_type'].replace(' ','_')}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
 
 
#  SUMMARY FIGURE 
def save_summary_figure(cases_data):
    n = len(cases_data)
    fig, axes = plt.subplots(n, 2, figsize=(8, n * 3.5))
    fig.suptitle("DeiT-Small Attention Rollout — Representative Cases",
                 fontsize=14, fontweight="bold", y=1.01)
 
    colors = {"True Positive":"#2E75B6","True Negative":"#70AD47",
              "False Positive":"#ED7D31","False Negative":"#FF0000"}
 
    for i, (case_row, overlay, raw_img, _) in enumerate(cases_data):
        color = colors.get(case_row["case_type"], "black")
        axes[i, 0].imshow(raw_img)
        axes[i, 0].set_title(
            f"{case_row['case_type']} | "
            f"{'Pneumonia' if case_row['label']==1 else 'Normal'} → "
            f"{'Pneumonia' if case_row['pred']==1 else 'Normal'} "
            f"(p={case_row['prob']:.2f})",
            fontsize=8, color=color, fontweight="bold"
        )
        axes[i, 0].axis("off")
        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title("Attention Rollout", fontsize=8)
        axes[i, 1].axis("off")
 
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "summary_attention_deit.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  Summary figure saved: {path}")
    return path
 
 
#  MAIN 
if __name__ == "__main__":
 
    model = load_deit(CKPT_PATH)
    cases = pd.read_csv(CASES_CSV)
    print(f"\n Loaded {len(cases)} ")
 
    print("\n Generating Attention Rollout maps...")
    cases_data = []
    for idx, (_, case_row) in enumerate(cases.iterrows()):
        print(f"  [{idx+1}/{len(cases)}] {case_row['case_type']}")
        img_pil = Image.open(case_row["png_path"]).convert("RGB")
        attention_map, raw_img = get_attention_rollout(model, img_pil)
        overlay = overlay_attention(raw_img, attention_map)
        save_individual(case_row, overlay, raw_img, attention_map, idx)
        cases_data.append((case_row, overlay, raw_img, attention_map))
 
    save_summary_figure(cases_data)
 
