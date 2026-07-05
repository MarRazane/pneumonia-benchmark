import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import torch
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, os.path.dirname(__file__))
CASES_CSV = "./outputs/gradcam_resnet50/selected_cases.csv"
GRADCAM_DIR  = "./outputs/gradcam_resnet50"
DEIT_ATTN_DIR   = "./outputs/attention_deit"
OUTPUT_DIR  = "./outputs/dino_attention_rad_dino"
COMPARISON_DIR  = "./outputs/comparison_figure"

HUGGINGFACE_MODEL = "microsoft/rad-dino"
IMG_SIZE = 224
PATCH_SIZE = 14  

os.makedirs(OUTPUT_DIR,     exist_ok=True)
os.makedirs(COMPARISON_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#LOAD RAD-DINO 
def load_rad_dino():
    processor = AutoProcessor.from_pretrained(HUGGINGFACE_MODEL)
    model = AutoModel.from_pretrained(HUGGINGFACE_MODEL).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  RAD-DINO loaded. Backbone frozen.")
    return model, processor


#DINO ATTENTION MAP 
def get_dino_attention(model, processor, img_pil):
    inputs       = processor(images=img_pil, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)

    native_h     = pixel_values.shape[2]
    native_w     = pixel_values.shape[3]
    n_patches_h  = native_h // PATCH_SIZE
    n_patches_w  = native_w // PATCH_SIZE

    attention_store = []

    def hook_fn(module, input, output):
        
        attention_store.append(input[0].detach().cpu())

    last_attn_module = model.encoder.layer[-1].attention.attention
    hook = last_attn_module.register_forward_hook(hook_fn)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values,
                        output_attentions=True)

    hook.remove()

    if outputs.attentions is not None and len(outputs.attentions) > 0:
        last_attn = outputs.attentions[-1][0]  
        print(f"  Using output_attentions: {last_attn.shape}")
    else:
        print(f"  output_attentions empty, using manual extraction")
        with torch.no_grad():
            temp_outputs = model(pixel_values=pixel_values,
                                 output_hidden_states=True)
            hidden = temp_outputs.hidden_states[-2] 

            attn_layer = model.encoder.layer[-1].attention.attention
            num_heads  = attn_layer.num_attention_heads
            head_dim   = attn_layer.attention_head_size
            seq_len  = hidden.shape[1]

            qkv = attn_layer.query, attn_layer.key, attn_layer.value
            q = qkv[0](hidden).reshape(1, seq_len, num_heads, head_dim).permute(0,2,1,3)
            k = qkv[1](hidden).reshape(1, seq_len, num_heads, head_dim).permute(0,2,1,3)
            scale = head_dim ** -0.5
            attn_mat = torch.softmax((q @ k.transpose(-2,-1)) * scale, dim=-1)
            last_attn = attn_mat[0]  
            print(f"Manual attention shape: {last_attn.shape}")

    n_heads   = last_attn.shape[0]
    cls_attn  = last_attn[:, 0, 1:].numpy()   

    head_maps = []
    for h in range(n_heads):
        hmap = cls_attn[h].reshape(n_patches_h, n_patches_w)
        hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
        head_maps.append(cv2.resize(hmap, (IMG_SIZE, IMG_SIZE)))

    mean_attn = cls_attn.mean(axis=0).reshape(n_patches_h, n_patches_w)
    mean_attn = (mean_attn - mean_attn.min()) / (mean_attn.max() - mean_attn.min() + 1e-8)
    attention_map = cv2.resize(mean_attn, (IMG_SIZE, IMG_SIZE))
    raw_img = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0

    return attention_map, raw_img, head_maps



def overlay_attention(raw_img, attention_map, alpha=0.5):
    heatmap = cv2.applyColorMap(
        (attention_map * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = (1 - alpha) * raw_img + alpha * heatmap
    return np.clip(overlay, 0, 1)


#  SAVE  DINO MAPS 
def save_individual_dino(case_row, overlay, raw_img, attention_map, head_maps, idx):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(
        f"RAD-DINO DINO Attention {case_row['case_type']}\n"
        f"True: {'Pneumonia' if case_row['label']==1 else 'Normal'}  |  "
        f"Predicted: {'Pneumonia' if case_row['pred']==1 else 'Normal'}",
        fontsize=10, fontweight="bold"
    )
    axes[0,0].imshow(raw_img); axes[0,0].set_title("Original"); axes[0,0].axis("off")
    axes[0,1].imshow(overlay); axes[0,1].set_title("Mean Attention (all heads)"); axes[0,1].axis("off")
    axes[0,2].imshow(attention_map, cmap="inferno")
    axes[0,2].set_title("Raw Attention Map"); axes[0,2].axis("off")

    for i, h_idx in enumerate([0, 3, 6, 9]):
        if h_idx < len(head_maps):
            r, c = 1, i % 3 if i < 3 else 2
            ax_idx = (1, i) if i < 3 else (1, 2)
            axes[1, i % 3].imshow(head_maps[h_idx], cmap="inferno")
            axes[1, i % 3].set_title(f"Head {h_idx+1}", fontsize=8)
            axes[1, i % 3].axis("off")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"case{idx+1:02d}_{case_row['case_type'].replace(' ','_')}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


#  comparison figure
def build_comparison_figure(cases, dino_data):
    
    n = len(cases)
    fig, axes = plt.subplots(n, 4, figsize=(16, n * 3.8))
    fig.suptitle(
        "Explainability Comparison: ResNet50 Grad-CAM vs DeiT Attention Rollout vs RAD-DINO DINO Attention",
        fontsize=13, fontweight="bold", y=1.01
    )

    col_titles = ["Original X-Ray", "ResNet50\nGrad-CAM", "DeiT-Small\nAttention Rollout", "RAD-DINO\nDINO Attention"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold", pad=10)

    colors = {"True Positive":"#2E75B6","True Negative":"#70AD47",
              "False Positive":"#ED7D31","False Negative":"#FF0000"}

    for i, (_, case_row) in enumerate(cases.iterrows()):
        color = colors.get(case_row["case_type"], "black")
        img_pil = Image.open(case_row["png_path"]).convert("RGB")
        raw_img = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0

        # Row label
        label_str = (f"{case_row['case_type']}\n"
                     f"True: {'Pneu' if case_row['label']==1 else 'Norm'} | "
                     f"Pred: {'Pneu' if case_row['pred']==1 else 'Norm'}")
        axes[i, 0].set_ylabel(label_str, fontsize=8, color=color,
                               fontweight="bold", rotation=0,
                               labelpad=80, va="center")

        axes[i, 0].imshow(raw_img); axes[i, 0].axis("off")

        gradcam_path = os.path.join(
            GRADCAM_DIR,
            f"case{i+1:02d}_{case_row['case_type'].replace(' ','_')}.png"
        )
        if os.path.exists(gradcam_path):
            gc_img = np.array(Image.open(gradcam_path))
            w = gc_img.shape[1] // 2
            axes[i, 1].imshow(gc_img[:, w:])
        else:
            axes[i, 1].text(0.5, 0.5, "Missing", ha="center", va="center")
        axes[i, 1].axis("off")

        deit_path = os.path.join(
            DEIT_ATTN_DIR,
            f"case{i+1:02d}_{case_row['case_type'].replace(' ','_')}.png"
        )
        if os.path.exists(deit_path):
            dt_img = np.array(Image.open(deit_path))
            w = dt_img.shape[1] // 3
            axes[i, 2].imshow(dt_img[:, 2*w:])
        else:
            axes[i, 2].text(0.5, 0.5, "Missing\nRun Day 16", ha="center", va="center")
        axes[i, 2].axis("off")

        dino_overlay = dino_data[i]["overlay"]
        axes[i, 3].imshow(dino_overlay); axes[i, 3].axis("off")

    plt.tight_layout()
    path = os.path.join(COMPARISON_DIR, "comparison_all_models.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


#main
if __name__ == "__main__":

    # Load RAD-DINO
    model, processor = load_rad_dino()

    cases = pd.read_csv(CASES_CSV)
    print(f"\n  Loaded {len(cases)}")

    # Generate DINO attention maps
    print("\n  Part A: Generating DINO attention maps...")
    dino_data = []
    for idx, (_, case_row) in enumerate(cases.iterrows()):
        print(f"  [{idx+1}/{len(cases)}] {case_row['case_type']}")
        img_pil = Image.open(case_row["png_path"]).convert("RGB")
        attention_map, raw_img, head_maps = get_dino_attention(model, processor, img_pil)
        overlay = overlay_attention(raw_img, attention_map)
        save_individual_dino(case_row, overlay, raw_img, attention_map, head_maps, idx)
        dino_data.append({
            "case_row":     case_row,
            "overlay":      overlay,
            "raw_img":      raw_img,
            "attention_map":attention_map,
            "head_maps":    head_maps,
        })

    print("\n Part B: Building 3-panel comparison figure...")
    build_comparison_figure(cases, dino_data)

    print(f"Individual DINO maps: {OUTPUT_DIR}/")