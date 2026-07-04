import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as T
import torch.nn as nn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.insert(0, os.path.dirname(__file__))
from dataload import get_transforms

#  CONFIG 
CKPT_PATH  = "./outputs/resnet50_seed42/best_model.pth"
TEST_CSV = "./data/splits/test.csv"
OUTPUT_DIR = "./outputs/gradcam_resnet50"
IMG_SIZE   = 224

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# lead model
def load_resnet50(ckpt_path):
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, 2),
    )
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  ResNet50 loaded from {ckpt_path}")
    return model.to(DEVICE)


# predciting all images
def predict_all(model, test_df, transform):
    results = []
    model.eval()
    with torch.no_grad():
        for _, row in test_df.iterrows():
            img = Image.open(row["png_path"]).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            logits = model(tensor)
            prob = torch.softmax(logits, dim=1)[0, 1].item()
            pred = int(logits.argmax(dim=1).item())
            results.append({
                "patientId": row["patientId"],
                "png_path":  row["png_path"],
                "label":     int(row["label"]),
                "pred":      pred,
                "prob":      prob,
            })
    return pd.DataFrame(results)


#presentation 
def select_cases(preds_df):
    
    tp = preds_df[(preds_df["label"]==1) & (preds_df["pred"]==1)].nlargest(2, "prob")
    tn = preds_df[(preds_df["label"]==0) & (preds_df["pred"]==0)].nsmallest(2, "prob")
    fp = preds_df[(preds_df["label"]==0) & (preds_df["pred"]==1)]
    fn = preds_df[(preds_df["label"]==1) & (preds_df["pred"]==0)]

    fp_case = fp.iloc[[0]] if len(fp) > 0 else tp.iloc[[0]]
    fn_case = fn.iloc[[0]] if len(fn) > 0 else tn.iloc[[0]]

    cases = pd.concat([tp, tn, fp_case, fn_case]).reset_index(drop=True)
    case_types = ["True Positive", "True Positive",
                  "True Negative", "True Negative",
                  "False Positive", "False Negative"]
    cases["case_type"] = case_types[:len(cases)]

    print(f"\n  Selected cases:")
    for _, row in cases.iterrows():
        print(f" {row['case_type']:16s} | label={row['label']} pred={row['pred']} prob={row['prob']:.3f}")
    return cases


# Generate Grad Cam
def generate_gradcam(model, img_pil, target_class=1):
 
    transform = get_transforms("val", IMG_SIZE)

    input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

    raw_img = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0

    target_layers = [model.layer4[-1]]

    # Run Grad-CAM
    with GradCAM(model=model, target_layers=target_layers) as cam:
        targets  = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0]  

    cam_image = show_cam_on_image(raw_img, grayscale_cam, use_rgb=True)
    return cam_image, raw_img, grayscale_cam


# individual heatmap
def save_individual(case_row, cam_image, raw_img, idx):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(
        f"ResNet50 Grad-CAM — {case_row['case_type']}\n"
        f"True label: {'Pneumonia' if case_row['label']==1 else 'Normal'}  |  "
        f"Predicted: {'Pneumonia' if case_row['pred']==1 else 'Normal'}  |  "
        f"Confidence: {case_row['prob']:.3f}",
        fontsize=10, fontweight="bold"
    )
    axes[0].imshow(raw_img); axes[0].set_title("Original X-Ray"); axes[0].axis("off")
    axes[1].imshow(cam_image); axes[1].set_title("Grad-CAM Heatmap"); axes[1].axis("off")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"case{idx+1:02d}_{case_row['case_type'].replace(' ','_')}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# PANEL FIGURE 
def save_summary_figure(cases_data):
    
    n = len(cases_data)
    fig, axes = plt.subplots(n, 2, figsize=(8, n * 3.5))
    fig.suptitle("ResNet50 Grad-CAM — Representative Cases",
                 fontsize=14, fontweight="bold", y=1.01)

    colors = {
        "True Positive":  "#2E75B6",
        "True Negative":  "#70AD47",
        "False Positive": "#ED7D31",
        "False Negative": "#FF0000",
    }

    for i, (case_row, cam_image, raw_img, _) in enumerate(cases_data):
        color = colors.get(case_row["case_type"], "black")
        axes[i, 0].imshow(raw_img, cmap="gray" if raw_img.mean() < 0.5 else None)
        axes[i, 0].set_title(
            f"{case_row['case_type']} | "
            f"{'Pneumonia' if case_row['label']==1 else 'Normal'} → "
            f"{'Pneumonia' if case_row['pred']==1 else 'Normal'} "
            f"(p={case_row['prob']:.2f})",
            fontsize=8, color=color, fontweight="bold"
        )
        axes[i, 0].axis("off")
        axes[i, 1].imshow(cam_image)
        axes[i, 1].set_title("Grad-CAM", fontsize=8)
        axes[i, 1].axis("off")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "summary_gradcam_resnet50.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n Summary figure saved: {path}")
    return path


#  MAIN 
if __name__ == "__main__":


    # Load model
    model = load_resnet50(CKPT_PATH)

    # Load test set
    test_df   = pd.read_csv(TEST_CSV)
    transform = get_transforms("val", IMG_SIZE)

    # Predict all test images
    print("\n  Running inference on test set...")
    preds_df = predict_all(model, test_df, transform)
    acc = (preds_df["label"] == preds_df["pred"]).mean()
    print(f"  Test accuracy: {acc:.3f} ({int(acc*len(preds_df))}/{len(preds_df)} correct)")

    #
    cases = select_cases(preds_df)

    # Generate Grad-CAM for each case
    print("\n  Generating Grad-CAM heatmaps...")
    cases_data = []
    for idx, (_, case_row) in enumerate(cases.iterrows()):
        print(f"  [{idx+1}/6] {case_row['case_type']} — {os.path.basename(case_row['png_path'])}")
        img_pil = Image.open(case_row["png_path"]).convert("RGB")
        cam_image, raw_img, grayscale = generate_gradcam(model, img_pil, target_class=1)
        save_individual(case_row, cam_image, raw_img, idx)
        cases_data.append((case_row, cam_image, raw_img, grayscale))

    # Save summary figure
    summary_path = save_summary_figure(cases_data)

    cases[["patientId","png_path","label","pred","prob","case_type"]].to_csv(
        os.path.join(OUTPUT_DIR, "selected_cases.csv"), index=False
    )
    print("Done")
