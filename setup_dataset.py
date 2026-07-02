import os
import random
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


DATA_ROOT = "./Data"

DCMS_DIR = os.path.join(DATA_ROOT, "stage_2_train_images")
CLASS_CSV = os.path.join(DATA_ROOT, "stage_2_detailed_class_info.csv")
Labels_CSV = os.path.join(DATA_ROOT, "stage_2_train_labels.csv")

PNG_DIR = os.path.join(DATA_ROOT, "png_images")
SPLIT_DIR = os.path.join(DATA_ROOT, "splits")


N_TRAIN = 300
N_VAL = 75
N_TEST = 75

SEED = 42
random.seed(SEED); np.random.seed(SEED)

os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs("./outputs", exist_ok=True)

#Load Labels 

class_df = pd.read_csv(CLASS_CSV).drop_duplicates(subset="patientId")
class_df["label"] = (class_df["class"] == "Lung Opacity").astype(int)
 

print(f"Total number of patients: {len(class_df)}")
print(class_df["class"].value_counts().to_string())
print(f"Number of patients with lung opacity: {class_df['label'].sum()}")

clean_df = class_df[class_df["class"].isin(["Lung Opacity", "Normal"])].copy()
print(f"\nAfter dropping ambiguous class: {len(clean_df)} patients kept")
print(f"  Pneumonia : {clean_df['label'].sum()}")
print(f"  Normal    : {(clean_df['label']==0).sum()}")

#Plot Distribution of Classes

fig, ax = plt.subplots(figsize=(7, 4))
counts = class_df["class"].value_counts()
colors = ["#2E75B6", "#ED7D31"]
ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.2)
for i, (name, val) in enumerate(counts.items()):
    ax.text(i, val + 80, str(val), ha="center", fontsize=12, fontweight="bold")
ax.set_title("RSNA Dataset — Class Distribution (clean subset)", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Number of Patients")
ax.set_ylim(0, counts.max() * 1.15)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("./outputs/class_distribution.png", dpi=150)
plt.show()
print("\nSaved: ./outputs/class_distribution.png")

#DICOM to PNG
def read_dicom_to_array(patient_id):
    
    path = os.path.join(DCMS_DIR, f"{patient_id}.dcm")
    dcm  = pydicom.dcmread(path)
    arr  = dcm.pixel_array.astype(np.float32)
 
    if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
 
    # Normalise to 0-255
    lo, hi = arr.min(), arr.max()
    arr = (arr - lo) / (hi - lo + 1e-8) * 255.0
    return arr.astype(np.uint8)
 
failed = []
all_ids = clean_df["patientId"].tolist()
 
for pid in tqdm(all_ids, desc="DICOM→PNG"):
    out_path = os.path.join(PNG_DIR, f"{pid}.png")
    if os.path.exists(out_path):
        continue
    try:
        arr = read_dicom_to_array(pid)
        img = Image.fromarray(arr)          
        img.save(out_path)
    except Exception as e:
        failed.append(pid)
 
print(f"\nDone. Converted {len(all_ids) - len(failed)} images.")
if failed:
    print(f"  {len(failed)} failed (will be excluded from splits): {failed[:3]}")

#Visualize some images
 
clean_df["png_exists"] = clean_df["patientId"].apply(
    lambda x: os.path.exists(os.path.join(PNG_DIR, f"{x}.png"))
)
clean_df = clean_df[clean_df["png_exists"]].copy()
print(f"Images available after filtering failed conversions: {len(clean_df)}")

pneumonia_ids = clean_df[clean_df["label"]==1]["patientId"].sample(3, random_state=SEED).tolist()
normal_ids    = clean_df[clean_df["label"]==0]["patientId"].sample(3, random_state=SEED).tolist()
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
fig.suptitle("RSNA Sample Images\nTop row: Pneumonia  |  Bottom row: Normal",
             fontsize=13, fontweight="bold")
 
for col, pid in enumerate(pneumonia_ids):
    arr = read_dicom_to_array(pid)
    axes[0, col].imshow(arr, cmap="gray")
    axes[0, col].set_title("PNEUMONIA", color="#ED7D31", fontweight="bold", fontsize=10)
    axes[0, col].axis("off")
 
for col, pid in enumerate(normal_ids):
    arr = read_dicom_to_array(pid)
    axes[1, col].imshow(arr, cmap="gray")
    axes[1, col].set_title("NORMAL", color="#2E75B6", fontweight="bold", fontsize=10)
    axes[1, col].axis("off")
 
plt.tight_layout()
plt.savefig("./outputs/sample_xrays.png", dpi=150)
plt.show()
print("Saved: ./outputs/sample_xrays.png")


# Build balaced splits

clean_df["png_path"] = clean_df["patientId"].apply(
    lambda x: os.path.join(PNG_DIR, f"{x}.png")
)
clean_df = clean_df[clean_df["png_path"].apply(os.path.exists)].copy()
print(f"Images available after PNG check: {len(clean_df)}")
 
pneumonia = clean_df[clean_df["label"]==1].sample(frac=1, random_state=SEED).reset_index(drop=True)
normal    = clean_df[clean_df["label"]==0].sample(frac=1, random_state=SEED).reset_index(drop=True)
 
needed = N_TRAIN + N_VAL + N_TEST
assert len(pneumonia) >= needed, f"Not enough pneumonia images: have {len(pneumonia)}, need {needed}"
assert len(normal)    >= needed, f"Not enough normal images: have {len(normal)}, need {needed}"
 
def make_split(pneu, norm, start, n):
    chunk = pd.concat([pneu.iloc[start:start+n], norm.iloc[start:start+n]])
    return chunk.sample(frac=1, random_state=SEED).reset_index(drop=True)
 
train_df = make_split(pneumonia, normal, 0,               N_TRAIN)
val_df   = make_split(pneumonia, normal, N_TRAIN,          N_VAL)
test_df  = make_split(pneumonia, normal, N_TRAIN + N_VAL,  N_TEST)
 
train_df.to_csv(os.path.join(SPLIT_DIR, "train.csv"), index=False)
val_df.to_csv(  os.path.join(SPLIT_DIR, "val.csv"),   index=False)
test_df.to_csv( os.path.join(SPLIT_DIR, "test.csv"),  index=False)
 
print(f"\n✅ Splits saved to {SPLIT_DIR}/")
print(f"   train.csv : {len(train_df):4d} images  "
      f"({train_df['label'].sum()} pneumonia, {(train_df['label']==0).sum()} normal)")
print(f"   val.csv   : {len(val_df):4d} images  "
      f"({val_df['label'].sum()} pneumonia, {(val_df['label']==0).sum()} normal)")
print(f"   test.csv  : {len(test_df):4d} images  "
      f"({test_df['label'].sum()} pneumonia, {(test_df['label']==0).sum()} normal)")
 
