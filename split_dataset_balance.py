import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)  # reproducible

# === CONFIG ===
SOURCE_DIR = Path("/home/ml164/YOLO/dataset")
IMAGES_DIR = SOURCE_DIR / "images"
LABELS_DIR = SOURCE_DIR / "labels"

DEST_DIR = Path("/home/ml164/YOLO/data")
DEST_IMAGES = DEST_DIR / "images"
DEST_LABELS = DEST_DIR / "labels"

# ratios must sum to 1.0 (test can be 0.0)
SPLIT_RATIOS = {"train": 0.85, "val": 0.15, "test": 0.0}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# === PREP DEST ===
for split in SPLIT_RATIOS:
    (DEST_IMAGES / split).mkdir(parents=True, exist_ok=True)
    (DEST_LABELS / split).mkdir(parents=True, exist_ok=True)

def parse_label_file(lbl_path: Path):
    """
    Parse YOLO label file -> set of class ids (as strings).
    If file is empty/missing -> return empty set (you can choose to skip those).
    """
    cls_ids = set()
    try:
        with open(lbl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # YOLO: class cx cy w h [other...]
                cls_ids.add(parts[0])
    except FileNotFoundError:
        pass
    return cls_ids

# === GATHER (image, label, key) ===
# key = tuple of sorted class ids in that image (multi-label "combination stratification")
pairs = []
for img_path in sorted(IMAGES_DIR.iterdir()):
    if img_path.suffix.lower() not in IMG_EXTS:
        continue
    lbl_path = LABELS_DIR / f"{img_path.stem}.txt"
    cls_ids = parse_label_file(lbl_path)
    # Skip images without labels if you want a pure detection set
    if not cls_ids or not lbl_path.exists():
        # If you want to include background-only images, comment the next line:
        continue
    key = tuple(sorted(cls_ids))  # e.g., ('0','3') means this image has classes 0 and 3
    pairs.append((img_path, lbl_path, key))

if not pairs:
    raise SystemExit("No (image,label) pairs found. Check paths and label files.")

# === GROUP BY LABEL-COMBO KEY ===
groups = defaultdict(list)
for idx, triple in enumerate(pairs):
    groups[triple[2]].append(idx)

# === STRATIFIED SPLIT BY GROUP (no sklearn) ===
def split_by_group(groups_dict, total_count, ratios):
    """
    For each group of identical label-combos, allocate items to train/val/test
    proportionally to ratios to preserve distribution of combos.
    """
    targets = {k: int(round(v * total_count)) for k, v in ratios.items()}
    # Adjust rounding so sum matches exactly
    diff = total_count - sum(targets.values())
    # fix rounding drift by adjusting train set (or any)
    if diff != 0:
        targets["train"] += diff

    # counters to track how many we already assigned
    counts = {"train": 0, "val": 0, "test": 0}
    split_indices = {"train": [], "val": [], "test": []}

    for key, idx_list in groups_dict.items():
        n = len(idx_list)
        # desired per group
        g_train = int(round(ratios["train"] * n))
        g_val   = int(round(ratios["val"] * n))
        g_test  = n - g_train - g_val  # make sure totals match n

        # randomize order within group, then slice
        random.shuffle(idx_list)
        train_idxs = idx_list[:g_train]
        val_idxs   = idx_list[g_train:g_train+g_val]
        test_idxs  = idx_list[g_train+g_val:]

        split_indices["train"].extend(train_idxs)
        split_indices["val"].extend(val_idxs)
        split_indices["test"].extend(test_idxs)

        counts["train"] += len(train_idxs)
        counts["val"]   += len(val_idxs)
        counts["test"]  += len(test_idxs)

    # Optional: small post-adjustment to hit exact totals (usually close enough already)
    # Here we leave as-is to keep it simple & reproducible.

    return split_indices

splits = split_by_group(groups, total_count=len(pairs), ratios=SPLIT_RATIOS)

# If you set test=0.0, ensure it's empty
if SPLIT_RATIOS.get("test", 0.0) == 0.0:
    splits["test"] = []

# === COPY FILES ===
def copy_split(split_name, indices):
    for i in indices:
        img_path, lbl_path, _ = pairs[i]
        shutil.copy2(img_path, DEST_IMAGES / split_name / img_path.name)
        shutil.copy2(lbl_path, DEST_LABELS / split_name / lbl_path.name)

for split_name, idxs in splits.items():
    random.shuffle(idxs)  # shuffle within split
    copy_split(split_name, idxs)

# === REPORT ===
def class_hist(indices):
    hist = defaultdict(int)
    for i in indices:
        _, lbl_path, _ = pairs[i]
        for cid in parse_label_file(lbl_path):
            hist[cid] += 1
    return dict(sorted(hist.items(), key=lambda x: int(x[0])))

print("✅ Done stratified splitting by label-combinations!")
for split_name in ["train", "val", "test"]:
    idxs = splits[split_name]
    print(f"{split_name}: {len(idxs)} images; class counts ≈ {class_hist(idxs)}")
