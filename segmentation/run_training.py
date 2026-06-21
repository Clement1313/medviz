# /// script
# dependencies = [
#   "scikit-learn",
#   "joblib",
# ]
# ///

import os
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))

from training import train


DB_DIR = os.path.join(os.path.dirname(__file__), "..", "ddb1_v02_01")
IMG_DIR = os.path.join(DB_DIR, "images")
GT_DIR = os.path.join(DB_DIR, "groundtruth")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "clf.joblib")

LIMIT = None  # mettre une limite seulement si on veut tester rapidement


def load_dataset_paths(img_dir: str, gt_dir: str, annotator: str = "01") -> tuple:
    """
    Retourne les listes de chemins (images, gt) appariés.
    Cherche les GT sous la forme <stem>_<annotator>.xml.
    """
    image_paths, gt_paths = [], []
    for fname in sorted(f for f in os.listdir(img_dir) if f.endswith(".png")):
        stem = fname.replace(".png", "")
        gt = os.path.join(gt_dir, f"{stem}_{annotator}.xml")
        if os.path.exists(gt):
            image_paths.append(os.path.join(img_dir, fname))
            gt_paths.append(gt)
    return image_paths, gt_paths

# LOAD DATASET
image_paths, gt_paths = load_dataset_paths(IMG_DIR, GT_DIR)

if LIMIT:
    image_paths = image_paths[:LIMIT]
    gt_paths = gt_paths[:LIMIT]

print(f"Images : {len(image_paths)} (LIMIT={LIMIT})")

# SPLIT TRAIN/TEST IMAGES
idx_train, idx_test = train_test_split(
    range(len(image_paths)), test_size=0.3, random_state=42
)
train_imgs = [image_paths[i] for i in idx_train]
train_gts = [gt_paths[i] for i in idx_train]
test_imgs = [image_paths[i] for i in idx_test]
test_gts = [gt_paths[i] for i in idx_test]

print(f"Train : {len(train_imgs)} | Test : {len(test_imgs)}")

# TRAINING
clf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42, class_weight="balanced")
print("Training...")
clf = train(train_imgs, train_gts, clf)
joblib.dump(clf, MODEL_OUT)
print(f"Model saved in {MODEL_OUT}")
