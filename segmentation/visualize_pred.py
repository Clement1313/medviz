# /// script
# dependencies = [
#   "higra",
#   "numpy",
#   "scikit-image",
#   "scikit-learn",
#   "joblib",
#   "matplotlib",
# ]
# ///

import os
import sys
import joblib
import matplotlib.pyplot as plt
from skimage import io

sys.path.insert(0, os.path.dirname(__file__))

from maxtree import build_maxtree, compute_attributes
from predict import predict, cut_tree
from evaluation import load_mask

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "ddb1_v02_01")
IMG_DIR = os.path.join(DB_DIR, "images")
GT_DIR = os.path.join(DB_DIR, "groundtruth")
CLF_PATH = os.path.join(os.path.dirname(__file__), "clf.joblib")

clf = joblib.load(CLF_PATH)

THRESHOLD = 0.5
MAX_AREA = 5000
MIN_AREA = 500
N = 6  # nombre d'images à afficher

fnames = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".png"))[:N]

fig, axes = plt.subplots(N, 3, figsize=(15, 4 * N))

for i, fname in enumerate(fnames):
    stem = fname.replace(".png", "")
    gt_path = os.path.join(GT_DIR, stem + "_01.xml")
    image = io.imread(os.path.join(IMG_DIR, fname))

    # prédiction
    tree, altitudes, image_gray, graph = build_maxtree(image)
    attributes = compute_attributes(tree, altitudes, image_gray, graph)
    labels = predict(
        attributes, clf, threshold=THRESHOLD, max_area=MAX_AREA, min_area=MIN_AREA
    )
    mask_pred = cut_tree(tree, labels)
    if mask_pred.ndim == 1:
        mask_pred = mask_pred.reshape(image.shape[:2])

    # GT
    mask_gt = load_mask(gt_path, shape=image.shape[:2])

    # colonne 1 : image originale
    axes[i, 0].imshow(image)
    axes[i, 0].set_title(fname, fontsize=8)
    axes[i, 0].axis("off")

    # colonne 2 : GT en vert
    overlay_gt = image.copy()
    overlay_gt[mask_gt > 0] = [0, 200, 0]
    axes[i, 1].imshow(overlay_gt)
    axes[i, 1].set_title(f"GT ({(mask_gt > 0).sum()} px)", fontsize=8)
    axes[i, 1].axis("off")

    # colonne 3 : prédiction en rouge + GT en vert (contour)
    overlay_pred = image.copy()
    overlay_pred[mask_pred > 0] = [200, 0, 0]
    overlay_pred[mask_gt > 0] = [0, 200, 0]  # GT par-dessus en vert
    overlap = (mask_pred > 0) & (mask_gt > 0)
    overlay_pred[overlap] = [255, 165, 0]  # overlap en orange
    pred_px = int((mask_pred > 0).sum())
    axes[i, 2].imshow(overlay_pred)
    axes[i, 2].set_title(
        f"Pred (rouge) GT (vert) overlap (orange) — {pred_px} px pred", fontsize=8
    )
    axes[i, 2].axis("off")

    print(
        f"{fname} : pred={pred_px}px  gt={(mask_gt > 0).sum()}px  overlap={overlap.sum()}px"
    )

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "pred_visualization.png")
plt.savefig(out, dpi=80)
print(f"\nSauvegardé → {out}")
plt.show()
