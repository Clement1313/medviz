import os

import joblib
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from segmentation import segment
from evaluation import (
    load_mask,
    confusionCounts,
    sensitivity,
    specificity,
    false_positive_rate,
    false_negative_rate,
    weighted_error_rate,
    roc_curve,
)

# dossier racine du dataset DiaRetDB1
# segmentation/ -> medviz/ -> medviz/ (racine repo) -> ddb1_v02_01
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
DATA_DIR = os.path.join(_REPO_ROOT, "ddb1_v02_01")
CLF_PATH = os.path.join(_HERE, "clf.joblib")  # modèle pré-entraîné
N_TEST = 60  # nb d'images de test


def parse_split(data_dir: str, split_file: str) -> list:
    """
    Lit un fichier de split DiaRetDB1 et retourne une liste de (image, gt_xml).
    On garde le 1er annotateur (suffixe _01) comme ground truth.
    """
    pairs = []
    with open(os.path.join(data_dir, split_file)) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            image = os.path.join(data_dir, parts[0])
            gt = os.path.join(data_dir, parts[1])  # premier annotateur
            pairs.append((image, gt))
    return pairs


def main():
    test_pairs = parse_split(DATA_DIR, "ddb1_v02_01_test.txt")[:N_TEST]

    print(f"Chargement du modèle {os.path.basename(CLF_PATH)}...")
    clf = joblib.load(CLF_PATH)

    print("Evaluation :")
    mask_gt_lst, mask_pred_lst = [], []
    for img, gt in test_pairs:
        # même pipeline que training.evaluate : segment + reconstruction du masque
        image = io.imread(img)
        h, w = image.shape[:2]
        mask_pred = np.zeros((h, w), dtype=np.int32)
        for label, binary_mask in segment(img, clf):
            mask_pred[binary_mask] = label
        mask_gt = load_mask(gt, shape=(h, w))

        mask_gt_lst.append(mask_gt)
        mask_pred_lst.append(mask_pred)

        # IoU gardé comme diagnostic par image (le marquage GT est grossier,
        # donc à ne pas prendre comme métrique principale)
        # print(f"  {os.path.basename(img)} : IoU={compute_iou(mask_gt, mask_pred):.4f}")

    # protocole DiaRetDB1 : matrice de confusion au niveau IMAGE
    tp, fp, fn, tn = confusionCounts(mask_gt_lst, mask_pred_lst, min_pixels=15)
    print(f"\nConfusion (image-based) : TP={tp} FP={fp} FN={fn} TN={tn}")

    if tp + fn > 0:
        print(f"Sensibilité = {sensitivity(tp, fn):.3f}")
    else:
        print("Sensibilité = n/a (aucune image anormale dans le test)")
    if tn + fp > 0:
        print(f"Spécificité = {specificity(tn, fp):.3f}")
    else:
        print("Spécificité = n/a (aucune image normale dans le test)")
    if (tp + fn > 0) and (tn + fp > 0):
        fpr = false_positive_rate(fp, tn)
        fnr = false_negative_rate(fn, tp)
        print(f"WER(R=10) = {weighted_error_rate(fpr, fnr, R=10):.3f}")

    # --- affichage matplotlib : matrice de confusion + courbe ROC ---
    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    # matrice de confusion (lignes = vérité, colonnes = prédiction)
    cm = np.array([[tp, fn], [fp, tn]])
    ax = axes[0]
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["anormale", "normale"])
    ax.set_yticks([0, 1], ["anormale", "normale"])
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Vérité (GT)")
    ax.set_title(f"Matrice de confusion (image-based, N={len(test_pairs)})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16)

    # courbe ROC (balayage du score-image), seulement si le test est mixte
    ax = axes[1]
    if (tp + fn > 0) and (tn + fp > 0):
        points = roc_curve(mask_gt_lst, mask_pred_lst, number_thresholds=50)
        fprs = [p[1] for p in points]
        tprs = [p[2] for p in points]
        ax.plot(fprs, tprs, marker=".", label="modèle")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="aléatoire")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("FPR (1 - spécificité)")
        ax.set_ylabel("TPR (sensibilité)")
        ax.set_title("Courbe ROC (DiaRetDB1, image-based)")
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "ROC indisponible\n(le test doit mélanger\nimages normales et anormales)",
            ha="center",
            va="center",
        )
        ax.axis("off")

    plt.tight_layout()
    out = os.path.join(_HERE, "eval_metrics.png")
    plt.savefig(out, dpi=100)
    print(f"\nFigure sauvegardée → {out}")
    plt.show()


if __name__ == "__main__":
    main()
