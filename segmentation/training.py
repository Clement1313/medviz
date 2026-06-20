import higra as hg
import numpy as np
from skimage import io

from evaluation import load_mask, compute_iou
from segmentation import segment
from maxtree import build_maxtree, compute_attributes


def build_node_labels(tree, gt_mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convertit un masque GT (niveau pixel) en un label binaire par noeud du maxtree.

    Les feuilles du maxtree correspondent aux pixels (ordre reshape(-1)). On
    accumule le GT binarise des feuilles vers tous les noeuds, puis un noeud est
    positif si la fraction de ses pixels appartenant a un exsudat depasse le seuil.

    Args:
        tree:      maxtree higra
        gt_mask:   masque ground truth (H, W), >0 sur les exsudats (multi-label)
        threshold: fraction minimale de pixels exsudat pour qu'un noeud soit positif

    Returns:
        labels: (nb_noeuds,) int32, 1 = exsudat, 0 = fond
    """
    # 1 si exudat 0 sinon (plus de distinctions entre les exudats)
    leaf_gt = (gt_mask.reshape(-1) > 0).astype(np.float64)

    # nb de pixels exsudat par noeud
    positive_count = hg.accumulate_sequential(tree, leaf_gt, hg.Accumulators.sum)

    # nb de pixels total par noeud
    area = hg.attribute_area(tree)
    coverage = positive_count / area

    return (coverage > threshold).astype(np.int32)


def evaluate(image_path: str, gt_mask_path: str, clf) -> float:
    """
    Évalue la segmentation sur une image avec un masque GT.

    Args:
        image_path: chemin vers l'image
        gt_mask_path: chemin vers le masque ground truth
        clf: classifieur sklearn entraîné

    Returns:
        score IoU
    """
    # segmentation
    masks = segment(image_path, clf)

    # reconstruction masque predit
    image = io.imread(image_path)
    h, w = image.shape[:2]
    mask_pred = np.zeros((h, w), dtype=np.int32)
    for label, binary_mask in masks:
        mask_pred[binary_mask] = label

    mask_gt = load_mask(gt_mask_path)

    score = compute_iou(mask_gt, mask_pred)

    return score


def train(image_paths: list, ground_truth_mask_paths: list, clf):
    """
    Entraîne le classifieur sur une liste d'images annotées.

    Args:
        image_paths: liste de chemins vers les images
        ground_truth_mask_paths: liste de chemins vers les masques GT correspondants
        clf: classifieur sklearn

    Returns:
        clf entraîné
    """
    X_list, y_list = [], []

    for image_path, gt_path in zip(image_paths, ground_truth_mask_paths):
        image = io.imread(image_path)

        tree, altitudes, image_gray, graph = build_maxtree(image)
        attributes = compute_attributes(
            tree, altitudes, image_gray, graph
        )  # X: (nb_noeuds, 5)

        gt_mask = load_mask(gt_path, shape=image.shape[:2])  # (H, W) multi-label
        labels = build_node_labels(tree, gt_mask)  # y: (nb_noeuds,)

        X_list.append(attributes)
        y_list.append(labels)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    clf.fit(X, y)

    return clf
