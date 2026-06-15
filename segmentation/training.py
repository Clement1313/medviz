import numpy as np
from skimage import io

from evaluation import load_mask, compute_iou
from segmentation import segment

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
    h, w  = image.shape[:2]
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
    # TODO
    return clf
