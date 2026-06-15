import numpy as np

def load_mask(path: str) -> np.ndarray:
    """
    Charge un masque de segmentation ground truth de la DB

    Args:
        path: chemin vers le fichier masque

    Returns:
        mask: tableau d'entiers, un label par pixel
    """
    # TODO
    return


def compute_iou(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    """
    Calcule le IoU entre un masque prédit et le ground truth.

    Args:
        mask_gt:   masque ground truth
        mask_pred: masque prédit

    Returns:
        score IoU
    """
    # TODO
    return