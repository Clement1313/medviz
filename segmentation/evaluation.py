import numpy as np

IOU_THRESHOLD = 0.5

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

def confusionCounts(mask_gt_lst: np.ndarray, mask_pred_lst: np.ndarray):
        """
        Calcule la matrice de confusion 

        Args:
            mask_gt_lst: liste de masque ground truth
            mask_pred_lst : liste de masque prédit

        Returns:
            La matrice de confusion du modèle (VP, FP, FN, VN) sur tout le dataset
        """
        tp = fp = fn = tn = 0
        for pred, gt in zip(mask_gt_lst, mask_pred_lst):
            countPred = np.count_nonzero(pred)
            countGt = np.count_nonzero(gt)
            if countPred == 0 and countGt == 0:
                tn += 1
            elif countPred == 0:
                fn += 1
            elif countGt == 0:
                fp += 1
            elif compute_iou(pred, gt) >= IOU_THRESHOLD:
                tp += 1
            else:
                fp += 1
                fn += 1
        return tp, fp, fn, tn