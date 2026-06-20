import numpy as np
from skimage import io

IOU_THRESHOLD = 0.5

def load_mask(path: str) -> np.ndarray:
    """
    Charge un masque de segmentation ground truth de la DB

    Args:
        path: chemin vers le fichier masque

    Returns:
        mask: tableau d'entiers, un label par pixel
    """
    return io.imread(path, as_gray=True).astype(np.uint8)


def compute_iou(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    """
    Calcule le IoU entre un masque prédit et le ground truth.

    Args:
        mask_gt:   masque ground truth
        mask_pred: masque prédit

    Returns:
        score IoU
    """
    intersection = mask_gt & mask_pred
    union = mask_gt | mask_pred
    i_rate = np.count_nonzero(intersection)
    u_rate = np.count_nonzero(union)
    if u_rate == 0:
        return 1.0
    return i_rate/u_rate

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
    for gt,pred in zip(mask_gt_lst, mask_pred_lst):
        countPred = np.count_nonzero(pred)
        countGt = np.count_nonzero(gt)
        if countPred == 0 and countGt == 0:
            tn += 1
        elif compute_iou(pred, gt) >= IOU_THRESHOLD:
            tp += 1
        elif countPred == 0:
            fn += 1
        elif countGt == 0:
            fp += 1
        else:
            fp += 1
            fn += 1
    return tp, fp, fn, tn


def sensitivity(tp: int, fn: int) -> float:
    """
    Calcule la sensibilité (taux de vrais positifs, TPR) selon le protocole DIARETDB1.
    Proportion d'images anormales correctement classées comme anormales.

    Args:
        tp: nombre d'images anormales détectées comme anormales
        fn: nombre d'images anormales détectées comme normales

    Returns:
        SN = TP / (TP + FN)
    """
    return float(tp) / float(tp + fn)


def specificity(tn: int, fp: int) -> float:
    """
    Calcule la spécificité (taux de vrais négatifs, TNR) selon le protocole DIARETDB1.
    Proportion d'images normales correctement classées comme normales.

    Args:
        tn: nombre d'images normales détectées comme normales
        fp: nombre d'images normales détectées comme anormales

    Returns:
        SP = TN / (TN + FP)
    """
    return float(tn) / float(tn + fp)


def false_positive_rate(fp: int, tn: int) -> float:
    """
    Calcule le taux de faux positifs (FPR = 1 - spécificité).

    Args:
        fp: nombre d'images normales détectées comme anormales
        tn: nombre d'images normales détectées comme normales

    Returns:
        FPR = FP / (FP + TN)
    """
    return float(fp) / float(fp + tn)

def false_negative_rate(fn: int, tp: int) -> float:
    """
    Calcule le taux de faux négatifs (FNR = 1 - sensibilité).

    Args:
        fn: nombre d'images anormales détectées comme normales
        tp: nombre d'images anormales détectées comme anormales

    Returns:
        FNR = FN / (FN + TP)
    """
    return float(fn) / float(fn + tp)


def weighted_error_rate(fpr: float, fnr: float, R: float) -> float:
    """
    Calcule le taux d'erreur pondéré (WER) selon le protocole DIARETDB1.
    Le protocole DIARETDB1 trois valeurs : R=0.1, R=1, R=10
    Nous choisirons R=10 car nous voulons donner du poids au faux negatif
    pour ne pas rater de potentiels exudats.

    Args:
        fpr: taux de faux positifs
        fnr: taux de faux négatifs
        R:   ratio de coût entre FNR et FPR

    Returns:
        WER(R) = (FPR + R * FNR) / (1 + R)
    """
    return (fpr + R  * fnr ) / (1 + R)



def roc_curve_(labels:np.ndarray,ious:np.ndarray,threshold:float):
    tp = fp = fn = tn = 0
    for label,iou in zip(labels, ious):
        if iou >= threshold:
            if label == 1:
                tp += 1
            else:
                fp += 1
        else:
            if label == 1:
                fn += 1
            else:
                tn += 1
    return tp, fp, fn, tn



def roc_curve(mask_gt_lst: np.ndarray, mask_pred_lst: np.ndarray,number_thresholds:int):
    """
    Retourne la ROC curve
    Args:
        mask_gt_lst: liste de masque ground truth
        mask_pred_lst : liste de masque prédit
        number_thresholds: nombre de seuil à tester
    Returns:
        les points de la courbe (threshold,FPR,TPR)
    """
    labels = np.array([1 if np.count_nonzero(gt) else 0 for gt in mask_gt_lst])
    ious = np.array([compute_iou(gt,pred) for gt,pred in zip(mask_gt_lst,mask_pred_lst)])
    thresholds = np.linspace(0,1,number_thresholds)
    result = []

    for threshold in thresholds:
        tp, fp, fn, tn = roc_curve_(labels,ious,threshold)
        FPR  =false_positive_rate(fp,tn)
        TPR = sensitivity(tp,fn)
        result.append((threshold,FPR,TPR))
    return result


