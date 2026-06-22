import xml.etree.ElementTree as ET

import numpy as np
from skimage import draw

# taille standard des images DiaRetDB1 (hauteur, largeur)
DIARETDB1_SHAPE = (1152, 1500)


def _coords(text: str) -> tuple:
    """'x,y' (colonne,ligne) -> (x, y) entiers"""
    x, y = text.split(",")
    return int(x), int(y)


def _region_to_indices(region, shape: tuple) -> tuple:
    """
    Rasterise une region (cercle / ellipse / polygone) en indices (rr, cc).
    Le type de forme depend du fichier, on dispatch sur le tag.
    """
    tag = region.tag
    if tag == "circleregion":
        cx, cy = _coords(region.find("centroid/coords2d").text)
        r = int(region.find("radius").text)
        return draw.disk((cy, cx), r, shape=shape)

    if tag == "ellipseregion":
        cx, cy = _coords(region.find("centroid/coords2d").text)
        rx = int(region.find('radius[@direction="x"]').text)
        ry = int(region.find('radius[@direction="y"]').text)
        angle = region.find("angle")
        rotation = np.deg2rad(float(angle.text)) if angle is not None else 0.0
        return draw.ellipse(cy, cx, ry, rx, shape=shape, rotation=rotation)

    if tag == "polygonregion":
        # coords2d enfants directs = sommets (le centroid est sous <centroid>)
        pts = [_coords(c.text) for c in region.findall("coords2d")]
        rows = [y for _, y in pts]
        cols = [x for x, _ in pts]
        return draw.polygon(rows, cols, shape=shape)

    raise ValueError(f"Type de region non gere : {tag}")


def load_mask(path: str, shape: tuple = DIARETDB1_SHAPE) -> np.ndarray:
    """
    Charge le ground truth depuis un XML d'annotation DiaRetDB1 et rasterise
    TOUS les marquages dont le markingtype contient 'exudates'
    (Hard_exudates / Soft_exudates). Chaque exsudat recoit un label entier
    distinct (1, 2, 3, ...).

    Args:
        path:  chemin vers le fichier XML d'annotation
        shape: (hauteur, largeur) du masque a produire

    Returns:
        mask: (H, W) int32, un label par exsudat, 0 = fond
    """
    root = ET.parse(path).getroot()
    mask = np.zeros(shape, dtype=np.int32)

    label = 0
    for marking in root.iter("marking"):
        mtype = marking.find("markingtype")
        if mtype is None or mtype.text is None or "exudates" not in mtype.text.lower():
            continue

        label += 1
        # la region est le 1er enfant dont le tag se termine par 'region'
        region = next(c for c in marking if c.tag.endswith("region"))
        rr, cc = _region_to_indices(region, shape)
        mask[rr, cc] = label

    return mask


def compute_iou(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    """
    Calcule le IoU entre un masque prédit et le ground truth.

    Args:
        mask_gt:   masque ground truth
        mask_pred: masque prédit

    Returns:
        score IoU
    """
    intersection = (mask_gt > 0) & (mask_pred > 0)
    union = (mask_gt > 0) | (mask_pred > 0)
    i_rate = np.count_nonzero(intersection)
    u_rate = np.count_nonzero(union)
    if u_rate == 0:
        return 1.0
    return i_rate / u_rate


def confusionCounts(mask_gt_lst: list, mask_pred_lst: list, min_pixels: int = 1):
    """
    Calcule la matrice de confusion au niveau IMAGE, selon le protocole
    DiaRetDB1 (evaluation image-based, pas de recouvrement spatial).

    Une image est anormale (cote GT) si elle contient au moins un marquage
    d'exsudat. La methode classe une image comme anormale si son score
    (= nombre de pixels predits exsudat) atteint min_pixels. Aucun appariement
    detection<->GT n'est fait : on compare juste les deux labels presence/absence.

    Args:
        mask_gt_lst: liste de masques ground truth
        mask_pred_lst: liste de masques predits
        min_pixels: seuil sur le score-image (nb de pixels flaggés) au-dessus
            duquel la methode declare l'image anormale

    Returns:
        La matrice de confusion du modèle (VP, FP, FN, VN) sur tout le dataset
    """
    tp = fp = fn = tn = 0
    for gt, pred in zip(mask_gt_lst, mask_pred_lst):
        gt_abnormal = np.count_nonzero(gt) > 0
        pred_abnormal = np.count_nonzero(pred) >= min_pixels
        if gt_abnormal and pred_abnormal:
            tp += 1
        elif not gt_abnormal and not pred_abnormal:
            tn += 1
        elif pred_abnormal:  # GT normale, methode anormale
            fp += 1
        else:  # GT anormale, methode normale
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
    return (fpr + R * fnr) / (1 + R)


def roc_curve_(labels: np.ndarray, scores: np.ndarray, threshold: float):
    tp = fp = fn = tn = 0
    for label, score in zip(labels, scores):
        if score >= threshold:  # image declaree anormale par la methode
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


def roc_curve(mask_gt_lst: list, mask_pred_lst: list, number_thresholds: int):
    """
    Retourne la ROC curve (protocole DiaRetDB1, image-based).

    On balaie le seuil sur le score-image (= nombre de pixels predits exsudat),
    comme dans le papier : une image est declaree anormale si son score atteint
    le seuil. Aucun recouvrement spatial n'intervient.

    Args:
        mask_gt_lst: liste de masque ground truth
        mask_pred_lst : liste de masque prédit
        number_thresholds: nombre de seuil à tester
    Returns:
        les points de la courbe (threshold,FPR,TPR)
    """
    labels = np.array([1 if np.count_nonzero(gt) else 0 for gt in mask_gt_lst])
    scores = np.array([np.count_nonzero(pred) for pred in mask_pred_lst])
    max_score = scores.max() if len(scores) else 0
    thresholds = np.linspace(0, max_score, number_thresholds)
    result = []

    for threshold in thresholds:
        tp, fp, fn, tn = roc_curve_(labels, scores, threshold)
        FPR = false_positive_rate(fp, tn)
        TPR = sensitivity(tp, fn)
        result.append((threshold, FPR, TPR))
    return result
