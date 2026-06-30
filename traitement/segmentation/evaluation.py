import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy import ndimage
from skimage import draw

IOU_THRESHOLD = 0.5

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


def load_mask_union(
    gt_dir: str,
    stem: str,
    shape: tuple = DIARETDB1_SHAPE,
    annotators: tuple = ("01", "02", "03", "04"),
) -> np.ndarray:
    """
    Ground truth plus complet : UNION des marquages exsudats de TOUS les
    annotateurs DiaRetDB1 (et non du seul annotateur 01).

    DiaRetDB1 est annote par 4 experts ; un exsudat vu par l'un mais pas par
    l'autre, si on ne lit que l'annotateur 01, devient un faux "FP" (l'image est
    declaree saine a tort). En unionnant, une image n'est consideree saine que si
    AUCUN expert n'y a marque d'exsudat -> metriques niveau image honnetes.

    Les marquages qui se recouvrent (meme lesion vue par plusieurs experts) sont
    fusionnes par etiquetage en composantes connexes, pour ne pas gonfler n_gt.

    Args:
        gt_dir:     dossier des annotations
        stem:       nom de base de l'image (sans extension)
        shape:      (H, W) du masque
        annotators: suffixes d'annotateurs a unionner

    Returns:
        mask: (H, W) int32, un label par lesion (union), 0 = fond
    """
    union = np.zeros(shape, dtype=bool)
    for a in annotators:
        path = os.path.join(gt_dir, f"{stem}_{a}.xml")
        if os.path.exists(path):
            union |= load_mask(path, shape) > 0

    labeled, _ = ndimage.label(union)
    return labeled.astype(np.int32)


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


def lesion_detection_counts(
    gt_mask: np.ndarray, pred_components: list, min_overlap: int = 1
) -> tuple:
    """
    Matching detection au NIVEAU LESION (et non IoU pixel a pixel).

    Le IoU pixel est inadapte ici : les annotations DiaRetDB1 sont de larges
    cercles / ellipses / polygones couvrant largement plus que la surface reelle
    de l'exsudat. Une detection fine et correcte aurait donc un IoU tres faible.
    On compte donc des detections, pas des recouvrements exacts :

      - une region GT est "detectee" si au moins une composante predite la
        recouvre (>= min_overlap pixels) ;
      - une composante predite est un vrai positif si elle recouvre au moins une
        region GT, un faux positif sinon.

    Args:
        gt_mask: (H, W) int, un label entier distinct par region GT, 0 = fond
        pred_components: liste de masques binaires predits (composantes exsudat)
        min_overlap: nb min de pixels de recouvrement pour valider un hit

    Returns:
        (n_gt, tp_lesion, fn_lesion, tp_det, fp_det)
          n_gt:       nb de regions GT
          tp_lesion:  nb de regions GT detectees
          fn_lesion:  nb de regions GT manquees
          tp_det:     nb de composantes predites qui tombent sur une region GT
          fp_det:     nb de composantes predites hors de toute region GT
    """
    gt_labels = np.unique(gt_mask)
    gt_labels = gt_labels[gt_labels != 0]
    n_gt = int(gt_labels.size)

    detected = set()
    tp_det = fp_det = 0

    for comp in pred_components:
        vals, counts = np.unique(gt_mask[comp], return_counts=True)
        hit_labels = vals[(vals != 0) & (counts >= min_overlap)]
        if hit_labels.size > 0:
            tp_det += 1
            detected.update(hit_labels.tolist())
        else:
            fp_det += 1

    tp_lesion = len(detected)
    fn_lesion = n_gt - tp_lesion
    return n_gt, tp_lesion, fn_lesion, tp_det, fp_det


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
    for gt, pred in zip(mask_gt_lst, mask_pred_lst):
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
    return (fpr + R * fnr) / (1 + R)


def roc_curve_(labels: np.ndarray, ious: np.ndarray, threshold: float):
    tp = fp = fn = tn = 0
    for label, iou in zip(labels, ious):
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


def roc_curve(
    mask_gt_lst: np.ndarray, mask_pred_lst: np.ndarray, number_thresholds: int
):
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
    ious = np.array(
        [compute_iou(gt, pred) for gt, pred in zip(mask_gt_lst, mask_pred_lst)]
    )
    thresholds = np.linspace(0, 1, number_thresholds)
    result = []

    for threshold in thresholds:
        tp, fp, fn, tn = roc_curve_(labels, ious, threshold)
        FPR = false_positive_rate(fp, tn)
        TPR = sensitivity(tp, fn)
        result.append((threshold, FPR, TPR))
    return result
