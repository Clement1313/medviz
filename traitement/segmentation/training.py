import os

import higra as hg
import numpy as np
from skimage import io
from skimage.filters import threshold_otsu

try:
    from .evaluation import load_mask, load_mask_union, compute_iou
    from .segmentation import segment
    from .maxtree import build_maxtree, compute_attributes
    from .preprocessing import make_retina_mask
except ImportError:  # execution "a plat" (run_training depuis le dossier segmentation)
    from evaluation import load_mask, load_mask_union, compute_iou
    from segmentation import segment
    from maxtree import build_maxtree, compute_attributes
    from preprocessing import make_retina_mask


def refine_gt_by_intensity(gt_mask: np.ndarray, image_gray: np.ndarray) -> np.ndarray:
    """
    Resserre un masque GT grossier autour des vrais exsudats par l'intensite.

    Les annotations DiaRetDB1 sont de larges cercles couvrant ~10x la surface
    reelle de l'exsudat : sans correction, la majorite des pixels marques
    "exsudat" sont en fait de la retine normale, ce qui apprend au classifieur a
    detecter n'importe quelle structure brillante -> precision catastrophique.

    Un exsudat etant clair, on separe (Otsu) le brillant du fond UNIQUEMENT
    parmi les pixels situes dans les cercles GT, sur le meme gris rehausse
    (vert + CLAHE) que celui du maxtree. On ne garde positifs que les pixels a la
    fois dans un cercle ET au-dessus du seuil.

    Args:
        gt_mask:    (H, W) int, un label par region GT, 0 = fond
        image_gray: (H, W) gris rehausse (sortie de build_maxtree)

    Returns:
        gt_mask resserre (memes labels, mais restreint aux pixels brillants)
    """
    inside = gt_mask > 0
    if not inside.any():
        return gt_mask

    vals = image_gray[inside]
    if vals.max() == vals.min():
        return gt_mask  # rien a separer

    thr = threshold_otsu(vals)
    bright = image_gray > thr  # convention skimage : separation par image > seuil

    refined = np.where(inside & bright, gt_mask, 0)
    # garde-fou : si Otsu vide tout (ex. cercles peu contrastes), on garde le GT brut
    if not refined.any():
        return gt_mask
    return refined


def build_node_labels(
    tree, gt_mask: np.ndarray, retina_mask: np.ndarray = None, threshold: float = 0.5
) -> np.ndarray:
    """
    Convertit un masque GT (niveau pixel) en un label binaire par noeud du maxtree.

    Les feuilles du maxtree correspondent aux pixels (ordre reshape(-1)). On
    accumule le GT binarise des feuilles vers tous les noeuds, puis un noeud est
    positif si la fraction de ses pixels appartenant a un exsudat depasse le seuil.

    Args:
        tree:      maxtree higra
        gt_mask:   masque ground truth (H, W), >0 sur les exsudats (multi-label)
        retina_mask: masque booléen de la rétine
        threshold: fraction minimale de pixels exsudat pour qu'un noeud soit positif

    Returns:
        labels: (nb_noeuds,) int32, 1 = exsudat, 0 = fond, -1 = hors retine
    """
    # 1 si exudat 0 sinon (plus de distinctions entre les exudats)
    leaf_gt = (gt_mask.reshape(-1) > 0).astype(np.float64)

    # nb de pixels exsudat par noeud
    positive_count = hg.accumulate_sequential(tree, leaf_gt, hg.Accumulators.sum)

    # nb de pixels total par noeud
    area = hg.attribute_area(tree)
    coverage = (
        positive_count / area
    )  # proportion de pixel masques comme exsudat dans le gt dans la composante connexe
    labels = (coverage > threshold).astype(np.int32)

    if retina_mask is not None:
        leaf_retina = retina_mask.reshape(-1).astype(np.float64)  # flatten en place
        retina_count = hg.accumulate_sequential(tree, leaf_retina, hg.Accumulators.sum)
        labels[
            retina_count == 0
        ] = -1  # exclu les noeuds qui ne sont pas dans la retine (fond)

    return labels


def evaluate(image_path: str, gt_mask_path: str, clf, threshold: float = 0.5) -> float:
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
    masks = segment(image_path, clf, threshold=threshold)

    # reconstruction masque predit
    image = io.imread(image_path)
    h, w = image.shape[:2]
    mask_pred = np.zeros((h, w), dtype=np.int32)
    for label, binary_mask in masks:
        mask_pred[binary_mask] = label

    mask_gt = load_mask(gt_mask_path)

    score = compute_iou(mask_gt, mask_pred)

    return score


def train(
    image_paths: list,
    ground_truth_mask_paths: list,
    clf,
    neg_ratio: int = 20,
    max_neg_no_positive: int = 5000,
    use_union: bool = False,
    annotators: tuple = ("01", "02", "03", "04"),
):
    """
    Entraîne le classifieur sur une liste d'images annotées.

    Deux choix d'echantillonnage, cles pour la qualite :
      - on garde TOUS les noeuds positifs (rares) et on sous-echantillonne les
        negatifs (un tirage uniforme jetait l'essentiel du signal positif) ;
      - les negatifs sont des HARD NEGATIVES : on les tire ponderes par leur
        intensite (altitude du noeud max-tree). Un tirage uniforme est domine par
        du fond sombre "facile" ; le modele n'apprend alors jamais qu'une
        structure brillante peut NE PAS etre un exsudat (reflets, bords de
        vaisseaux, texture rehaussee par CLAHE) -> precision catastrophique.
        Comme un negatif est par definition hors des cercles GT, ce tirage revient
        a cibler les "structures brillantes hors GT" (le masque sert donc bien,
        mais uniquement a l'entrainement -> aucune fuite vers l'evaluation).

    Args:
        image_paths: liste de chemins vers les images
        ground_truth_mask_paths: liste de chemins vers les masques GT correspondants
        clf: classifieur sklearn
        neg_ratio: nb de negatifs conserves par positif (par image)
        max_neg_no_positive: nb de negatifs conserves pour une image sans exsudat

    Returns:
        clf entraîné
    """
    X_list, y_list = [], []
    rng = np.random.default_rng(42)

    n_total = len(image_paths)
    for i, (image_path, gt_path) in enumerate(
        zip(image_paths, ground_truth_mask_paths), start=1
    ):
        print(f"  features {i}/{n_total} : {os.path.basename(image_path)}", flush=True)
        image = io.imread(image_path)

        tree, altitudes, image_gray, graph, image_color = build_maxtree(image)
        attributes = compute_attributes(
            tree, altitudes, image_gray, graph, image_color
        )  # X: (nb_noeuds, nb_features)

        # GT : union des annotateurs (apprend aussi les lesions faibles vues par
        # d'autres experts que le 01) ou annotateur 01 seul
        if use_union:
            gt_dir = os.path.dirname(gt_path)
            stem = os.path.basename(image_path).replace(".png", "")
            gt_mask = load_mask_union(
                gt_dir, stem, shape=image.shape[:2], annotators=annotators
            )
        else:
            gt_mask = load_mask(gt_path, shape=image.shape[:2])  # (H, W) multi-label
        # resserre les cercles GT autour des pixels reellement brillants (exsudats)
        gt_mask = refine_gt_by_intensity(gt_mask, image_gray)
        retina_mask = make_retina_mask(image)
        labels = build_node_labels(tree, gt_mask, retina_mask)  # y: (nb_noeuds,)

        # exclure les noeuds hors retine
        valid = labels >= 0
        attributes = attributes[valid]
        labels = labels[valid]
        node_altitudes = altitudes[valid]  # intensite par noeud (pour hard negatives)

        # subsampling : on garde tous les positifs, on echantillonne les negatifs
        pos_idx = np.flatnonzero(labels == 1)
        neg_idx = np.flatnonzero(labels == 0)

        if len(pos_idx) > 0:
            max_neg = len(pos_idx) * neg_ratio
        else:
            max_neg = max_neg_no_positive

        if len(neg_idx) > max_neg:
            # hard negatives : proba de tirage croissante avec l'intensite du noeud
            weights = node_altitudes[neg_idx].astype(np.float64) + 1.0
            probs = weights / weights.sum()
            neg_idx = rng.choice(neg_idx, size=max_neg, replace=False, p=probs)

        keep = np.concatenate([pos_idx, neg_idx])
        attributes = attributes[keep]
        labels = labels[keep]

        X_list.append(attributes)
        y_list.append(labels)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    clf.fit(X, y)

    return clf
