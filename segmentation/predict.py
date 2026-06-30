import higra as hg
import numpy as np
from scipy import ndimage


def predict(
    attributes: np.ndarray,
    clf,
    threshold: float = 0.5,
    n_pixels: int = None,
    min_area_frac: float = 3e-6,
    max_area_frac: float = 2e-2,
) -> np.ndarray:
    """
    classifie chaque noeud

    Les bornes d'aire sont exprimees en FRACTION du nombre de pixels de la
    retine (et non en pixels absolus). Deux raisons :
      - independance a la resolution : une image uploadee dans une autre taille
        que celle du dataset garde des seuils coherents ;
      - on evite l'ancien bug ou la fenetre [min_area, max_area] = [4000, 5000]
        ne laissait passer qu'une bande de 1000 px, supprimant la quasi-totalite
        des exsudats (petits exsudats durs en particulier).

    Args:
        attributes: matrice de features (nb_noeuds, nb_features), colonne 0 = aire
        clf: classifieur sklearn entraîné
        threshold: seuil de probabilité pour la classe exsudat
        n_pixels: nombre de pixels de la retine (si None, pas de filtre d'aire)
        min_area_frac: aire min, en fraction de n_pixels (retire le bruit)
        max_area_frac: aire max, en fraction de n_pixels (retire les grosses
                       structures brillantes, ex. residu de disque optique)
    Returns:
        labels: tableau d'entiers
    """
    proba = clf.predict_proba(attributes)[:, 1]
    labels = (proba >= threshold).astype(np.int32)

    if n_pixels is not None:
        min_area = max(5, int(min_area_frac * n_pixels))
        max_area = int(max_area_frac * n_pixels)
        labels[attributes[:, 0] < min_area] = 0
        labels[attributes[:, 0] > max_area] = 0

    return labels


def cut_tree(tree, labels: np.ndarray) -> np.ndarray:
    """
    selection des composantes connexes que l'on garde
    si un noeud est predit, sa descendance, peu importe
    sa prediction, fait partie de la meme partie segmentee
    cela permet d'eviter d'avoir des prediction imbriquees

    Args:
        tree: maxtree higra
        labels: tableau (nb_noeuds,) de labels entiers
                0 = non marqué / fond

    Returns:
        mask: image segmentée (H, W) avec un label par pixel
    """
    parents = tree.parents()

    propagated = labels.copy()
    for node in tree.root_to_leaves_iterator():
        parent = parents[node]
        if node != tree.root() and propagated[parent] != 0:
            propagated[node] = propagated[parent]

    mask = hg.reconstruct_leaf_data(tree, propagated)

    return mask


def get_connected_component_masks(mask: np.ndarray) -> list:
    """
    Convertit un masque de labels en une liste de masques binaires,
    un par composante CONNEXE (région de pixels adjacents partageant
    le même label), et non simplement par valeur de label.

    Args:
        mask: label de chacun des pixels / carte d'appartenance

    Returns:
        liste de couples (label, masque binaire)
    """
    labels_unique = np.unique(mask)
    masks = []
    for label in labels_unique:
        binary_mask = mask == label
        connected_components, n_components = ndimage.label(binary_mask)

        for component_id in range(1, n_components + 1):
            component_mask = connected_components == component_id
            masks.append((label, component_mask))

    return masks
