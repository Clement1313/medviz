import higra as hg
import numpy as np


def predict(
    attributes: np.ndarray,
    clf,
    threshold: float = 0.5,
    max_area: int = 5000,
    min_area: int = 4000,
) -> np.ndarray:
    """
    classifie chaque noeud

    Args:
        attributes: matrice de features (nb_noeuds, nb_features)
        clf: classifieur sklearn entraîné
        threshold: seuil de probabilité pour la classe exsudat
        max_area: aire max en pixels
        min_area: aire min en pixels
    Returns:
        labels: tableau d'entiers
    """
    proba = clf.predict_proba(attributes)[:, 1]
    labels = (proba >= threshold).astype(np.int32)
    labels[attributes[:, 0] > max_area] = 0
    labels[attributes[:, 0] < min_area] = 0
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
    un par composante (label unique).

    Args:
        label_map: label de chacun des pixel / carte d'appartenance des pixels a une composante connexe

    Returns:
        liste de couple (label, masque binaire)
    """
    labels_unique = np.unique(mask)
    masks = []
    for label in labels_unique:
        binary_mask = mask == label
        masks.append((label, binary_mask))
    return masks
