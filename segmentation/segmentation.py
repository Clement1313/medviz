from skimage import io

from maxtree import build_maxtree, compute_attributes
from predict import predict, cut_tree, get_connected_component_masks


def segment(image_path: str, clf=None) -> list:
    """
    Pipeline complete de segmentationd des exhudats.

    Args:
        image_path: chemin vers l'image
        clf: classifieur sklearn entraîné

    Returns:
        liste de (label, masque_binaire)
    """
    image = io.imread(image_path)

    # maxtree
    tree, altitudes, image_gray = build_maxtree(image)

    # calcul des attributs
    attributes = compute_attributes(tree, altitudes, image_gray)

    # prediction
    labels = predict(attributes, clf)

    # suppression des composantes selectionnees imbriquees
    mask = cut_tree(tree, labels)

    # masques binaires par composante connexe maintenue
    masks = get_connected_component_masks(mask)

    return masks
