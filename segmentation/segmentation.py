from skimage import io

try:
    from .maxtree import build_maxtree, compute_attributes
    from .predict import predict, cut_tree, get_connected_component_masks
    from .preprocessing import make_retina_mask, make_optic_disc_mask
except ImportError:  # execution "a plat" (depuis le dossier segmentation)
    from maxtree import build_maxtree, compute_attributes
    from predict import predict, cut_tree, get_connected_component_masks
    from preprocessing import make_retina_mask, make_optic_disc_mask


def segment(image_path: str, clf=None, threshold: float = 0.5) -> list:
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
    tree, altitudes, image_gray, graph, image_color = build_maxtree(image)

    # calcul des attributs
    attributes = compute_attributes(tree, altitudes, image_gray, graph, image_color)

    # zones a exclure : hors retine et disque optique (faux positifs)
    retina_mask = make_retina_mask(image)
    optic_disc_mask = make_optic_disc_mask(image, retina_mask)
    n_pixels = int(retina_mask.sum())

    # prediction (filtres d'aire relatifs a la taille de la retine)
    labels = predict(attributes, clf, threshold=threshold, n_pixels=n_pixels)

    # suppression des composantes selectionnees imbriquees
    mask = cut_tree(tree, labels)

    # on jette tout ce qui tombe hors retine ou dans le disque optique
    mask[~retina_mask] = 0
    mask[optic_disc_mask] = 0

    # masques binaires par composante connexe maintenue
    masks = get_connected_component_masks(mask)

    return masks
