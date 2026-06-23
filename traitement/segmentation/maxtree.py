import higra as hg
import numpy as np
import skimage as sk


def build_maxtree(image: np.ndarray) -> tuple:
    """
    construit le maxtree d'une image

    Args:
        image: tableau 2d d'une image en niveau de gris ou d'une image rgb

    Returns:
        (tree, altitudes, image_gray, graph)
        tree, altitudes, et graph sont des objets manipules par la librairie higra,
        utilises lors pour le calcul d'attributs
    """
    # si rgb, conversion en gray level
    if image.ndim == 3:
        image = (sk.color.rgb2gray(image) * 255).astype(np.uint8)

    graph = hg.get_4_adjacency_graph(image.shape)
    tree, altitudes = hg.component_tree_max_tree(graph, image)

    return tree, altitudes, image, graph


def compute_attributes(
    maxtree, altitudes: np.ndarray, image: np.ndarray, graph
) -> np.ndarray:
    """
    calcule les attributs de chaque noeud du maxtree

    Args:
        maxtree: maxtree
        altitudes: altitudes
        image: image en niveaux de gris

    Returns:
        attributes: attributs de chacun des noeuds de l'arbre
    """
    # attributs geometriques
    area = hg.attribute_area(maxtree)  # nombre de pixels
    volume = hg.attribute_volume(
        maxtree, altitudes
    )  # somme des poids des pixels de la composante connexe (CC)
    contour = hg.attribute_contour_length(maxtree)  # perimetre de la CC
    compactness = hg.attribute_compactness(
        maxtree, area=area, contour_length=contour
    )  # aire divisee par le carre du perimetre
    edge_weights = hg.weight_graph(
        graph, image, hg.WeightFunction.L1
    )  # graph represente les liens entres nos pixels (4 connexite chez nous). cette fonction pondere les arretes par la difference d'intensite entre les pixels
    contour_strength = hg.attribute_contour_strength(
        maxtree, edge_weights
    )  # nettete des contours de la CC

    # attributs sur l'intensite
    dynamics = hg.attribute_dynamics(
        maxtree, altitudes
    )  # difference entre la valueur du pixel canonique de la CC et celle de son pere

    attributes = np.stack(
        [area, volume, compactness, contour_strength, dynamics], axis=1
    )
    return attributes
