import higra as hg
import numpy as np
from scipy.ndimage import uniform_filter

try:
    from .preprocessing import to_enhanced_gray
except ImportError:  # execution "a plat" (run_training depuis le dossier segmentation)
    from preprocessing import to_enhanced_gray


def build_maxtree(image: np.ndarray) -> tuple:
    """
    construit le maxtree d'une image

    Args:
        image: tableau 2d d'une image en niveau de gris ou d'une image rgb

    Returns:
        (tree, altitudes, image_gray, graph, image_color)
        tree, altitudes, et graph sont des objets manipules par la librairie higra ;
        image_gray est le gris rehausse servant a construire l'arbre ; image_color
        est l'image RGB d'origine (ou None si entree deja en niveaux de gris),
        conservee pour les attributs de couleur.
    """
    image_color = image if image.ndim == 3 else None

    # canal vert + CLAHE : meilleur contraste exsudat/fond et robustesse a
    # l'illumination (cf. preprocessing.to_enhanced_gray)
    gray = to_enhanced_gray(image)

    graph = hg.get_4_adjacency_graph(gray.shape)
    tree, altitudes = hg.component_tree_max_tree(graph, gray)

    return tree, altitudes, gray, graph, image_color


def _node_mean(maxtree, area: np.ndarray, leaf_values: np.ndarray) -> np.ndarray:
    """Moyenne, par noeud, d'une valeur definie au niveau pixel (feuilles)."""
    total = hg.accumulate_sequential(
        maxtree, leaf_values.reshape(-1).astype(np.float64), hg.Accumulators.sum
    )
    return total / np.maximum(area, 1)


def compute_attributes(
    maxtree, altitudes: np.ndarray, image: np.ndarray, graph, image_color=None
) -> np.ndarray:
    """
    calcule les attributs de chaque noeud du maxtree

    Args:
        maxtree: maxtree
        altitudes: altitudes
        image: image en niveaux de gris (gris rehausse)
        graph: graphe d'adjacence des pixels
        image_color: image RGB d'origine (pour les attributs de couleur), ou None

    Returns:
        attributes: attributs de chacun des noeuds de l'arbre. La colonne 0 reste
                    l'aire (predict.py s'en sert pour le filtre d'aire).
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

    # contraste local : intensite moyenne du noeud par rapport au fond local
    # (image - fond moyen). Un exsudat ressort fortement de son voisinage, beaucoup
    # de fausses structures brillantes non. Le fond est estime par un filtre
    # moyenneur (box) : separable a somme glissante, donc bien plus rapide qu'un
    # flou gaussien a grand rayon.
    gimg = image.astype(np.float64)
    window = int(max(image.shape) * 0.05) | 1  # fenetre impaire ~5% de l'image
    background = uniform_filter(gimg, size=window, mode="reflect")
    local_contrast = _node_mean(maxtree, area, gimg - background)

    # attributs de couleur : un exsudat est jaunatre (R et G eleves, B plus bas),
    # contrairement aux reflets (blancs, B eleve) et aux vaisseaux (rouges).
    if image_color is not None:
        rgb = image_color[:, :, :3]
    else:
        rgb = np.stack([image] * 3, axis=-1)  # fallback niveaux de gris
    mean_r = _node_mean(maxtree, area, rgb[:, :, 0])
    mean_g = _node_mean(maxtree, area, rgb[:, :, 1])
    mean_b = _node_mean(maxtree, area, rgb[:, :, 2])

    attributes = np.stack(
        [
            area,
            volume,
            compactness,
            contour_strength,
            dynamics,
            local_contrast,
            mean_r,
            mean_g,
            mean_b,
        ],
        axis=1,
    )
    return attributes
