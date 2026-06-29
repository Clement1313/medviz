import numpy as np
from scipy.ndimage import median_filter
from skimage import color, exposure, filters


def to_enhanced_gray(image: np.ndarray, median_size: int = 3) -> np.ndarray:
    """
    Convertit une image de fond d'oeil en niveaux de gris "rehausses" adaptes a
    la detection d'exsudats.

    Trois corrections par rapport a un simple rgb2gray :
      - on prend le CANAL VERT : les exsudats y ressortent nettement mieux que
        sur le gris RGB qui melange le rouge (souvent sature sur un fond d'oeil)
        et ecrase donc le contraste exsudat / retine.
      - on debruite par un filtre MEDIAN leger AVANT CLAHE : sans lui, CLAHE
        amplifie le bruit des images de moindre qualite et fabrique des dizaines
        de petites taches brillantes parasites, indiscernables de vrais petits
        exsudats par le classifieur. Le median supprime ces taches isolees
        (1-2 px) tout en preservant les vrais exsudats (plus larges) et leurs
        bords (contrairement a un flou).
      - on egalise l'illumination via CLAHE (contraste adaptatif local) afin de
        reduire la forte variabilite d'eclairage d'un cliche a l'autre.

    Args:
        image: image RGB (H, W, 3) ou deja en niveaux de gris (H, W)
        median_size: taille du filtre median (0 ou 1 pour le desactiver)

    Returns:
        gris rehausse uint8 (H, W), valeurs dans [0, 255]
    """
    if image.ndim == 3:
        green = image[:, :, 1].astype(np.float64)
    else:
        green = image.astype(np.float64)

    if green.max() > 0:
        green = green / green.max()

    # debruitage : enleve les taches de bruit sous-resolution que CLAHE amplifierait
    if median_size and median_size > 1:
        green = median_filter(green, size=median_size)

    # CLAHE : normalise le contraste localement -> robustesse a l'illumination
    equalized = exposure.equalize_adapthist(green, clip_limit=0.01)
    return (equalized * 255).astype(np.uint8)


def make_retina_mask(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Genere un masque booleen du champ visuel (retine) par seuillage.
    Les pixels hors retine sont quasi-noirs sur les 3 canaux.

    Args:
        image: image RGB (H, W, 3) ou niveaux de gris (H, W)
        threshold: seuil d'intensite separant fond noir et retine

    Returns:
        masque booleen (H, W), True a l'interieur de la retine
    """
    if image.ndim == 3:
        gray = (color.rgb2gray(image) * 255).astype(np.uint8)
    else:
        gray = image
    return gray > threshold


def make_optic_disc_mask(
    image: np.ndarray,
    retina_mask: np.ndarray = None,
    radius_frac: float = 0.12,
) -> np.ndarray:
    """
    Approxime le disque optique par un masque circulaire.

    Le disque optique est la region large la plus brillante de la retine ; sans
    traitement il est systematiquement confondu avec un gros exsudat (meme
    aspect : tache claire et compacte). On le retire donc en amont.

    Heuristique : on lisse fortement l'intensite (pour viser une zone large et
    non un pixel isole), on prend le point le plus brillant DANS la retine comme
    centre, et on masque un disque de rayon proportionnel a la taille de l'image
    (donc independant de la resolution).

    Args:
        image: image RGB (H, W, 3) ou niveaux de gris (H, W)
        retina_mask: masque booleen de la retine (recommande, sinon le bord
                     brillant peut tromper la recherche du centre)
        radius_frac: rayon du disque, en fraction de la plus petite dimension

    Returns:
        masque booleen (H, W), True a l'interieur du disque optique
    """
    if image.ndim == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image.astype(np.float64)
        if gray.max() > 0:
            gray = gray / gray.max()

    h, w = gray.shape

    # lissage fort : on cherche la zone large la plus brillante, pas un point
    sigma = max(h, w) * 0.02
    smooth = filters.gaussian(gray, sigma=sigma)

    if retina_mask is not None:
        # on ignore tout ce qui est hors retine pour le choix du centre
        smooth = np.where(retina_mask, smooth, -1.0)

    cy, cx = np.unravel_index(int(np.argmax(smooth)), smooth.shape)

    radius = int(min(h, w) * radius_frac)
    yy, xx = np.ogrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
