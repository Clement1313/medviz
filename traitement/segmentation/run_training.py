import os
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))

from training import train


# racine du projet (deux niveaux au-dessus de segmentation/)
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
# dataset DiaRetDB1 place a la racine du projet (contient images/ et groundtruth/)
DB_DIR = os.path.join(PROJECT_ROOT, "database")
IMG_DIR = os.path.join(DB_DIR, "images")
GT_DIR = os.path.join(DB_DIR, "groundtruth")
# le modele est ecrit la ou l'API le charge (traitement/app/clf.joblib)
MODEL_OUT = os.path.join(os.path.dirname(__file__), "..", "app", "clf.joblib")

LIMIT = None  # mettre une limite seulement si on veut tester rapidement
# nb de negatifs gardes par positif. 10 = iterations rapides ;
# 20 = config finale (plus de hard negatives -> moins de faux positifs).
NEG_RATIO = 20
N_ESTIMATORS = 100  # baisser (ex. 50) pour un fit plus rapide en phase de diagnostic
# taille min d'une feuille : limite la taille du modele (sinon plusieurs Go) et
# regularise. Monter (ex. 100) si le clf.joblib est encore trop gros.
MIN_SAMPLES_LEAF = 50
# GT d'entrainement : union des 4 annotateurs (apprend les lesions faibles vues
# par d'autres experts que le 01 -> meilleure sensibilite). Doit matcher l'eval.
USE_ANNOTATOR_UNION = True
ANNOTATORS = ("01", "02", "03", "04")


def load_dataset_paths(img_dir: str, gt_dir: str, annotator: str = "01") -> tuple:
    """
    Retourne les listes de chemins (images, gt) appariés.
    Cherche les GT sous la forme <stem>_<annotator>.xml.
    """
    image_paths, gt_paths = [], []
    for fname in sorted(f for f in os.listdir(img_dir) if f.endswith(".png")):
        stem = fname.replace(".png", "")
        gt = os.path.join(gt_dir, f"{stem}_{annotator}.xml")
        if os.path.exists(gt):
            image_paths.append(os.path.join(img_dir, fname))
            gt_paths.append(gt)
    return image_paths, gt_paths


# LOAD DATASET
image_paths, gt_paths = load_dataset_paths(IMG_DIR, GT_DIR)

if LIMIT:
    image_paths = image_paths[:LIMIT]
    gt_paths = gt_paths[:LIMIT]

print(f"Images : {len(image_paths)} (LIMIT={LIMIT})")

# SPLIT TRAIN/TEST IMAGES
idx_train, idx_test = train_test_split(
    range(len(image_paths)), test_size=0.3, random_state=42
)
train_imgs = [image_paths[i] for i in idx_train]
train_gts = [gt_paths[i] for i in idx_train]
test_imgs = [image_paths[i] for i in idx_test]
test_gts = [gt_paths[i] for i in idx_test]

print(f"Train : {len(train_imgs)} | Test : {len(test_imgs)}")

# TRAINING
# n_jobs=-1 : entraine les arbres sur tous les coeurs (8 sur un M2) -> fit bien
# plus rapide, sans effet sur la qualite.
# min_samples_leaf : empeche les arbres de pousser jusqu'a 1 echantillon/feuille.
# Sans lui, sur des millions d'echantillons le modele atteint plusieurs Go ;
# il regularise aussi (moins d'overfit -> potentiellement moins de FP).
clf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
    min_samples_leaf=MIN_SAMPLES_LEAF,
)
print("Training...")
clf = train(
    train_imgs,
    train_gts,
    clf,
    neg_ratio=NEG_RATIO,
    use_union=USE_ANNOTATOR_UNION,
    annotators=ANNOTATORS,
)
joblib.dump(clf, MODEL_OUT)
print(f"Model saved in {MODEL_OUT}")
