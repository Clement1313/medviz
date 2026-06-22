import os
import sys
import joblib
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))

from training import train


DB_DIR = os.path.join(os.path.dirname(__file__), "..", "ddb1_v02_01")
IMG_DIR = os.path.join(DB_DIR, "images")
GT_DIR = os.path.join(DB_DIR, "groundtruth")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "clf.joblib")

LIMIT = None  # mettre une limite seulement si on veut tester rapidement
# cap du nb de noeuds/image : borne la memoire de la transformee Nystroem
# (matrice n_echantillons x n_components). ~5000 x ~60 images ~ 300k lignes.
MAX_NODES_PER_IMAGE = 5000


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

# MODELE : SVM a noyau RBF approxime (Nystroem) + SVM lineaire scalable.
#   - StandardScaler : un SVM exige des features standardisees.
#   - Nystroem : approxime le noyau RBF -> non-linearite tout en passant a
#     l'echelle (un vrai SVC RBF est en O(n^2), infaisable sur ~300k noeuds).
#   - SGDClassifier(loss="modified_huber") : SVM lineaire (hinge lisse) qui
#     scale ET fournit predict_proba (requis par predict.py). class_weight
#     compense le desequilibre fond/exsudat.
pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("nystroem", Nystroem(kernel="rbf", n_components=200, random_state=42)),
        (
            "svm",
            SGDClassifier(
                loss="modified_huber",
                class_weight="balanced",
                random_state=42,
                max_iter=2000,
                tol=1e-3,
            ),
        ),
    ]
)

# Recherche legere des hyperparametres les plus sensibles :
#   gamma  = largeur du noyau RBF (Nystroem)
#   alpha  = force de regularisation du SVM lineaire
param_grid = {
    "nystroem__gamma": [0.1, 0.2, 0.5],
    "svm__alpha": [1e-5, 1e-4, 1e-3],
}
clf = GridSearchCV(
    pipe,
    param_grid,
    scoring="f1",  # classe exsudat minoritaire -> f1 plutot qu'accuracy
    cv=3,
    n_jobs=-1,
    verbose=2,
)

# TRAINING
print("Training (Nystroem RBF + SVM lineaire, GridSearchCV)...")
clf = train(train_imgs, train_gts, clf, max_nodes_per_image=MAX_NODES_PER_IMAGE)

print(f"Best params : {clf.best_params_}")
print(f"Best CV f1  : {clf.best_score_:.4f}")

# on sauvegarde le meilleur pipeline (et non tout l'objet GridSearchCV)
best = clf.best_estimator_
joblib.dump(best, MODEL_OUT)
print(f"Model saved in {MODEL_OUT}")
