# MODEL_DISPATCHER.PY
from sklearn import ensemble
from sklearn import tree
from sklearn import svm
from sklearn import linear_model
from sklearn import neighbors
from sklearn import neural_network

SEED = 42
MAX_ITER = 500
TOLERANCE = 1e-4
models = {
    # --- Lineal Baseline ---
    "logistic_regression": linear_model.LogisticRegression(
        solver='saga',
        random_state=SEED,
        max_iter=MAX_ITER,
        tol=TOLERANCE,
        n_jobs=-1
    ),

    # --- k-NN --- 
    "knn": neighbors.KNeighborsClassifier(
        n_jobs=-1
    ),

    # --- Trees ---
    "decision_tree": tree.DecisionTreeClassifier(random_state=SEED),
    
    "rf": ensemble.RandomForestClassifier(
        random_state=SEED, 
        n_jobs=-1
    ),

    "extra_trees": ensemble.ExtraTreesClassifier(
        random_state=SEED, 
        n_jobs=-1
    ),

    # --- SVM ---
    "svm_rbf": svm.SVC(
        random_state=SEED,
        tol=TOLERANCE 
    ),

    # --- MLP ---
    "mlp": neural_network.MLPClassifier(
        random_state=SEED, 
        max_iter=MAX_ITER,
        tol=TOLERANCE
    )
}