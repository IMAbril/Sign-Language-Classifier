#TRAIN_FINAL_MODEL.PY
"""
Train final model on entire training set using optimal parameters from config.
"""
import argparse
import joblib
import copy
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler   
from src import model_dispatcher
from src import config

def train_final_model(model_name):
    # Fetch best parameters and n_components from config
    params = config.BEST_PARAMS.get(model_name, {}).copy()
    n_components = params.pop('PCA', None) # Extract PCA components separately
    
    # Load full training data
    df = pd.read_csv(config.TRAIN_FILE)
    X = df.drop("label", axis=1).values
    y = df.label.values

    # Preprocessing: Scaling and Dimensionality Reduction
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)   
    
    # Model Training
    base_clf = model_dispatcher.models[model_name]
    clf = copy.deepcopy(base_clf)
    
    if params:
        clf.set_params(**params) # Inject optimal hyperparameters
    
    clf.fit(X_pca, y)

    # Save Model Package 
    model_package = {
        'scaler': scaler,      
        'pca': pca,
        'classifier': clf
    }
    
    output_path = f"{config.MODEL_OUTPUT}final_{model_name}_n{n_components}.bin"
    joblib.dump(model_package, output_path)
    print(f"Final model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name defined in dispatcher")
    args = parser.parse_args()
    
    train_final_model(model_name=args.model)