# TRAIN.PY
"""
Train a model on a specific fold with PCA and track performance metrics.
Includes memory footprint and wall-clock time measurement.
"""
import time
import argparse
import os
import copy
import joblib
import io
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler  
from src import config
from src import model_dispatcher

def get_model_size(model_obj):
    """
    Calculate the serialized size of the model object in Megabytes.
    Useful for comparing the memory footprint of different architectures.
    """
    buffer = io.BytesIO()
    joblib.dump(model_obj, buffer)
    size_mb = buffer.getbuffer().nbytes / (1024 * 1024)
    return size_mb

def run(fold, model, n_components, **model_kwargs):
    # Load training data containing fold information
    df = pd.read_csv(config.TRAINING_FILE)
    
    # Split data into training and validation sets for the current fold
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    x_train = df_train.drop("label", axis=1).values
    y_train = df_train.label.values
    x_valid = df_valid.drop("label", axis=1).values
    y_valid = df_valid.label.values

    # Feature Scaling: Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_valid_scaled = scaler.transform(x_valid)

    # Dimensionality Reduction: Apply PCA to reduce feature space
    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train_scaled) 
    x_valid_pca = pca.transform(x_valid_scaled) 

    # Classifier Initialization
    base_clf = model_dispatcher.models[model]
    clf = copy.deepcopy(base_clf) # Ensure we don't modify the global dispatcher object
    
    # Inject dynamic hyperparameters (e.g., n_estimators, max_depth, C, etc.)
    if model_kwargs:
        clf.set_params(**model_kwargs) 
    
    # Training: Measure Wall-Clock time (actual elapsed time)
    start_time = time.perf_counter()
    clf.fit(x_train_pca, y_train)
    end_time = time.perf_counter()
    fit_time = end_time - start_time

    # Objective Memory Measurement: Size of the trained model object
    model_size_mb = get_model_size(clf)

    # Prediction and Evaluation
    y_preds = clf.predict(x_valid_pca)
    report = metrics.classification_report(y_valid, y_preds, output_dict=True)
    conf_matrix = metrics.confusion_matrix(y_valid, y_preds)
    
    # --- MODEL SAVING ---
    # Create a unique parameter suffix for the filename to prevent overwriting
    params_suffix = "_".join([f"{k}{v}" for k, v in model_kwargs.items()])
    filename = f"{model}_{fold}_pca{n_components}_{params_suffix}.bin"
    output_path = os.path.join(config.MODEL_OUTPUT, filename)
    
    # Package components: Scaler and PCA must be saved alongside the classifier
    model_package = {
        'scaler': scaler,      
        'pca': pca,
        'classifier': clf
    }
    
    # Save the package using compression (level 3) to optimize disk space
    joblib.dump(model_package, output_path, compress=3)
    
    # Result dictionary for inter-model comparison analysis
    result_dict = {
        'fold': fold, 
        'model': model, 
        'n_components': n_components, 
        'macro_f1': report['macro avg']['f1-score'], # Directly extracted for plotting
        'accuracy': report['accuracy'],
        'fit_time': fit_time,
        'model_size_mb': model_size_mb,
        'confusion_matrix': conf_matrix
    }
    # Append hyperparameters to the result dictionary
    result_dict.update(model_kwargs)
    
    return result_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n_components", type=int, required=True)
    args = parser.parse_args()
    
    run(fold=args.fold, model=args.model, n_components=args.n_components)