# PREDICT_TEST.PY
"""
Generate predictions on test.csv using a trained final model package.
"""
import argparse
import joblib
import pandas as pd
import os
from src import config

def predict_test(model_path):
    """
    Load model package, transform test data, and save predictions to CSV.
    """
    # Load the serialized model package
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    package = joblib.load(model_path)
    scaler = package['scaler']   
    pca = package['pca']
    clf = package['classifier']

    # Read and prepare test data
    df_test = pd.read_csv(config.TEST_FILE)
    X_test = df_test.drop("label", axis=1).values
    y_test = df_test["label"].values
   
    # Apply stored transformations
    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)
   
    # Run Inference
    y_preds = clf.predict(X_test_pca)

    # Save results for later evaluation in the notebook
    submission = pd.DataFrame({
        'true_label': y_test,
        'prediction': y_preds
    })
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(config.TEST_OUTPUT), exist_ok=True)
    
    submission.to_csv(config.TEST_OUTPUT, index=False)
    print(f"Predictions saved to {config.TEST_OUTPUT}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict on test set using a trained model package.")
    parser.add_argument("--model_path", required=True, help="Path to the .bin model file")
    args = parser.parse_args()
    
    predict_test(model_path=args.model_path)