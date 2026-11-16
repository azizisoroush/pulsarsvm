"""
Inference Script for Pulsar Prediction using Saved SVM Model
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import joblib


def load_model(model_path="svm_pulsar_pipeline.joblib"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please run train_svm.py first.")
    package = joblib.load(model_path)
    return package


def predict_samples(pipeline, feature_names, df_samples):
    X = df_samples[feature_names]
    scores = pipeline.decision_function(X)
    preds = pipeline.predict(X)
    
    results = df_samples.copy()
    results["SVM_Decision_Score"] = scores
    results["Predicted_Class"] = preds
    results["Prediction_Label"] = results["Predicted_Class"].map({
        0: "Non-Pulsar (Noise / RFI)",
        1: "🌟 PULSAR CANDIDATE"
    })
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict pulsars using trained SVM model.")
    parser.add_argument("--model", type=str, default="svm_pulsar_pipeline.joblib", help="Path to saved pipeline")
    parser.add_argument("--data", type=str, default="Pulsar.csv", help="CSV data to test predictions on")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of random samples to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    args = parser.parse_args()

    print("=" * 70)
    print("PULSAR DETECTION INFERENCE")
    print("=" * 70)

    package = load_model(args.model)
    pipeline = package["pipeline"]
    feature_names = package["feature_names"]
    print(f"Loaded Model: {package['model_name']}")
    print(f"Features: {', '.join(feature_names)}")

    if os.path.exists(args.data):
        df = pd.read_csv(args.data)
        
        # Take both known pulsars and non-pulsars for illustration
        np.random.seed(args.seed)
        pulsars = df[df["Class"] == 1].sample(min(args.n_samples, (df["Class"] == 1).sum()), random_state=args.seed)
        non_pulsars = df[df["Class"] == 0].sample(min(args.n_samples, (df["Class"] == 0).sum()), random_state=args.seed)
        sample_df = pd.concat([pulsars, non_pulsars]).reset_index(drop=True)

        print(f"\nTesting on {len(sample_df)} sample candidates from {args.data}...")
        results = predict_samples(pipeline, feature_names, sample_df)

        display_cols = ["Mean_Integrated", "EK", "Skewness", "Mean_DMSNR_Curve", "Class", "SVM_Decision_Score", "Prediction_Label"]
        print("\n" + "-" * 70)
        print(results[display_cols].to_string(index=False))
        print("-" * 70)
        
        # Accuracy on this sample
        correct = (results["Class"] == results["Predicted_Class"]).sum()
        print(f"\nCorrect Predictions on sample: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
    else:
        print(f"Data file '{args.data}' not found.")


if __name__ == "__main__":
    main()
