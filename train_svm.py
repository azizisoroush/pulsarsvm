"""
SVM Pulsar Classification Pipeline
Dataset: HTRU2 / Pulsar Classification Dataset
"""

import os
import tempfile

# Prevent matplotlib cache permission issues
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def load_and_explore_data(csv_path="Pulsar.csv"):
    print("=" * 70)
    print("1. LOADING & EXPLORING PULSAR DATASET")
    print("=" * 70)
    
    df = pd.read_csv(csv_path)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Missing values check
    missing = df.isnull().sum()
    print(f"Total Missing Values: {missing.sum()}")
    
    # Target distribution
    class_counts = df['Class'].value_counts()
    class_pct = df['Class'].value_counts(normalize=True) * 100
    print("\nTarget Class Distribution:")
    print(f"  Class 0 (Non-Pulsar / Noise / RFI): {class_counts[0]:>5} ({class_pct[0]:.2f}%)")
    print(f"  Class 1 (True Pulsar):              {class_counts[1]:>5} ({class_pct[1]:.2f}%)")
    print(f"  Imbalance Ratio: ~{class_counts[0]/class_counts[1]:.1f} : 1")
    
    print("\nFeature Correlations with Class (sorted):")
    corrs = df.corr()['Class'].drop('Class').sort_values(ascending=False)
    for feat, corr in corrs.items():
        print(f"  {feat:<25}: {corr:+.4f}")
        
    return df


def evaluate_models(X_train, X_test, y_train, y_test, feature_names):
    print("\n" + "=" * 70)
    print("2. TRAINING & EVALUATING SVM VARIANTS")
    print("=" * 70)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models to compare
    models = {
        "Linear SVM (Default)": SVC(kernel="linear", C=1.0, random_state=42),
        "Linear SVM (Balanced)": SVC(kernel="linear", C=1.0, class_weight="balanced", random_state=42),
        "RBF SVM (Default)": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
        "RBF SVM (Balanced)": SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=42),
        "RBF SVM (Tuned C=10, γ=0.1)": SVC(kernel="rbf", C=10.0, gamma=0.1, random_state=42),
        "RBF SVM (Tuned Balanced)": SVC(kernel="rbf", C=10.0, gamma=0.1, class_weight="balanced", random_state=42),
    }

    results = []
    trained_clfs = {}
    test_scores = {}

    for name, clf in models.items():
        print(f"\n--> Training: {name}...")
        clf.fit(X_train_scaled, y_train)
        trained_clfs[name] = clf
        
        y_pred = clf.predict(X_test_scaled)
        y_scores = clf.decision_function(X_test_scaled)
        test_scores[name] = (y_scores, y_pred)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_scores)
        pr_auc = average_precision_score(y_test, y_scores)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision (Pulsar)": prec,
            "Recall (Pulsar)": rec,
            "F1-Score": f1,
            "ROC-AUC": roc,
            "PR-AUC": pr_auc,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn
        })

    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("3. MODEL COMPARISON SUMMARY TABLE")
    print("=" * 70)
    summary_cols = ["Model", "Accuracy", "Precision (Pulsar)", "Recall (Pulsar)", "F1-Score", "ROC-AUC", "PR-AUC", "TP", "FP", "FN"]
    print(results_df[summary_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return results_df, trained_clfs, test_scores, scaler


def plot_diagnostics(results_df, test_scores, y_test, output_path="svm_pulsar_evaluation.png"):
    print("\n" + "=" * 70)
    print("4. GENERATING DIAGNOSTIC PLOTS")
    print("=" * 70)
    
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. ROC Curves
    ax_roc = axes[0, 0]
    for name, (scores, _) in test_scores.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_val = roc_auc_score(y_test, scores)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_val:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Chance")
    ax_roc.set_title("ROC Curves Comparison", fontsize=13, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate (Recall)")
    ax_roc.legend(loc="lower right", fontsize=9)

    # 2. Precision-Recall Curves
    ax_pr = axes[0, 1]
    for name, (scores, _) in test_scores.items():
        prec, rec, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        ax_pr.plot(rec, prec, label=f"{name} (AP = {ap:.3f})")
    ax_pr.set_title("Precision-Recall Curves (Vital for Imbalanced Data)", fontsize=13, fontweight="bold")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="lower left", fontsize=9)

    # 3. Precision vs Recall Bar Comparison
    ax_bar = axes[1, 0]
    bar_data = results_df.melt(
        id_vars=["Model"],
        value_vars=["Precision (Pulsar)", "Recall (Pulsar)", "F1-Score"],
        var_name="Metric",
        value_name="Score"
    )
    sns.barplot(data=bar_data, x="Model", y="Score", hue="Metric", ax=ax_bar, palette="muted")
    ax_bar.set_title("Precision vs. Recall vs. F1 Trade-off", fontsize=13, fontweight="bold")
    models_list = results_df["Model"].tolist()
    ax_bar.set_xticks(range(len(models_list)))
    ax_bar.set_xticklabels(models_list, rotation=25, ha="right", fontsize=9)
    ax_bar.set_ylim(0.7, 1.02)
    ax_bar.legend(loc="lower right")

    # 4. Confusion Matrix for Best Balanced Model
    ax_cm = axes[1, 1]
    best_balanced_name = "RBF SVM (Balanced)"
    _, y_pred_best = test_scores[best_balanced_name]
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm, cbar=False,
                xticklabels=["Non-Pulsar (0)", "Pulsar (1)"],
                yticklabels=["Non-Pulsar (0)", "Pulsar (1)"])
    ax_cm.set_title(f"Confusion Matrix: {best_balanced_name}", fontsize=13, fontweight="bold")
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f" Diagnostic plots successfully saved to: {output_path}")


def save_best_pipeline(trained_clfs, scaler, feature_names, output_path="svm_pulsar_pipeline.joblib"):
    print("\n" + "=" * 70)
    print("5. SAVING BEST PIPELINE")
    print("=" * 70)
    
    # We choose Tuned RBF or RBF Balanced depending on goals.
    # We build a sklearn Pipeline with StandardScaler + SVC
    best_clf = trained_clfs["RBF SVM (Balanced)"]
    pipeline = Pipeline([
        ("scaler", scaler),
        ("svm", best_clf)
    ])
    
    package = {
        "pipeline": pipeline,
        "feature_names": feature_names,
        "model_name": "RBF SVM (Balanced)",
        "kernel": best_clf.kernel,
        "C": best_clf.C,
        "gamma": best_clf.gamma
    }
    
    joblib.dump(package, output_path)
    print(f" Pipeline successfully saved to: {output_path}")


def main():
    # 1. Load Data
    df = load_and_explore_data("Pulsar.csv")
    
    X = df.drop("Class", axis=1)
    y = df["Class"]
    feature_names = list(X.columns)

    # 2. Train-Test Split (Stratified 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain set: {X_train.shape[0]} samples ({y_train.sum()} pulsars)")
    print(f"Test set:  {X_test.shape[0]} samples ({y_test.sum()} pulsars)")

    # 3. Model Training & Evaluation
    results_df, trained_clfs, test_scores, scaler = evaluate_models(
        X_train, X_test, y_train, y_test, feature_names
    )

    # 4. Generate Plots
    plot_diagnostics(results_df, test_scores, y_test, "svm_pulsar_evaluation.png")

    # 5. Save Model Pipeline
    save_best_pipeline(trained_clfs, scaler, feature_names, "svm_pulsar_pipeline.joblib")
    
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
