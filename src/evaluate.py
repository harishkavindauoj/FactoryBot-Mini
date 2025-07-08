"""Evaluation functions for risk level and TTF prediction."""

import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    mean_absolute_error,
    r2_score
)
from scipy.stats import pearsonr


def evaluate_model(model, config):
    print("[Evaluate] 📊 Evaluating student model...")


    # Load features and true labels
    X = np.load(config['data']['features_cache'])
    y_risk_true = np.load('data/processed/risk_labels.npy')
    y_ttf_true = np.load('data/processed/ttf_labels.npy')

    preds = model.predict(X, verbose=0)
    print(f"Prediction output type: {type(preds)}")
    if isinstance(preds, dict):
        for k, v in preds.items():
            print(f" - {k}: shape {v.shape}")
    elif isinstance(preds, (list, tuple)):
        print("Shapes:", [p.shape for p in preds])
    else:
        print("Output shape:", preds.shape)

    # Predict outputs
    preds = model.predict(X, verbose=0)

    # If predictions are returned as a dict
    if isinstance(preds, dict):
        y_pred_risk = preds["risk_level"]
        y_pred_ttf = preds["ttf"]
    elif isinstance(preds, (list, tuple)):
        y_pred_risk, y_pred_ttf = preds
    else:
        raise ValueError("Unexpected model prediction output type.")

    # Convert risk labels from one-hot to class indices
    y_true_risk_labels = np.argmax(y_risk_true, axis=1)
    y_pred_risk_labels = np.argmax(y_pred_risk, axis=1)

    # 📌 Confusion Matrix
    cm = confusion_matrix(y_true_risk_labels, y_pred_risk_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low", "Moderate", "High", "Critical"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("🛡 Risk Level Confusion Matrix")
    if config.get('evaluate', {}).get('save_plots', False):
        plt.savefig("reports/risk_confusion_matrix.png")
    plt.show()

    # 🔍 Classification Report
    report = classification_report(y_true_risk_labels, y_pred_risk_labels, target_names=["Low", "Moderate", "High", "Critical"])
    print("\n📋 Risk Level Classification Report:\n", report)

    # 🧠 TTF Evaluation
    # Denormalize TTF using saved max
    ttf_max_path = 'data/processed/ttf_max_scale.pkl'
    if os.path.exists(ttf_max_path):
        with open(ttf_max_path, 'rb') as f:
            ttf_max = pickle.load(f)
        y_ttf_true = y_ttf_true * ttf_max
        y_pred_ttf = y_pred_ttf * ttf_max
    else:
        print("[Evaluate] ⚠️ Warning: ttf_max_scale not found. Assuming raw scale.")

    mae = mean_absolute_error(y_ttf_true, y_pred_ttf)
    r2 = r2_score(y_ttf_true, y_pred_ttf)
    corr, _ = pearsonr(y_ttf_true.flatten(), y_pred_ttf.flatten())

    print(f"\n🕒 TTF Metrics:")
    print(f" - MAE: {mae:.2f} hours")
    print(f" - R² Score: {r2:.3f}")
    print(f" - Pearson Correlation: {corr:.3f}")

    # 🔬 TTF Scatter Plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_ttf_true.flatten(), y=y_pred_ttf.flatten(), alpha=0.6)
    plt.xlabel("True TTF")
    plt.ylabel("Predicted TTF")
    plt.title("True vs Predicted TTF")
    plt.grid(True)
    if config.get('evaluate', {}).get('save_plots', False):
        os.makedirs("reports", exist_ok=True)
        plt.savefig("reports/ttf_scatter_plot.png")
    plt.show()
