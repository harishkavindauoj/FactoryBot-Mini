"""Evaluation functions for risk level and TTF."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error


def evaluate_model(model, config):
    print("[Evaluate] Evaluating student model...")

    X = np.load(config['data']['features_cache'])
    y_risk_true = np.load('data/processed/risk_labels.npy')
    y_ttf_true = np.load('data/processed/ttf_labels.npy')

    y_pred_risk, y_pred_ttf = model.predict(X)
    y_pred_risk_labels = np.argmax(y_pred_risk, axis=1)
    y_true_risk_labels = np.argmax(y_risk_true, axis=1)

    cm = confusion_matrix(y_true_risk_labels, y_pred_risk_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low", "Moderate", "High", "Critical"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Risk Level Confusion Matrix")
    plt.show()

    mae = mean_absolute_error(y_ttf_true, y_pred_ttf)
    print(f"Mean Absolute Error (TTF): {mae:.2f} hours")

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_ttf_true.flatten(), y=y_pred_ttf.flatten(), alpha=0.6)
    plt.xlabel("True TTF")
    plt.ylabel("Predicted TTF")
    plt.title("True vs Predicted TTF")
    plt.grid(True)
    plt.show()

