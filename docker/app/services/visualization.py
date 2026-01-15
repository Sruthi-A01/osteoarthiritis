import matplotlib
matplotlib.use("Agg")  # REQUIRED for Flask / macOS

import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc

PLOT_DIR = "app/static/plots"


def clear_plots():
    for f in os.listdir(PLOT_DIR):
        if f.endswith(".png"):
            os.remove(os.path.join(PLOT_DIR, f))


def save_confusion(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    path = f"{PLOT_DIR}/cm_{name}.png"

    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(path)
    plt.close()

    return f"cm_{name}.png"


def save_roc(y_true, y_proba, name):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    path = f"{PLOT_DIR}/roc_{name}.png"
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.legend()
    plt.savefig(path)
    plt.close()

    return f"roc_{name}.png"
