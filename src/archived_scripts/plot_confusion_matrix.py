import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from train_model import load_data, BASELINE_DIR, OBFS4_DIR


def plot_cm(X, y, title, filename, colormap):
    if len(X) == 0:
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Use tuned RF parameters
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    labels = np.unique(y)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(12, 10))
    # Pass the color scheme through cmap
    sns.heatmap(cm, annot=False, cmap=colormap, xticklabels=labels, yticklabels=labels)

    plt.title(f"Confusion Matrix - {title}", fontsize=16)
    plt.ylabel("True website", fontsize=12)
    plt.xlabel("Predicted website", fontsize=12)
    plt.xticks(rotation=90)
    plt.tight_layout()

    plt.savefig(filename, dpi=300)
    print(f"Confusion matrix saved: {filename} (colormap: {colormap})")


def main():
    print("Generating confusion matrices...")

    # 1. Baseline matrix - blue
    X_base, y_base = load_data(BASELINE_DIR)
    plot_cm(X_base, y_base, "Baseline (regular Tor)", "cm_baseline.png", "Blues")

    # 2. Obfs4 matrix - yellow/orange
    X_obfs, y_obfs = load_data(OBFS4_DIR)
    plot_cm(X_obfs, y_obfs, "Obfs4 (obfuscated)", "cm_obfs4.png", "YlOrBr")


if __name__ == "__main__":
    main()
