import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from train_model import load_data, BASELINE_DIR, OBFS4_DIR


def run_zero_shot():
    print("Zero-shot evaluation started...\n")

    print("[1/3] Loading baseline data for training...")
    X_train, y_train = load_data(BASELINE_DIR)

    if len(X_train) == 0:
        return

    print("[2/3] Training model on baseline traffic only...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    print("[3/3] Loading obfs4 data for testing...")
    X_test, y_test = load_data(OBFS4_DIR)

    if len(X_test) == 0:
        return

    print("\nRunning prediction on unseen obfs4 traffic...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("-" * 50)
    print(f"Zero-shot accuracy: {accuracy * 100:.2f}%")
    print("-" * 50)

    # --- CONFUSION MATRIX ---
    print("Generating zero-shot confusion matrix...")
    labels = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(12, 10))
    # Use red colormap so it is visually distinct from baseline and obfs4 matrices.
    sns.heatmap(cm, annot=False, cmap="Reds", xticklabels=labels, yticklabels=labels)

    plt.title("Confusion Matrix - Zero-Shot (unexpected obfs4 traffic)", fontsize=16)
    plt.ylabel("True website", fontsize=12)
    plt.xlabel("Predicted website", fontsize=12)
    plt.xticks(rotation=90)
    plt.tight_layout()

    plt.savefig("cm_zeroshot.png", dpi=300)
    print("Chart saved: cm_zeroshot.png")


if __name__ == "__main__":
    run_zero_shot()
