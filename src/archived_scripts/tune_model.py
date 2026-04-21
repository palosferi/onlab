import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from train_model import load_data, BASELINE_DIR


def optimize_random_forest():
    print("Loading baseline data for hyperparameter tuning...")
    X, y = load_data(BASELINE_DIR)

    if len(X) == 0:
        print("Not enough data.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define parameter search space
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
    }

    rf = RandomForestClassifier(random_state=42)

    print("Starting grid search (this may take a while, 3-fold cross-validation)...")
    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2
    )

    grid_search.fit(X_train, y_train)

    print("\n--- TUNING RESULTS ---")
    print(f"Best parameters: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_
    accuracy = best_model.score(X_test, y_test)
    print(f"Optimized baseline accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    optimize_random_forest()
