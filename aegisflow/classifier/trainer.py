"""
AegisFlow ML Triage Model Training Script
=========================================
Trains and evaluates the production XGBoost classifier on the curated CI/CD
failure log dataset, computes validation metrics, and serializes artifacts.

Usage:
    python -m aegisflow.classifier.trainer
"""

import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from aegisflow.classifier.dataset import generate_augmented_dataset
from aegisflow.classifier.model import TriageClassifier
from aegisflow.config import settings


def train_and_evaluate(save_artifact: bool = True):
    print("=" * 70)
    print("  AegisFlow: Training Classical ML Triage Gatekeeper (XGBoost)")
    print("=" * 70)

    # 1. Generate realistic augmented dataset
    print("[1/4] Generating augmented CI failure dataset...")
    full_dataset = generate_augmented_dataset(multiplier=25)
    print(f"      Total dataset size: {len(full_dataset)} realistic CI failure logs")

    # 2. Split train/test
    train_data, test_data = train_test_split(full_dataset, test_size=0.25, random_state=42)
    print(f"      Train set: {len(train_data)} | Test set: {len(test_data)}")

    # 3. Fit pipeline
    print("[2/4] Fitting feature union (Scalar + TF-IDF) and XGBClassifier...")
    classifier = TriageClassifier()
    classifier.fit(train_data)

    # 4. Evaluate on unseen test set
    print("[3/4] Evaluating out-of-sample performance...")
    y_true = []
    y_pred = []

    for item in test_data:
        result = classifier.predict(
            log_text=item["log"],
            exit_code=item.get("exit_code", 1),
            diff_churn_lines=item.get("churn", 0),
            historical_failure_rate=item.get("history", 0.15)
        )
        y_true.append(item["category"])
        y_pred.append(result.category)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n>>> Test Accuracy: {acc * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    # 5. Save model artifact
    if save_artifact:
        model_path = settings.model_dir / settings.triage_model_filename
        print(f"[4/4] Serializing trained model artifact to:\n      {model_path}")
        classifier.save(model_path)
        print(">>> Model successfully trained and persisted!")

    return classifier


if __name__ == "__main__":
    train_and_evaluate()
