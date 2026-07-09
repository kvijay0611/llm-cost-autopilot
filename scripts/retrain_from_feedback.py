"""
Phase 3 (feedback loop) — Run this on a schedule (cron / weekly GitHub Action)
to fold accumulated routing failures back into the classifier's training set.

Every logged failure in data/routing_failures.jsonl becomes a new labeled
example (corrected_tier) added to the synthetic dataset before retraining.
This is the flywheel that makes the router get smarter over time.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.classifier.dataset import generate_dataset, Example
from app.classifier.train import build_training_matrix, MODEL_PATH
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

FEEDBACK_PATH = "./data/routing_failures.jsonl"


def load_feedback_examples() -> list[Example]:
    if not os.path.exists(FEEDBACK_PATH):
        return []
    examples = []
    with open(FEEDBACK_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(
                Example(prompt=row["prompt"], context=None, tier=row["corrected_tier"])
            )
    return examples


def main():
    base_examples = generate_dataset(per_tier=80)
    feedback_examples = load_feedback_examples()
    print(f"Base synthetic examples: {len(base_examples)}")
    print(f"Feedback examples from routing failures: {len(feedback_examples)}")

    all_examples = base_examples + feedback_examples
    X, y = build_training_matrix(all_examples)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    joblib.dump(clf, MODEL_PATH)

    print(f"Retrained classifier on {len(all_examples)} total examples")
    print(f"Held-out accuracy: {acc:.3f}")
    print(f"Saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
