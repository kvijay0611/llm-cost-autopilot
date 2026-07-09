"""
Phase 2 — Train the complexity classifier.

Deliberately a simple RandomForest on hand-designed features (see features.py).
We're not chasing classifier perfection, we're building the routing skeleton.
Anything above 80% held-out accuracy is fine for V1 — the async verifier (Phase 3)
is the real safety net that catches what this misses.
"""
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from app.classifier.dataset import generate_dataset
from app.classifier.features import extract_features, FEATURE_NAMES

MODEL_PATH = "app/classifier/model.joblib"


def build_training_matrix(examples):
    X = np.stack([extract_features(e.prompt, e.context) for e in examples])
    y = np.array([e.tier for e in examples])
    return X, y


def train_and_save(model_path: str = MODEL_PATH, per_tier: int = 80) -> dict:
    examples = generate_dataset(per_tier=per_tier)
    X, y = build_training_matrix(examples)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    joblib.dump(clf, model_path)

    importances = sorted(
        zip(FEATURE_NAMES, clf.feature_importances_), key=lambda t: -t[1]
    )

    return {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "report": report,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_importances": importances,
    }


if __name__ == "__main__":
    results = train_and_save()
    print(f"Held-out accuracy: {results['accuracy']:.3f}")
    print(f"Train size: {results['n_train']}  Test size: {results['n_test']}")
    print("\nConfusion matrix (rows=true tier 1/2/3, cols=pred tier 1/2/3):")
    for row in results["confusion_matrix"]:
        print(row)
    print("\n" + results["report"])
    print("Feature importances:")
    for name, imp in results["feature_importances"]:
        print(f"  {name:28s} {imp:.3f}")
    print(f"\nSaved model to {MODEL_PATH}")
