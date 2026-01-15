from sklearn.metrics import accuracy_score, recall_score, precision_score

def evaluate_models(models, X_test, y_test, threshold):
    results = []

    for name, model in models.items():
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)

        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "precision": precision_score(y_test, preds)
        })

    return results
