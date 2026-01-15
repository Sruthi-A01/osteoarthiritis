from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
}
