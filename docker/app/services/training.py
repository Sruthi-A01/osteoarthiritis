from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from imblearn.over_sampling import SMOTE

# CATEGORICAL columns (strings)
CATEGORICAL_FEATURES = [
    "Category_Grouped",
    "AgeGroup",
    "Gender",
    "type",
    "DrugName",
    "CompanyName",
]

# NUMERICAL columns (numbers only)
NUMERICAL_FEATURES = [
    "Total_Volume",
    "Volume_Percentile",
]


def prepare_and_train(df_model, models, random_state):
    X = df_model.drop(columns=["Target"])
    y = df_model["Target"]

    if len(df_model) < 10:
        raise ValueError("Not enough rows to train models")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=random_state
    )

    # -------- Encode categorical features --------
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = X_test[col].astype(str).map(
            lambda x: le.transform([x])[0]
            if x in le.classes_
            else X_train[col].mode()[0]
        )
        encoders[col] = le

    # -------- Scale numerical features ONLY --------
    scaler = RobustScaler()
    X_train[NUMERICAL_FEATURES] = scaler.fit_transform(
        X_train[NUMERICAL_FEATURES]
    )
    X_test[NUMERICAL_FEATURES] = scaler.transform(
        X_test[NUMERICAL_FEATURES]
    )

    # -------- Handle imbalance --------
    X_train, y_train = SMOTE(random_state=random_state).fit_resample(
        X_train, y_train
    )

    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model

    return fitted_models, X_test, y_test
