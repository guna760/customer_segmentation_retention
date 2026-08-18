import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from .data_preprocessing import CHURN_FEATURES


def train_churn_models(df):

    # -------------------------------------------------
    # Features and target
    # -------------------------------------------------

    X = df[CHURN_FEATURES]

    y = df["churn"]

    # -------------------------------------------------
    # Train/Test Split
    # -------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

    # -------------------------------------------------
    # Preprocessing
    # -------------------------------------------------

    preprocessing = Pipeline([

        (
            "imputer",

            SimpleImputer(
                strategy="median"
            )
        )
    ])

    X_train_processed = (
        preprocessing.fit_transform(
            X_train
        )
    )

    X_test_processed = (
        preprocessing.transform(
            X_test
        )
    )

    # =================================================
    # LOGISTIC REGRESSION
    # =================================================

    logistic_model = LogisticRegression(

        max_iter=1000,

        random_state=42
    )

    logistic_model.fit(

        X_train_processed,

        y_train
    )

    # =================================================
    # RANDOM FOREST
    # =================================================

    random_forest = RandomForestClassifier(

        n_estimators=300,

        random_state=42,

        class_weight="balanced",

        n_jobs=-1
    )

    random_forest.fit(

        X_train_processed,

        y_train
    )

    # -------------------------------------------------
    # Random Forest Predictions
    # -------------------------------------------------

    predictions = random_forest.predict(
        X_test_processed
    )

    probabilities = random_forest.predict_proba(
        X_test_processed
    )[:, 1]

    # -------------------------------------------------
    # Evaluation
    # -------------------------------------------------

    print("\n")
    print("=" * 50)
    print("RANDOM FOREST CHURN MODEL")
    print("=" * 50)

    print(
        f"Accuracy: "
        f"{accuracy_score(y_test, predictions):.4f}"
    )

    print(
        f"Precision: "
        f"{precision_score(y_test, predictions, zero_division=0):.4f}"
    )

    print(
        f"Recall: "
        f"{recall_score(y_test, predictions, zero_division=0):.4f}"
    )

    print(
        f"F1 Score: "
        f"{f1_score(y_test, predictions, zero_division=0):.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{roc_auc_score(y_test, probabilities):.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # -------------------------------------------------
    # Save models
    # -------------------------------------------------

    joblib.dump(

        preprocessing,

        "models/churn_preprocessor.pkl"
    )

    joblib.dump(

        logistic_model,

        "models/logistic_model.pkl"
    )

    joblib.dump(

        random_forest,

        "models/random_forest_model.pkl"
    )

    return random_forest