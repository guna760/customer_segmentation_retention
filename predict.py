import os
import joblib
import pandas as pd

from src.feature_engineering import create_features

from src.data_preprocessing import (
    SEGMENT_FEATURES,
    CHURN_FEATURES,
    clean_data
)


# =====================================================
# LOAD TRAINED MODELS
# =====================================================

def load_models():

    scaler = joblib.load(
        "models/scaler_segmentation.pkl"
    )

    kmeans = joblib.load(
        "models/kmeans_model.pkl"
    )

    preprocessor = joblib.load(
        "models/churn_preprocessor.pkl"
    )

    churn_model = joblib.load(
        "models/random_forest_model.pkl"
    )

    return (
        scaler,
        kmeans,
        preprocessor,
        churn_model
    )


# =====================================================
# RETENTION RISK
# =====================================================

def get_risk_category(probability):

    if probability >= 0.70:

        return "High Risk"

    elif probability >= 0.40:

        return "Medium Risk"

    else:

        return "Low Risk"


# =====================================================
# RETENTION ACTION
# =====================================================

def get_retention_action(
    risk_category,
    lifetime_value
):

    if risk_category == "High Risk":

        if lifetime_value > 50000:

            return (
                "Priority retention + "
                "personalized offer"
            )

        return (
            "Retention campaign + coupon"
        )

    elif risk_category == "Medium Risk":

        return (
            "Engagement campaign"
        )

    else:

        return (
            "Loyalty + upselling"
        )


# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_customers(df):

    df = df.copy()

    # =================================================
    # FEATURE ENGINEERING
    # =================================================

    print(
        "\nCreating prediction features..."
    )

    df = create_features(df)

    # =================================================
    # CLEAN DATA
    # =================================================

    print(
        "Cleaning prediction data..."
    )

    df = clean_data(df)

    # =================================================
    # LOAD MODELS
    # =================================================

    (
        scaler,
        kmeans,
        preprocessor,
        churn_model
    ) = load_models()

    # =================================================
    # CUSTOMER SEGMENTATION
    # =================================================

    print(
        "Predicting customer segments..."
    )

    X_segment = df[
        SEGMENT_FEATURES
    ]

    # Scale segmentation features

    X_segment_scaled = scaler.transform(
        X_segment
    )

    # Predict clusters

    segments = kmeans.predict(
        X_segment_scaled
    )

    df["segment"] = segments

    # =================================================
    # CHURN PREDICTION
    # =================================================

    print(
        "Predicting churn probability..."
    )

    X_churn = df[
        CHURN_FEATURES
    ]

    # Apply saved preprocessing

    X_churn_processed = (
        preprocessor.transform(
            X_churn
        )
    )

    # Predict churn probability

    churn_probability = (

        churn_model

        .predict_proba(
            X_churn_processed
        )[:, 1]
    )

    df["churn_probability"] = (
        churn_probability
    )

    # =================================================
    # PREDICTED CHURN
    # =================================================

    df["predicted_churn"] = (

        df["churn_probability"]

        >= 0.50

    ).astype(int)

    # =================================================
    # RISK CATEGORY
    # =================================================

    df["risk_category"] = (

        df["churn_probability"]

        .apply(
            get_risk_category
        )
    )

    # =================================================
    # RETENTION ACTION
    # =================================================

    df["retention_action"] = df.apply(

        lambda row:

        get_retention_action(

            row["risk_category"],

            row["lifetime_value"]

        ),

        axis=1
    )

    return df


# =====================================================
# MAIN
# =====================================================

def main():

    print("\n")
    print("=" * 60)
    print(
        "CUSTOMER SEGMENTATION & RETENTION "
        "PREDICTION"
    )
    print("=" * 60)

    # =================================================
    # CHECK REQUIRED MODELS
    # =================================================

    required_models = [

        "models/scaler_segmentation.pkl",

        "models/kmeans_model.pkl",

        "models/churn_preprocessor.pkl",

        "models/random_forest_model.pkl"
    ]

    print(
        "\nChecking trained models..."
    )

    for model in required_models:

        if not os.path.exists(model):

            raise FileNotFoundError(

                f"\nModel not found: {model}\n\n"

                "Please run:\n"

                "python train.py\n\n"

                "before running predict.py."
            )

    print(
        "All required models found."
    )

    # =================================================
    # LOAD RAW DATA
    # =================================================

    print(
        "\nLoading sales.csv..."
    )

    data_path = "data/sales.csv"

    if not os.path.exists(data_path):

        raise FileNotFoundError(

            f"\nDataset not found: {data_path}\n\n"

            "Make sure sales.csv is inside:\n"

            "data/sales.csv"
        )

    df = pd.read_csv(
        data_path
    )

    print(
        f"Customers loaded: {len(df)}"
    )

    print(
        f"Columns loaded: {len(df.columns)}"
    )

    # =================================================
    # PREDICTION
    # =================================================

    print(
        "\nGenerating predictions..."
    )

    results = predict_customers(
        df
    )

    # =================================================
    # SELECT OUTPUT COLUMNS
    # =================================================

    prediction_output = results[

        [
            "customer_id",

            "segment",

            "churn_probability",

            "predicted_churn",

            "risk_category",

            "lifetime_value",

            "retention_action"
        ]
    ]

    # =================================================
    # CREATE OUTPUT DIRECTORY
    # =================================================

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    # =================================================
    # SAVE PREDICTIONS
    # =================================================

    output_path = (
        "outputs/customer_predictions.csv"
    )

    prediction_output.to_csv(

        output_path,

        index=False
    )

    # =================================================
    # DISPLAY RESULTS
    # =================================================

    print("\n")
    print("=" * 60)
    print(
        "PREDICTION RESULTS"
    )
    print("=" * 60)

    print(
        prediction_output
        .head(10)
        .to_string(index=False)
    )

    # =================================================
    # SUMMARY
    # =================================================

    print("\n")
    print("=" * 60)
    print(
        "CUSTOMER RISK SUMMARY"
    )
    print("=" * 60)

    print(
        "\nRisk Distribution:"
    )

    print(
        prediction_output[
            "risk_category"
        ].value_counts()
    )

    print(
        "\nSegment Distribution:"
    )

    print(
        prediction_output[
            "segment"
        ].value_counts()
        .sort_index()
    )

    print("\n")

    print(
        "Prediction file created:"
    )

    print(
        output_path
    )

    print("\n")
    print(
        "Prediction completed successfully!"
    )


# =====================================================
# RUN PROGRAM
# =====================================================

if __name__ == "__main__":

    main()