import os
import pandas as pd
import joblib

from src.feature_engineering import (
    create_features
)

from src.data_preprocessing import (
    clean_data,
    CHURN_FEATURES
)

from src.segmentation import (
    train_segmentation
)

from src.churn_prediction import (
    train_churn_models
)

from src.segment_analysis import (
    analyze_segments
)

from src.retention_analysis import (
    create_retention_analysis
)


def main():

    print("\n")
    print("=" * 60)
    print(
        "CUSTOMER SEGMENTATION & RETENTION SYSTEM"
    )
    print("=" * 60)

    # =================================================
    # CREATE DIRECTORIES
    # =================================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    # =================================================
    # LOAD RAW DATA
    # =================================================

    print("\nLoading sales.csv...")

    data_path = "data/sales.csv"

    if not os.path.exists(data_path):

        raise FileNotFoundError(
            f"\nDataset not found: {data_path}\n"
            "Please make sure sales.csv is inside "
            "the data folder."
        )

    df = pd.read_csv(
        data_path
    )

    print(
        f"Raw dataset shape: {df.shape}"
    )

    print(
        f"Number of columns: {len(df.columns)}"
    )

    # =================================================
    # CHECK REQUIRED COLUMNS
    # =================================================

    required_columns = [

        "customer_id",

        "signup_date",

        "last_purchase_date",

        "churn",

        "total_visits",

        "avg_session_time",

        "pages_per_session",

        "total_spent",

        "avg_order_value",

        "support_tickets",

        "satisfaction_score",

        "nps_score",

        "lifetime_value",

        "last_3_month_purchase_freq"
    ]

    missing_columns = [

        column

        for column in required_columns

        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(

            "Missing required columns: "

            + str(missing_columns)
        )

    # =================================================
    # FEATURE ENGINEERING
    # =================================================

    print("\nCreating engineered features...")

    df = create_features(
        df
    )

    # =================================================
    # CLEAN DATA
    # =================================================

    print("\nCleaning data...")

    df = clean_data(
        df
    )

    print(
        f"Processed dataset shape: {df.shape}"
    )

    # =================================================
    # CUSTOMER SEGMENTATION
    # =================================================

    print("\n")
    print("=" * 60)
    print(
        "CUSTOMER SEGMENTATION"
    )
    print("=" * 60)

    df = train_segmentation(

        df,

        n_clusters=4
    )

    print(
        "\nCustomer segmentation completed."
    )

    # =================================================
    # SEGMENT ANALYSIS
    # =================================================

    print("\n")
    print("=" * 60)
    print(
        "CUSTOMER SEGMENT ANALYSIS"
    )
    print("=" * 60)

    segment_summary = analyze_segments(
        df
    )

    # =================================================
    # CHURN MODEL
    # =================================================

    print("\n")
    print("=" * 60)
    print(
        "CHURN PREDICTION"
    )
    print("=" * 60)

    random_forest = train_churn_models(
        df
    )

    # =================================================
    # CHURN PROBABILITY FOR ALL CUSTOMERS
    # =================================================

    print(
        "\nCalculating customer churn probabilities..."
    )

    X_all = df[
        CHURN_FEATURES
    ]

    # -------------------------------------------------
    # Load saved churn preprocessor
    # -------------------------------------------------

    preprocessor = joblib.load(

        "models/churn_preprocessor.pkl"
    )

    X_all_processed = (

        preprocessor

        .transform(
            X_all
        )
    )

    # -------------------------------------------------
    # Calculate probability
    # -------------------------------------------------

    churn_probability = (

        random_forest

        .predict_proba(
            X_all_processed
        )[:, 1]
    )

    df["churn_probability"] = (
        churn_probability
    )

    # =================================================
    # RETENTION ANALYSIS
    # =================================================

    print("\n")
    print("=" * 60)
    print(
        "RETENTION ANALYSIS"
    )
    print("=" * 60)

    df = create_retention_analysis(
        df
    )

    # =================================================
    # SAVE CUSTOMER SEGMENTS
    # =================================================

    print(
        "\nSaving customer segments..."
    )

    segment_output = df[

        [
            "customer_id",

            "segment"
        ]
    ]

    segment_output.to_csv(

        "outputs/customer_segments.csv",

        index=False
    )

    # =================================================
    # SAVE CHURN PREDICTIONS
    # =================================================

    print(
        "Saving churn predictions..."
    )

    churn_output = df[

        [
            "customer_id",

            "churn_probability",

            "risk_category"
        ]
    ]

    churn_output.to_csv(

        "outputs/churn_predictions.csv",

        index=False
    )

    # =================================================
    # SAVE SEGMENT SUMMARY
    # =================================================

    print(
        "Saving segment summary..."
    )

    segment_summary.to_csv(

        "outputs/segment_summary.csv",

        index=False
    )

    # =================================================
    # SAVE COMPLETE RETENTION ANALYSIS
    # =================================================

    print(
        "Saving complete retention analysis..."
    )

    df.to_csv(

        "outputs/retention_analysis.csv",

        index=False
    )

    # =================================================
    # FINAL SUMMARY
    # =================================================

    print("\n")
    print("=" * 60)
    print(
        "TRAINING COMPLETED"
    )
    print("=" * 60)

    # =================================================
    # MODEL FILES
    # =================================================

    print(
        "\nGenerated model files:"
    )

    print(
        "  models/scaler_segmentation.pkl"
    )

    print(
        "  models/kmeans_model.pkl"
    )

    print(
        "  models/churn_preprocessor.pkl"
    )

    print(
        "  models/logistic_model.pkl"
    )

    print(
        "  models/random_forest_model.pkl"
    )

    # =================================================
    # OUTPUT FILES
    # =================================================

    print(
        "\nGenerated output files:"
    )

    print(
        "  outputs/customer_segments.csv"
    )

    print(
        "  outputs/churn_predictions.csv"
    )

    print(
        "  outputs/segment_summary.csv"
    )

    print(
        "  outputs/retention_analysis.csv"
    )

    # =================================================
    # CUSTOMER COUNT
    # =================================================

    print(
        f"\nTotal customers processed: "
        f"{len(df)}"
    )

    # =================================================
    # SEGMENT DISTRIBUTION
    # =================================================

    print(
        "\nSegment distribution:"
    )

    print(
        df["segment"]
        .value_counts()
        .sort_index()
    )

    # =================================================
    # CHURN DISTRIBUTION
    # =================================================

    print(
        "\nChurn distribution:"
    )

    print(
        df["risk_category"]
        .value_counts()
    )

    print("\n")
    print("=" * 60)
    print(
        "ALL TRAINING TASKS COMPLETED SUCCESSFULLY"
    )
    print("=" * 60)


if __name__ == "__main__":

    main()