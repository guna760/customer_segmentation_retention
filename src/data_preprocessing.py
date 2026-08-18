import pandas as pd


# =====================================================
# SEGMENTATION FEATURES
# =====================================================

SEGMENT_FEATURES = [

    "age",

    "is_premium_user",

    "total_visits",

    "avg_session_time",

    "pages_per_session",

    "email_open_rate",

    "email_click_rate",

    "total_spent",

    "avg_order_value",

    "discount_used",

    "support_tickets",

    "refund_requested",

    "delivery_delay_days",

    "satisfaction_score",

    "nps_score",

    "lifetime_value",

    "last_3_month_purchase_freq",

    "days_since_last_purchase",

    "customer_tenure_days"
]


# =====================================================
# CHURN FEATURES
# =====================================================

CHURN_FEATURES = [

    "age",

    "is_premium_user",

    "total_visits",

    "avg_session_time",

    "pages_per_session",

    "email_open_rate",

    "email_click_rate",

    "total_spent",

    "avg_order_value",

    "discount_used",

    "support_tickets",

    "refund_requested",

    "delivery_delay_days",

    "satisfaction_score",

    "nps_score",

    "marketing_spend_per_user",

    "lifetime_value",

    "last_3_month_purchase_freq",

    "days_since_last_purchase",

    "customer_tenure_days"
]


def clean_data(df):

    df = df.copy()

    # Remove duplicate customers
    df = df.drop_duplicates(
        subset="customer_id"
    )

    # -------------------------------------------------
    # Numeric columns
    # -------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns

    for column in numeric_columns:

        df[column] = df[column].fillna(
            df[column].median()
        )

    # -------------------------------------------------
    # Categorical columns
    # -------------------------------------------------

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in categorical_columns:

        df[column] = df[column].fillna(
            "Unknown"
        )

    return df