import pandas as pd


def create_features(df):

    df = df.copy()

    # =================================================
    # DATE CONVERSION
    # =================================================

    df["signup_date"] = pd.to_datetime(
        df["signup_date"],
        errors="coerce",
        dayfirst=True
    )

    df["last_purchase_date"] = pd.to_datetime(
        df["last_purchase_date"],
        errors="coerce",
        dayfirst=True
    )

    # =================================================
    # REFERENCE DATE
    # =================================================

    reference_date = df["last_purchase_date"].max()

    # =================================================
    # CUSTOMER TENURE
    # =================================================

    df["customer_tenure_days"] = (
        reference_date - df["signup_date"]
    ).dt.days

    # =================================================
    # DAYS SINCE LAST PURCHASE
    # =================================================

    df["days_since_last_purchase"] = (
        reference_date - df["last_purchase_date"]
    ).dt.days

    # =================================================
    # COUPON USAGE
    # =================================================

    df["used_coupon"] = (
        df["coupon_code"]
        .notna()
        .astype(int)
    )

    # =================================================
    # REMOVE NEGATIVE VALUES
    # =================================================

    df["customer_tenure_days"] = (
        df["customer_tenure_days"]
        .clip(lower=0)
    )

    df["days_since_last_purchase"] = (
        df["days_since_last_purchase"]
        .clip(lower=0)
    )

    return df