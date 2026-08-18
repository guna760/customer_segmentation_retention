import pandas as pd


def create_retention_analysis(df):

    df = df.copy()

    # -------------------------------------------------
    # Risk category
    # -------------------------------------------------

    def risk_category(probability):

        if probability >= 0.70:

            return "High Risk"

        elif probability >= 0.40:

            return "Medium Risk"

        else:

            return "Low Risk"

    df["risk_category"] = (

        df["churn_probability"]

        .apply(risk_category)
    )

    # -------------------------------------------------
    # Retention strategy
    # -------------------------------------------------

    median_ltv = df[
        "lifetime_value"
    ].median()

    def retention_action(row):

        risk = row["risk_category"]

        ltv = row["lifetime_value"]

        if risk == "High Risk":

            if ltv >= median_ltv:

                return (
                    "Priority retention + "
                    "personalized offer"
                )

            return (
                "Retention campaign + coupon"
            )

        elif risk == "Medium Risk":

            return (
                "Engagement campaign"
            )

        else:

            return (
                "Loyalty + upselling"
            )

    df["retention_action"] = (

        df.apply(
            retention_action,
            axis=1
        )
    )

    return df