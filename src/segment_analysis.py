import pandas as pd

from .data_preprocessing import SEGMENT_FEATURES


def analyze_segments(df):

    print("\n")
    print("=" * 70)
    print("CUSTOMER SEGMENT ANALYSIS")
    print("=" * 70)

    # Calculate segment statistics
    segment_summary = df.groupby("segment").agg({

        "customer_id": "count",

        "age": "mean",

        "total_visits": "mean",

        "avg_session_time": "mean",

        "pages_per_session": "mean",

        "total_spent": "mean",

        "avg_order_value": "mean",

        "support_tickets": "mean",

        "satisfaction_score": "mean",

        "nps_score": "mean",

        "lifetime_value": "mean",

        "last_3_month_purchase_freq": "mean",

        "days_since_last_purchase": "mean",

        "churn": "mean"

    }).reset_index()

    # Rename columns
    segment_summary = segment_summary.rename(
        columns={
            "customer_id": "customer_count",
            "age": "avg_age",
            "total_visits": "avg_visits",
            "avg_session_time": "avg_session_time",
            "pages_per_session": "avg_pages_per_session",
            "total_spent": "avg_total_spent",
            "avg_order_value": "avg_order_value",
            "support_tickets": "avg_support_tickets",
            "satisfaction_score": "avg_satisfaction",
            "nps_score": "avg_nps",
            "lifetime_value": "avg_lifetime_value",
            "last_3_month_purchase_freq":
                "avg_purchase_frequency",
            "days_since_last_purchase":
                "avg_days_since_purchase",
            "churn": "churn_rate"
        }
    )

    # Convert churn rate to percentage
    segment_summary["churn_rate"] = (
        segment_summary["churn_rate"] * 100
    )

    # Print each segment
    for _, row in segment_summary.iterrows():

        print("\n")
        print("-" * 70)

        print(
            f"SEGMENT {int(row['segment'])}"
        )

        print("-" * 70)

        print(
            f"Customers: "
            f"{int(row['customer_count'])}"
        )

        print(
            f"Average Total Spent: "
            f"{row['avg_total_spent']:.2f}"
        )

        print(
            f"Average Order Value: "
            f"{row['avg_order_value']:.2f}"
        )

        print(
            f"Average Lifetime Value: "
            f"{row['avg_lifetime_value']:.2f}"
        )

        print(
            f"Average Purchase Frequency: "
            f"{row['avg_purchase_frequency']:.2f}"
        )

        print(
            f"Average Days Since Purchase: "
            f"{row['avg_days_since_purchase']:.2f}"
        )

        print(
            f"Average Satisfaction: "
            f"{row['avg_satisfaction']:.2f}"
        )

        print(
            f"Average NPS: "
            f"{row['avg_nps']:.2f}"
        )

        print(
            f"Average Visits: "
            f"{row['avg_visits']:.2f}"
        )

        print(
            f"Churn Rate: "
            f"{row['churn_rate']:.2f}%"
        )

    # Save summary
    segment_summary.to_csv(
        "outputs/segment_summary.csv",
        index=False
    )

    print("\n")
    print(
        "Segment analysis saved to:"
    )

    print(
        "outputs/segment_summary.csv"
    )

    return segment_summary