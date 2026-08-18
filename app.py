import os
import pandas as pd
import streamlit as st


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(

    page_title="Customer Segmentation & RetentionAnalysis",

    page_icon="👥",

    layout="wide"
)


# =====================================================
# TITLE
# =====================================================

st.title(
    "👥 Customer Segmentation & Retention Analysis"
)

st.markdown(
    """
    **Machine Learning Customer Intelligence Dashboard**

    Analyze customer segments, churn risk, customer value,
    and recommended retention strategies.
    """
)


# =====================================================
# FILE PATHS
# =====================================================

SEGMENT_FILE = (
    "outputs/customer_segments.csv"
)

CHURN_FILE = (
    "outputs/churn_predictions.csv"
)

SEGMENT_SUMMARY_FILE = (
    "outputs/segment_summary.csv"
)

RETENTION_FILE = (
    "outputs/retention_analysis.csv"
)


# =====================================================
# CHECK OUTPUT FILES
# =====================================================

required_files = [

    SEGMENT_FILE,

    CHURN_FILE,

    SEGMENT_SUMMARY_FILE,

    RETENTION_FILE
]


missing_files = [

    file

    for file in required_files

    if not os.path.exists(file)
]


if missing_files:

    st.error(
        "Required output files are missing."
    )

    st.write(
        "Please run the training pipeline first:"
    )

    st.code(
        "python train.py"
    )

    st.write(
        "Missing files:"
    )

    for file in missing_files:

        st.write(
            f"- {file}"
        )

    st.stop()


# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    segments = pd.read_csv(
        SEGMENT_FILE
    )

    churn = pd.read_csv(
        CHURN_FILE
    )

    segment_summary = pd.read_csv(
        SEGMENT_SUMMARY_FILE
    )

    retention = pd.read_csv(
        RETENTION_FILE
    )

    return (
        segments,
        churn,
        segment_summary,
        retention
    )


(
    segments,
    churn,
    segment_summary,
    retention
) = load_data()


# =====================================================
# MERGE CUSTOMER DATA
# =====================================================

customer_data = segments.merge(

    churn,

    on="customer_id",

    how="left"
)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header(
    "Dashboard Filters"
)


# =====================================================
# SEGMENT FILTER
# =====================================================

available_segments = sorted(

    customer_data[
        "segment"
    ].dropna().unique()
)


selected_segments = st.sidebar.multiselect(

    "Select Customer Segment",

    options=available_segments,

    default=available_segments
)


# =====================================================
# RISK FILTER
# =====================================================

available_risks = sorted(

    customer_data[
        "risk_category"
    ].dropna().unique()
)


selected_risks = st.sidebar.multiselect(

    "Select Risk Category",

    options=available_risks,

    default=available_risks
)


# =====================================================
# APPLY FILTERS
# =====================================================

filtered_data = customer_data[

    customer_data["segment"].isin(
        selected_segments
    )

    &

    customer_data["risk_category"].isin(
        selected_risks
    )
]


# =====================================================
# DASHBOARD METRICS
# =====================================================

st.header(
    "📊 Customer Overview"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(

        "Total Customers",

        f"{len(customer_data):,}"
    )


with col2:

    high_risk_count = (

        customer_data[
            "risk_category"
        ]

        .eq("High Risk")

        .sum()
    )

    st.metric(

        "High Risk Customers",

        f"{high_risk_count:,}"
    )


with col3:

    churn_rate = (

        customer_data[
            "risk_category"
        ]

        .eq("High Risk")

        .mean()

        * 100
    )

    st.metric(

        "High Risk %",

        f"{churn_rate:.2f}%"
    )


with col4:

    number_segments = (

        customer_data[
            "segment"
        ]

        .nunique()
    )

    st.metric(

        "Customer Segments",

        number_segments
    )


# =====================================================
# CUSTOMER SEGMENT DISTRIBUTION
# =====================================================

st.header(
    "👥 Customer Segment Distribution"
)


segment_counts = (

    filtered_data[
        "segment"
    ]

    .value_counts()

    .sort_index()
)


st.bar_chart(
    segment_counts
)


# =====================================================
# SEGMENT SUMMARY
# =====================================================

st.header(
    "📋 Segment Analysis"
)


st.dataframe(

    segment_summary,

    use_container_width=True
)


# =====================================================
# CHURN RISK DISTRIBUTION
# =====================================================

st.header(
    "⚠️ Customer Churn Risk"
)


risk_counts = (

    filtered_data[
        "risk_category"
    ]

    .value_counts()
)


st.bar_chart(
    risk_counts
)


# =====================================================
# CHURN PROBABILITY
# =====================================================

st.header(
    "📈 Churn Probability"
)


st.dataframe(

    filtered_data[

        [
            "customer_id",

            "segment",

            "churn_probability",

            "risk_category"
        ]

    ]

    .sort_values(

        "churn_probability",

        ascending=False

    )

    .head(100),

    use_container_width=True
)


# =====================================================
# RETENTION ACTIONS
# =====================================================

st.header(
    "🎯 Recommended Retention Actions"
)


if "retention_action" in retention.columns:

    retention_filtered = retention[

        retention["customer_id"].isin(

            filtered_data[
                "customer_id"
            ]
        )
    ]

    display_columns = [

        "customer_id",

        "segment",

        "churn_probability",

        "risk_category",

        "lifetime_value",

        "retention_action"
    ]


    display_columns = [

        column

        for column in display_columns

        if column in retention_filtered.columns
    ]


    st.dataframe(

        retention_filtered[
            display_columns
        ]

        .sort_values(

            "churn_probability",

            ascending=False

        )

        .head(100),

        use_container_width=True
    )

else:

    st.info(
        "Retention action data is not available."
    )


# =====================================================
# HIGH-RISK CUSTOMERS
# =====================================================

st.header(
    "🚨 Priority Customers"
)


high_risk_customers = filtered_data[

    filtered_data[
        "risk_category"
    ]

    == "High Risk"
]


if len(high_risk_customers) > 0:

    st.dataframe(

        high_risk_customers

        .sort_values(

            "churn_probability",

            ascending=False

        )

        .head(50),

        use_container_width=True
    )

else:

    st.success(
        "No high-risk customers in the selected filters."
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Customer Segmentation & Retention Analysis | "
    "Machine Learning Project"
)