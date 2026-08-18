import os
import joblib
import pandas as pd
import streamlit as st

from src.feature_engineering import create_features
from src.data_preprocessing import (
    clean_data,
    SEGMENT_FEATURES,
    CHURN_FEATURES
)


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Customer Segmentation & Retention Analysis",
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

DATA_FILE = "data/sales.csv"

SCALER_FILE = "models/scaler_segmentation.pkl"
KMEANS_FILE = "models/kmeans_model.pkl"
PREPROCESSOR_FILE = "models/churn_preprocessor.pkl"
CHURN_MODEL_FILE = "models/random_forest_model.pkl"


# =====================================================
# CHECK REQUIRED FILES
# =====================================================

required_files = [
    DATA_FILE,
    SCALER_FILE,
    KMEANS_FILE,
    PREPROCESSOR_FILE,
    CHURN_MODEL_FILE
]

missing_files = [
    file
    for file in required_files
    if not os.path.exists(file)
]

if missing_files:

    st.error(
        "Required files are missing."
    )

    st.write(
        "The following files were not found:"
    )

    for file in missing_files:
        st.write(f"- {file}")

    st.info(
        "Make sure sales.csv and the trained model files "
        "are available."
    )

    st.stop()


# =====================================================
# LOAD TRAINED MODELS
# =====================================================

@st.cache_resource
def load_models():

    scaler = joblib.load(
        SCALER_FILE
    )

    kmeans = joblib.load(
        KMEANS_FILE
    )

    preprocessor = joblib.load(
        PREPROCESSOR_FILE
    )

    churn_model = joblib.load(
        CHURN_MODEL_FILE
    )

    return (
        scaler,
        kmeans,
        preprocessor,
        churn_model
    )


(
    scaler,
    kmeans,
    preprocessor,
    churn_model
) = load_models()


# =====================================================
# LOAD RAW DATA
# =====================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        DATA_FILE
    )

    return data


df = load_data()


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header(
    "Dashboard"
)

st.sidebar.success(
    f"{len(df):,} customers loaded"
)


# =====================================================
# PREPARE DATA
# =====================================================

@st.cache_data
def prepare_data(data):

    data = data.copy()

    # -----------------------------------------------
    # Feature engineering
    # -----------------------------------------------

    data = create_features(
        data
    )

    # -----------------------------------------------
    # Cleaning
    # -----------------------------------------------

    data = clean_data(
        data
    )

    return data


with st.spinner(
    "Preparing customer data..."
):

    processed_data = prepare_data(
        df
    )


# =====================================================
# CUSTOMER SEGMENTATION
# IMPORTANT:
# DO NOT USE @st.cache_data HERE
# =====================================================

def predict_segments(
    data,
    scaler_model,
    kmeans_model
):

    X_segment = data[
        SEGMENT_FEATURES
    ].copy()

    # -----------------------------------------------
    # Make sure all values are numeric
    # -----------------------------------------------

    for column in X_segment.columns:

        X_segment[column] = pd.to_numeric(
            X_segment[column],
            errors="coerce"
        )

    # -----------------------------------------------
    # Handle missing values
    # -----------------------------------------------

    X_segment = X_segment.fillna(
        X_segment.median()
    )

    # If a column is completely NaN
    X_segment = X_segment.fillna(0)

    # -----------------------------------------------
    # Scaling
    # -----------------------------------------------

    X_segment_scaled = (
        scaler_model.transform(
            X_segment
        )
    )

    # -----------------------------------------------
    # K-Means prediction
    # -----------------------------------------------

    segments = (
        kmeans_model.predict(
            X_segment_scaled
        )
    )

    return segments


# =====================================================
# CHURN PREDICTION
# IMPORTANT:
# DO NOT USE @st.cache_data HERE
# =====================================================

def predict_churn(
    data,
    preprocessor_model,
    churn_model
):

    X_churn = data[
        CHURN_FEATURES
    ].copy()

    # -----------------------------------------------
    # Convert numeric columns where possible
    # -----------------------------------------------

    numeric_columns = X_churn.select_dtypes(
        include=["number"]
    ).columns

    for column in numeric_columns:

        X_churn[column] = X_churn[column].fillna(
            X_churn[column].median()
        )

    # -----------------------------------------------
    # Categorical missing values
    # -----------------------------------------------

    categorical_columns = X_churn.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:

        X_churn[column] = X_churn[column].fillna(
            "Unknown"
        )

    # -----------------------------------------------
    # Preprocessing
    # -----------------------------------------------

    X_churn_processed = (
        preprocessor_model.transform(
            X_churn
        )
    )

    # -----------------------------------------------
    # Random Forest prediction
    # -----------------------------------------------

    probabilities = (
        churn_model.predict_proba(
            X_churn_processed
        )[:, 1]
    )

    return probabilities


# =====================================================
# GENERATE PREDICTIONS
# =====================================================

with st.spinner(
    "Generating customer predictions..."
):

    processed_data = processed_data.copy()

    # -----------------------------------------------
    # Customer segmentation
    # -----------------------------------------------

    processed_data["segment"] = (
        predict_segments(
            processed_data,
            scaler,
            kmeans
        )
    )

    # -----------------------------------------------
    # Churn probability
    # -----------------------------------------------

    processed_data["churn_probability"] = (
        predict_churn(
            processed_data,
            preprocessor,
            churn_model
        )
    )


# =====================================================
# RISK CATEGORY
# =====================================================

def get_risk_category(
    probability
):

    if probability >= 0.70:

        return "High Risk"

    elif probability >= 0.40:

        return "Medium Risk"

    else:

        return "Low Risk"


processed_data[
    "risk_category"
] = (
    processed_data[
        "churn_probability"
    ]
    .apply(
        get_risk_category
    )
)


# =====================================================
# PREDICTED CHURN
# =====================================================

processed_data[
    "predicted_churn"
] = (
    processed_data[
        "churn_probability"
    ]
    >= 0.50
).astype(int)


# =====================================================
# RETENTION ACTION
# =====================================================

def get_retention_action(
    risk_category,
    lifetime_value
):

    if pd.isna(
        lifetime_value
    ):

        lifetime_value = 0

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


processed_data[
    "retention_action"
] = processed_data.apply(

    lambda row:

    get_retention_action(

        row[
            "risk_category"
        ],

        row.get(
            "lifetime_value",
            0
        )

    ),

    axis=1
)


# =====================================================
# CUSTOMER DATA
# =====================================================

customer_data = processed_data.copy()


# =====================================================
# SIDEBAR FILTERS
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
    ]
    .dropna()
    .unique()
)


selected_segments = st.sidebar.multiselect(

    "Select Customer Segment",

    options=available_segments,

    default=available_segments
)


# =====================================================
# RISK FILTER
# =====================================================

available_risks = [
    "High Risk",
    "Medium Risk",
    "Low Risk"
]


selected_risks = st.sidebar.multiselect(

    "Select Risk Category",

    options=available_risks,

    default=available_risks
)


# =====================================================
# APPLY FILTERS
# =====================================================

filtered_data = customer_data[
    customer_data[
        "segment"
    ].isin(
        selected_segments
    )
    &
    customer_data[
        "risk_category"
    ].isin(
        selected_risks
    )
]


# =====================================================
# CUSTOMER OVERVIEW
# =====================================================

st.header(
    "📊 Customer Overview"
)


col1, col2, col3, col4 = st.columns(4)


# -----------------------------------------------------
# TOTAL CUSTOMERS
# -----------------------------------------------------

with col1:

    st.metric(
        "Total Customers",
        f"{len(customer_data):,}"
    )


# -----------------------------------------------------
# HIGH RISK CUSTOMERS
# -----------------------------------------------------

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


# -----------------------------------------------------
# HIGH RISK %
# -----------------------------------------------------

with col3:

    high_risk_percentage = (
        customer_data[
            "risk_category"
        ]
        .eq("High Risk")
        .mean()
        * 100
    )

    st.metric(
        "High Risk %",
        f"{high_risk_percentage:.2f}%"
    )


# -----------------------------------------------------
# NUMBER OF SEGMENTS
# -----------------------------------------------------

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
# SEGMENT DISTRIBUTION
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
# SEGMENT ANALYSIS
# =====================================================

st.header(
    "📋 Segment Analysis"
)


segment_summary = (
    customer_data
    .groupby(
        "segment"
    )
    .agg(

        Customers=(
            "customer_id",
            "count"
        ),

        Average_Total_Spent=(
            "total_spent",
            "mean"
        ),

        Average_Order_Value=(
            "avg_order_value",
            "mean"
        ),

        Average_Lifetime_Value=(
            "lifetime_value",
            "mean"
        ),

        Average_Purchase_Frequency=(
            "last_3_month_purchase_freq",
            "mean"
        ),

        Average_Satisfaction=(
            "satisfaction_score",
            "mean"
        ),

        Average_NPS=(
            "nps_score",
            "mean"
        ),

        Average_Visits=(
            "total_visits",
            "mean"
        )
    )
    .reset_index()
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


churn_display_columns = [

    "customer_id",

    "segment",

    "churn_probability",

    "predicted_churn",

    "risk_category"
]


if "lifetime_value" in filtered_data.columns:

    churn_display_columns.append(
        "lifetime_value"
    )


st.dataframe(

    filtered_data[
        churn_display_columns
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


retention_columns = [

    "customer_id",

    "segment",

    "churn_probability",

    "risk_category",

    "retention_action"
]


if "lifetime_value" in filtered_data.columns:

    retention_columns.insert(
        4,
        "lifetime_value"
    )


st.dataframe(

    filtered_data[
        retention_columns
    ]
    .sort_values(
        "churn_probability",
        ascending=False
    )
    .head(100),

    use_container_width=True
)


# =====================================================
# PRIORITY CUSTOMERS
# =====================================================

st.header(
    "🚨 Priority Customers"
)


high_risk_customers = filtered_data[
    filtered_data[
        "risk_category"
    ] == "High Risk"
]


if len(
    high_risk_customers
) > 0:

    priority_columns = [

        "customer_id",

        "segment",

        "churn_probability",

        "risk_category",

        "retention_action"
    ]

    if "lifetime_value" in high_risk_customers.columns:

        priority_columns.insert(
            4,
            "lifetime_value"
        )

    st.dataframe(

        high_risk_customers[
            priority_columns
        ]
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
# DOWNLOAD PREDICTIONS
# =====================================================

st.header(
    "⬇️ Download Predictions"
)


download_columns = [

    "customer_id",

    "segment",

    "churn_probability",

    "predicted_churn",

    "risk_category",

    "retention_action"
]


if "lifetime_value" in customer_data.columns:

    download_columns.insert(
        5,
        "lifetime_value"
    )


download_data = customer_data[
    download_columns
]


csv_data = download_data.to_csv(
    index=False
)


st.download_button(

    label="Download Customer Predictions",

    data=csv_data,

    file_name="customer_predictions.csv",

    mime="text/csv"
)


# =====================================================
# FOOTER
# =====================================================

st.divider()


st.caption(
    "Customer Segmentation & Retention Analysis | "
    "Machine Learning Project"
)