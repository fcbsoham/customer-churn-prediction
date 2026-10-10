from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "outputs"
    / "customer_churn_model.joblib"
)

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DATA AND MODEL LOADING
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.stop()

df = load_data()
model = load_model()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(data):
    """Apply the feature engineering used during model training."""

    X = data.drop(
        columns=["customerID", "Churn"],
        errors="ignore",
    ).copy()

    X["TotalCharges"] = pd.to_numeric(
        X["TotalCharges"],
        errors="coerce",
    )

    X["TenureGroup"] = pd.cut(
        X["tenure"],
        bins=[-1, 12, 24, 48, float("inf")],
        labels=[
            "New",
            "Developing",
            "Established",
            "Long-term",
        ],
    )

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    def count_active_services(row):
        count = 0

        for column in service_columns:
            value = row[column]

            if value == "Yes":
                count += 1

            elif (
                column == "InternetService"
                and value in ["DSL", "Fiber optic"]
            ):
                count += 1

        return count

    X["ServiceUsageCount"] = X.apply(
        count_active_services,
        axis=1,
    )

    return X


def predict_customers(data):
    """Return predictions and churn probabilities."""

    X = engineer_features(data)

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    results = pd.DataFrame(
        {
            "PredictedChurn": [
                "Yes" if prediction == 1 else "No"
                for prediction in predictions
            ],
            "ChurnProbability": probabilities,
        }
    )

    return results


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Churn Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Predict Customer Churn",
        "Batch Predictions",
        "Model Performance",
    ],
)

st.sidebar.divider()

st.sidebar.caption(
    "Customer Churn Prediction\n"
    "Machine Learning Portfolio Project"
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.title("Customer Churn Analytics")
    st.markdown(
        "Understand customer retention, identify churn risks, "
        "and explore customer behavior using machine learning."
    )

    total_customers = len(df)

    actual_churn_count = (
        df["Churn"].astype(str).str.strip() == "Yes"
    ).sum()

    actual_churn_rate = (
        actual_churn_count / total_customers * 100
    )

    predictions = predict_customers(df)

    predicted_churn_count = (
        predictions["PredictedChurn"] == "Yes"
    ).sum()

    predicted_churn_rate = (
        predicted_churn_count / total_customers * 100
    )

    average_probability = (
        predictions["ChurnProbability"].mean() * 100
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}",
    )

    col2.metric(
        "Actual Churn Rate",
        f"{actual_churn_rate:.2f}%",
    )

    col3.metric(
        "Predicted Churn Rate",
        f"{predicted_churn_rate:.2f}%",
    )

    col4.metric(
        "Average Churn Probability",
        f"{average_probability:.2f}%",
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Actual Customer Churn")

        churn_counts = df["Churn"].value_counts()

        fig, ax = plt.subplots()

        ax.pie(
            churn_counts.values,
            labels=churn_counts.index,
            autopct="%1.1f%%",
            startangle=90,
        )

        ax.axis("equal")

        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Churn by Contract Type")

        contract_data = pd.crosstab(
            df["Contract"],
            df["Churn"],
            normalize="index",
        ) * 100

        if "Yes" in contract_data.columns:
            st.bar_chart(contract_data["Yes"])
        else:
            st.info("No churn records are available.")

    st.subheader("Monthly Charges by Churn Status")

    charge_summary = df.groupby(
        "Churn",
        observed=True,
    )["MonthlyCharges"].mean()

    st.bar_chart(charge_summary)

    st.caption(
        "Actual churn statistics describe the dataset. "
        "Predicted churn statistics describe the model's output."
    )


# ============================================================
# INDIVIDUAL CUSTOMER PREDICTION
# ============================================================

elif page == "Predict Customer Churn":

    st.title("Predict Customer Churn")

    st.write(
        "Enter customer information to estimate their "
        "likelihood of leaving the service."
    )

    with st.form("customer_prediction_form"):

        st.subheader("Customer Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox(
                "Gender",
                ["Male", "Female"],
            )

            senior_citizen = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: (
                    "Yes" if x == 1 else "No"
                ),
            )

            partner = st.selectbox(
                "Has Partner",
                ["Yes", "No"],
            )

            dependents = st.selectbox(
                "Has Dependents",
                ["Yes", "No"],
            )

        with col2:
            tenure = st.slider(
                "Tenure (months)",
                min_value=0,
                max_value=72,
                value=12,
            )

            phone_service = st.selectbox(
                "Phone Service",
                ["Yes", "No"],
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                [
                    "No phone service",
                    "No",
                    "Yes",
                ],
            )

            internet_service = st.selectbox(
                "Internet Service",
                [
                    "DSL",
                    "Fiber optic",
                    "No",
                ],
            )

        with col3:
            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year",
                ],
            )

            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"],
            )

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                step=1.0,
            )

        st.subheader("Internet Services")

        col1, col2, col3 = st.columns(3)

        with col1:
            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"],
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"],
            )

            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"],
            )

        with col2:
            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"],
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"],
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"],
            )

        with col3:
            online_security = online_security
            online_backup = online_backup
            device_protection = device_protection
            tech_support = tech_support
            streaming_tv = streaming_tv
            streaming_movies = streaming_movies

            # These selections are included in the prediction.
            st.info(
                "Review the selected services before "
                "generating your prediction."
            )

        st.subheader("Additional Information")

        col1, col2 = st.columns(2)

        with col1:
            multiple_lines = st.selectbox(
                "Confirm Multiple Lines",
                ["No phone service", "No", "Yes"],
                key="confirm_multiple_lines",
            )

            if phone_service == "No":
                multiple_lines = "No phone service"

        with col2:
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=20000.0,
                value=float(monthly_charges * tenure),
                step=10.0,
            )

        submitted = st.form_submit_button(
            "Predict Churn",
            use_container_width=True,
        )

    if submitted:

        customer = pd.DataFrame(
            [
                {
                    "gender": gender,
                    "SeniorCitizen": senior_citizen,
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "PhoneService": phone_service,
                    "MultipleLines": multiple_lines,
                    "InternetService": internet_service,
                    "OnlineSecurity": (
                        "No internet service"
                        if internet_service == "No"
                        else online_security
                    ),
                    "OnlineBackup": (
                        "No internet service"
                        if internet_service == "No"
                        else online_backup
                    ),
                    "DeviceProtection": (
                        "No internet service"
                        if internet_service == "No"
                        else device_protection
                    ),
                    "TechSupport": (
                        "No internet service"
                        if internet_service == "No"
                        else tech_support
                    ),
                    "StreamingTV": (
                        "No internet service"
                        if internet_service == "No"
                        else streaming_tv
                    ),
                    "StreamingMovies": (
                        "No internet service"
                        if internet_service == "No"
                        else streaming_movies
                    ),
                    "Contract": contract,
                    "PaperlessBilling": paperless_billing,
                    "PaymentMethod": payment_method,
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges,
                }
            ]
        )

        try:
            result = predict_customers(customer)

            probability = float(
                result["ChurnProbability"].iloc[0]
            )

            predicted_churn = (
                result["PredictedChurn"].iloc[0]
            )

            st.divider()
            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            col1.metric(
                "Predicted Outcome",
                (
                    "Likely to Churn"
                    if predicted_churn == "Yes"
                    else "Likely to Stay"
                ),
            )

            col2.metric(
                "Estimated Churn Probability",
                f"{probability:.2%}",
            )

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

            if predicted_churn == "Yes":
                st.warning(
                    "The model classifies this customer "
                    "as likely to churn. Consider reviewing "
                    "their contract, service experience, "
                    "and retention options."
                )
            else:
                st.success(
                    "The model classifies this customer "
                    "as likely to stay."
                )

            st.caption(
                "This is a model estimate, not a guarantee "
                "of future customer behavior."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# ============================================================
# BATCH PREDICTIONS
# ============================================================

elif page == "Batch Predictions":

    st.title("Batch Customer Predictions")

    st.write(
        "Upload a CSV containing customer records to generate "
        "churn predictions in bulk."
    )

    st.info(
        "Upload a CSV with the same customer feature columns "
        "as the original Telco Customer Churn dataset. "
        "The Churn and customerID columns are optional."
    )

    uploaded_file = st.file_uploader(
        "Upload customer CSV",
        type=["csv"],
    )

    if uploaded_file is not None:

        try:
            uploaded_df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data")

            st.write(
                f"Rows: {len(uploaded_df):,} | "
                f"Columns: {len(uploaded_df.columns)}"
            )

            st.dataframe(
                uploaded_df.head(10),
                use_container_width=True,
            )

            if st.button(
                "Generate Predictions",
                use_container_width=True,
            ):

                results = predict_customers(uploaded_df)

                output = uploaded_df.copy()

                output["PredictedChurn"] = (
                    results["PredictedChurn"]
                )

                output["ChurnProbability"] = (
                    results["ChurnProbability"].round(4)
                )

                st.subheader("Prediction Results")

                st.dataframe(
                    output.head(100),
                    use_container_width=True,
                )

                churn_count = (
                    results["PredictedChurn"] == "Yes"
                ).sum()

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Customers Analyzed",
                    f"{len(results):,}",
                )

                col2.metric(
                    "Predicted to Churn",
                    f"{churn_count:,}",
                )

                col3.metric(
                    "Predicted Churn Rate",
                    f"{churn_count / len(results):.2%}"
                    if len(results) else "N/A",
                )

                csv_data = output.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Predictions CSV",
                    data=csv_data,
                    file_name="customer_churn_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as error:
            st.error(
                f"Could not process the uploaded file: {error}"
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title("Model Performance")

    st.write(
        "Review the model evaluation results from the "
        "training run."
    )

    report_path = (
        BASE_DIR
        / "outputs"
        / "churn_analysis_report.txt"
    )

    if report_path.exists():

        report_text = report_path.read_text(
            encoding="utf-8"
        )

        st.subheader("Training Report")

        st.text(report_text)

    else:
        st.warning(
            "Training report not found. "
            "Run python src/train.py first."
        )

    st.subheader("Model Evaluation Charts")

    chart_files = [
        (
            "Confusion Matrix",
            "confusion_matrix.png",
        ),
        (
            "ROC Curve",
            "roc_curve.png",
        ),
        (
            "Model Comparison",
            "model_comparison.png",
        ),
    ]

    for title, filename in chart_files:

        chart_path = BASE_DIR / "outputs" / filename

        if chart_path.exists():

            st.markdown(f"#### {title}")

            st.image(
                str(chart_path),
                use_container_width=True,
            )

        else:
            st.info(
                f"{filename} has not been generated yet."
            )

    st.divider()

    st.caption(
        "Evaluation metrics reflect the saved training report. "
        "For an unbiased final evaluation, use a separate test set "
        "that was not used for model selection."
    )