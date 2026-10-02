import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📉",
    layout="wide"
)


@st.cache_resource
def load_artifact():
    return joblib.load("models/telco_churn_model.joblib")


artifact = load_artifact()

model = artifact["model"]
model_name = artifact["model_name"]
threshold = artifact["threshold"]

st.title("📉 Telecom Customer Churn Prediction")

st.write(
    "Enter customer details to estimate churn risk and support "
    "proactive retention decisions."
)

st.caption(
    f"Model: {model_name} | Decision threshold: {threshold:.2f}"
)

st.sidebar.header("Customer Details")

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)

senior_citizen = st.sidebar.selectbox(
    "Senior citizen",
    [0, 1],
    format_func=lambda value: "Yes" if value == 1 else "No"
)

partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.sidebar.slider(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=12
)

phone_service = st.sidebar.selectbox(
    "Phone service",
    ["Yes", "No"]
)

if phone_service == "No":
    multiple_lines = "No phone service"
else:
    multiple_lines = st.sidebar.selectbox(
        "Multiple lines",
        ["No", "Yes"]
    )

internet_service = st.sidebar.selectbox(
    "Internet service",
    ["DSL", "Fiber optic", "No"]
)

if internet_service == "No":
    online_security = "No internet service"
    online_backup = "No internet service"
    device_protection = "No internet service"
    tech_support = "No internet service"
    streaming_tv = "No internet service"
    streaming_movies = "No internet service"

else:
    online_security = st.sidebar.selectbox(
        "Online security",
        ["No", "Yes"]
    )

    online_backup = st.sidebar.selectbox(
        "Online backup",
        ["No", "Yes"]
    )

    device_protection = st.sidebar.selectbox(
        "Device protection",
        ["No", "Yes"]
    )

    tech_support = st.sidebar.selectbox(
        "Tech support",
        ["No", "Yes"]
    )

    streaming_tv = st.sidebar.selectbox(
        "Streaming TV",
        ["No", "Yes"]
    )

    streaming_movies = st.sidebar.selectbox(
        "Streaming movies",
        ["No", "Yes"]
    )

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.sidebar.selectbox(
    "Paperless billing",
    ["Yes", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.sidebar.number_input(
    "Monthly charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0,
    step=1.0
)

total_charges = st.sidebar.number_input(
    "Total charges",
    min_value=0.0,
    max_value=10000.0,
    value=float(monthly_charges * tenure),
    step=10.0
)

input_data = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}])

st.divider()

if st.button("Predict churn risk", type="primary"):
    churn_probability = model.predict_proba(input_data)[:, 1][0]

    predicted_churn = churn_probability >= threshold

    metric_left, metric_right = st.columns(2)

    with metric_left:
        st.metric(
            "Churn probability",
            f"{churn_probability:.1%}"
        )

    with metric_right:
        prediction_text = (
            "High risk of churn"
            if predicted_churn
            else "Lower churn risk"
        )

        st.metric(
            "Prediction",
            prediction_text
        )

    gauge_chart = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_probability * 100,
        number={"suffix": "%"},
        title={"text": "Predicted Churn Risk"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#E45756"},
            "steps": [
                {"range": [0, 35], "color": "#D9EAD3"},
                {"range": [35, 60], "color": "#FFF2CC"},
                {"range": [60, 100], "color": "#F4CCCC"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": threshold * 100
            }
        }
    ))

    st.plotly_chart(
        gauge_chart,
        use_container_width=True
    )

    if predicted_churn:
        st.warning(
            "Recommended action: consider a targeted retention offer, "
            "service-quality follow-up, or contract-upgrade incentive."
        )
    else:
        st.success(
            "The customer is below the selected churn-risk threshold."
        )

    with st.expander("View entered customer data"):
        st.dataframe(
            input_data,
            use_container_width=True
        )

st.divider()

st.subheader("About the project")

st.write(
    "This dashboard uses a trained machine-learning pipeline to estimate "
    "the probability that a telecom customer will churn. Predictions support "
    "retention decisions but do not guarantee future customer behavior."
)
