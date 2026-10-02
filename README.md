
# Telecom Customer Churn Prediction

An end-to-end data science project that predicts whether a telecom customer is likely to leave the company.

## Business Objective

Customer churn causes loss of revenue. This project identifies customers at high risk of churn and investigates the factors associated with their decision to leave.

## Project Goals

- Analyze telecom customer data.
- Clean and preprocess the dataset.
- Build customer churn classification models.
- Compare model performance using recall, precision, F1-score, and ROC-AUC.
- Explain model predictions using SHAP.
- Build an interactive Streamlit dashboard.

## Tech Stack

- Python
- Pandas and NumPy
- Matplotlib, Seaborn, and Plotly
- Scikit-learn and XGBoost
- SHAP
- Streamlit

## Project Structure

```text
telco-churn-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── models/
├── reports/
│   └── figures/
├── README.md
└── requirements.txt
```

## Dataset

IBM Telco Customer Churn dataset from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
## Live Demo

https://8kylanqcuuyp647dv6f5zb.streamlit.app/

## Dashboard Features

- Interactive customer data input form.
- Real-time churn probability prediction.
- Custom classification threshold.
- Churn-risk gauge visualization.
- Retention recommendation for high-risk customers.
- Display of the entered customer profile.

## Machine Learning Workflow

1. Cleaned the `TotalCharges` column and handled missing values.
2. Performed exploratory data analysis of contract type, tenure, charges, payment method, and internet service.
3. Built a Logistic Regression baseline model.
4. Compared Logistic Regression, Random Forest, and XGBoost.
5. Evaluated models with precision, recall, F1-score, ROC-AUC, and confusion matrices.
6. Tuned the classification threshold based on churn-retention trade-offs.
7. Used permutation importance and SHAP to interpret churn drivers.
8. Deployed the final pipeline through Streamlit Community Cloud.

## Results

| Item | Value |
|---|---|
| Final model | `XGBOOST` |
| Classification threshold | `0.40` |
| ROC-AUC | `0.84` |
| Recall for churn | `0.78` |
| F1-score for churn | `0.6136` |

## Project Structure

```text
telco-churn-prediction/
├── app.py
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   └── telco_churn_model.joblib
├── notebooks/
│   └── 01_eda_and_data_cleaning.ipynb
├── reports/
│   └── figures/
├── src/
├── requirements.txt
└── README.md
```

## Limitations

- The dataset represents a fictional telecom company, so results may not transfer directly to a real company.
- Model predictions show statistical churn risk, not guaranteed behavior.
- Feature importance identifies association with churn; it does not prove causation.
- The model should be monitored and retrained when real customer behavior or service plans change.
