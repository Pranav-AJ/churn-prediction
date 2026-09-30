# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to **churn or stay**, with an interactive Streamlit app for predictions, EDA, model analysis and business insights.

**Live app:** https://churn-prediction-tuhenpfhhpolqe5ngsec9g.streamlit.app/

---

## Business Problem

A telecom company wants to understand why customers leave and identify customers at risk of churning, so the retention team can act before they go. This project analyses customer demographics, subscribed services, contract details and billing information, and builds a classification model that outputs a **churn probability** for each customer.

## Dataset

- **Source file:** `Customer_Data.csv` (Telco customer churn data)
- **Size:** 7,043 customers, 21 columns
- **Target:** `Churn` (Yes/No), with about 26.5% churners, so the classes are imbalanced
- **Features:** demographics (gender, senior citizen, partner, dependents), services (phone, internet, security, backup, device protection, tech support, streaming), account details (tenure, contract, paperless billing, payment method) and billing (monthly and total charges)

## Approach

1. **Data quality:** converted `TotalCharges` from text to numeric. It had 11 blank values, all for customers with tenure 0, filled with 0. No duplicates found.
2. **EDA:** churn distribution, churn by contract, internet service, payment method, tenure and monthly charges.
3. **Features:** dropped `customerID` (identifier) and `MultipleLines`, so the model inputs match the app form. Numerical features are scaled and categorical features are one-hot encoded.
4. **Split:** 80/20 stratified train/test split (`random_state=42`).
5. **Imbalance:** handled with `class_weight="balanced"` and evaluated with recall, F1 and ROC-AUC instead of accuracy alone.
6. **Models:** Logistic Regression, Decision Tree and Random Forest, each built as a single scikit-learn `Pipeline` (preprocessing + model).
7. **Tuning:** `GridSearchCV` on the Random Forest, 5-fold stratified CV, optimising F1.
8. **Deployment:** the final pipeline is saved with `joblib` and loaded directly by the Streamlit app.

## Results

Test-set performance (values from my run; they may differ slightly from yours depending on library versions):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.739 | 0.505 | 0.789 | 0.616 | 0.842 |
| Decision Tree | 0.754 | 0.526 | 0.757 | 0.621 | 0.832 |
| Random Forest | 0.785 | 0.623 | 0.481 | 0.543 | 0.823 |
| **Tuned Random Forest (final)** | 0.767 | 0.542 | 0.778 | **0.639** | **0.843** |

**Final model:** the tuned Random Forest catches about 78% of churners (recall) and has the best F1 and ROC-AUC. **Recall** is the metric to monitor most closely: a missed churner (false negative) means lost revenue, while a false positive only costs an unnecessary retention offer.

## Key Insights

- **Month-to-month contracts** are the strongest churn driver. One- and two-year contracts retain far better.
- **New customers** (short tenure) are the most at risk.
- **Fiber optic** customers churn more, which points to a price or service-quality issue.
- Customers **without Online Security or Tech Support** churn more, so bundling these could help retention.
- **Electronic check** users churn more than customers on automatic payment methods.
- **Higher monthly charges** are associated with churn.

Top features by importance: contract type (month-to-month), tenure, total charges, no online security and monthly charges.

## Streamlit App

| Page | What it does |
|---|---|
| **Customer Churn Prediction** | Enter customer details and click **Predict Churn** to get *Likely to Churn* / *Likely to Stay* with the churn probability |
| **EDA Dashboard** | Interactive charts with short observations |
| **Model Analysis** | Model comparison table, confusion matrix and feature importance |
| **Business Insights** | Key findings and how the model helps retention |

## Project Structure

```
churn-prediction/
├── app.py                      # Streamlit application
├── churn_analysis.ipynb        # EDA, modelling, evaluation, final analysis
├── Customer_Data.csv           # Dataset
├── requirements.txt            # Python dependencies
├── README.md
└── artifacts/
    ├── churn_pipeline.joblib   # Trained preprocessing + model pipeline
    ├── model_comparison.csv    # Metrics for all models
    ├── feature_importance.csv  # Final model feature importances
    └── final_meta.json         # Confusion matrix and best hyperparameters
```

## Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Pranav-AJ/churn-prediction.git
cd churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Re-train the model by running all cells in the notebook
jupyter notebook churn_analysis.ipynb

# 4. Launch the app
streamlit run app.py
```

> **Important:** the saved model must be loaded with the same scikit-learn version that trained it. `requirements.txt` pins this version. If you re-train, run the notebook and the app in the same Python environment.

## Limitations

- Single snapshot of customers with no time dimension, so trends and seasonality are not captured.
- No data on *why* customers left (service issues, competitor offers, pricing changes).
- Precision is moderate (about 54%), so roughly half of the flagged customers would not actually churn.
- Performance may not generalise to other markets or time periods without retraining.

## Tech Stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Plotly, Streamlit, joblib

## Author

**Pranav**: [github.com/Pranav-AJ](https://github.com/Pranav-AJ)
