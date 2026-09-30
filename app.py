import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Customer Churn", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("artifacts/churn_pipeline.joblib")


@st.cache_data
def load_data():
    d = pd.read_csv("Customer_Data.csv")
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(0)
    return d


model, df = load_model(), load_data()
page = st.sidebar.radio("Go to", ["Customer Churn Prediction", "EDA Dashboard", "Model Analysis", "Business Insights"])

# ------------------------------------------------ 1. Prediction
if page == "Customer Churn Prediction":
    st.title("Customer Churn Prediction")
    c1, c2, c3 = st.columns(3)
    gender = c1.selectbox("Gender", ["Female", "Male"])
    senior = c1.selectbox("Senior Citizen", ["No", "Yes"])
    partner = c1.selectbox("Partner", ["Yes", "No"])
    dependents = c1.selectbox("Dependents", ["No", "Yes"])
    tenure = c1.slider("Tenure (months)", 0, 72, 12)
    phone = c1.selectbox("Phone Service", ["Yes", "No"])

    internet = c2.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    opt = ["Yes", "No", "No internet service"]
    sec = c2.selectbox("Online Security", opt)
    backup = c2.selectbox("Online Backup", opt)
    device = c2.selectbox("Device Protection", opt)
    tech = c2.selectbox("Tech Support", opt)
    stv = c2.selectbox("Streaming TV", opt)

    smov = c3.selectbox("Streaming Movies", opt)
    contract = c3.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = c3.selectbox("Paperless Billing", ["Yes", "No"])
    pay = c3.selectbox("Payment Method", ["Electronic check", "Mailed check",
                                          "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = c3.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total = c3.number_input("Total Charges", 0.0, 10000.0, float(round(monthly * tenure, 2)))

    if st.button("Predict Churn", type="primary"):
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": int(senior == "Yes"), "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "InternetService": internet, "OnlineSecurity": sec, "OnlineBackup": backup,
            "DeviceProtection": device, "TechSupport": tech, "StreamingTV": stv,
            "StreamingMovies": smov, "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": pay, "MonthlyCharges": monthly, "TotalCharges": total}])
        p = model.predict_proba(row)[0, 1]
        if p >= 0.5:
            st.error(f"Likely to Churn — probability {p:.1%}")
        else:
            st.success(f"Likely to Stay — churn probability {p:.1%}")
        st.progress(float(p))

# ------------------------------------------------ 2. EDA
elif page == "EDA Dashboard":
    st.title("EDA Dashboard")
    rate = lambda col: df.groupby(col)["Churn"].apply(lambda s: (s == "Yes").mean()).reset_index(name="Churn rate")

    a, b = st.columns(2)
    a.plotly_chart(px.pie(df, names="Churn", title="Churn distribution"), width="stretch")
    a.caption("About 26.5% of customers churn: the classes are imbalanced.")
    for col, box, note in [
        ("Contract", b, "Month-to-month customers churn far more than 1- or 2-year contracts."),
        ("InternetService", a, "Fiber optic customers churn the most; customers with no internet churn the least."),
        ("PaymentMethod", b, "Electronic check users churn at a much higher rate than other payment methods."),
    ]:
        box.plotly_chart(px.bar(rate(col), x=col, y="Churn rate", title=f"Churn by {col}"), width="stretch")
        box.caption(note)
    a.plotly_chart(px.box(df, x="Churn", y="tenure", title="Churn vs tenure"), width="stretch")
    a.caption("Churners are mostly new customers (short tenure).")
    b.plotly_chart(px.box(df, x="Churn", y="MonthlyCharges", title="Churn vs monthly charges"), width="stretch")
    b.caption("Churners pay higher monthly charges on average.")

# ------------------------------------------------ 3. Model analysis
elif page == "Model Analysis":
    st.title("Model Analysis")
    comp = pd.read_csv("artifacts/model_comparison.csv", index_col=0)
    st.subheader("Model comparison")
    #st.dataframe(comp.style.highlight_max(axis=0, color="#d4edda"))
    st.dataframe(comp)
    meta = json.load(open("artifacts/final_meta.json"))
    a, b = st.columns(2)
    cm = pd.DataFrame(meta["confusion_matrix"], index=["Actual Stay", "Actual Churn"], columns=["Pred Stay", "Pred Churn"])
    a.subheader("Confusion matrix (final model)")
    a.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues"), width="stretch")
    fi = pd.read_csv("artifacts/feature_importance.csv", index_col=0)["importance"].head(15).sort_values()
    b.subheader("Feature importance (top 15)")
    b.plotly_chart(px.bar(fi, orientation="h"), width="stretch")

# ------------------------------------------------ 4. Insights
else:
    st.title("Business Insights")
    st.markdown("""
1. **Contract type is the strongest driver**: month-to-month customers churn far more than 1/2-year customers.
2. **New customers are most at risk**: churn is concentrated in the first months of tenure.
3. **Fiber optic customers churn more**, pointing to a price or service-quality issue.
4. **No Online Security / Tech Support** is associated with higher churn: bundling these may help retention.
5. **Electronic check users churn more**: push automatic payment methods.
6. **Higher monthly charges** are associated with churn.

**How the model helps:** it scores every customer with a churn probability. The retention team can contact the
highest-risk customers first with targeted offers (contract upgrades, support bundles, discounts).
The final model is tuned for **recall**, because missing a churner (false negative) costs more than an unnecessary offer (false positive).
""")