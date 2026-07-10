import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# -------------------
# Needed to unpickle model.pkl (it contains this custom step)
# -------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        eps = 1e-6
        X['Loan_to_Income'] = X['loan_amount'] / (X['income'] + 1)
        X['Loan_to_Property'] = X['loan_amount'] / (X['property_value'] + 1)
        X['Debt_to_Income'] = X['dtir1'] / (X['income'] + 1)
        X['Score_LTV_ratio'] = X['Credit_Score'] / X['LTV'].clip(lower=eps)
        X['Interest_x_LTV'] = X['rate_of_interest'] * X['LTV'] / 100
        for col in ['loan_amount', 'property_value', 'income', 'rate_of_interest']:
            X[col] = np.log1p(X[col].clip(lower=0))
        return X

# -------------------
# Load model
# -------------------
model = pickle.load(open("model.pkl", "rb"))

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="Loan Default Prediction", layout="centered")

st.markdown("""
<style>
h1 { color: #D4AF37; }
div.stButton > button {
    background-color: #D4AF37;
    color: #111111;
    font-weight: 700;
    border: none;
}
div.stButton > button:hover {
    background-color: #b8952e;
    color: #111111;
}
</style>
""", unsafe_allow_html=True)

st.title("🏦 Loan Default Prediction")
st.write("Enter the loan and borrower information to predict default risk.")

# -------------------
# User Input
# -------------------
loan_amount = st.number_input("Loan Amount", 1000.0, 4000000.0, 296500.0)
income = st.number_input("Annual Income", 0.0, 500000.0, 6500.0)
property_value = st.number_input("Property Value", 10000.0, 5000000.0, 398000.0)
credit_score = st.number_input("Credit Score", 500, 900, 700)
rate_of_interest = st.number_input("Interest Rate %", 0.0, 10.0, 3.75)
ltv = st.number_input("Loan-to-Value %", 0.0, 200.0, 85.0)
term = st.number_input("Term (months)", 12, 480, 360)
dtir1 = st.number_input("Debt-to-Income %", 0.0, 100.0, 37.0)
age = st.selectbox("Borrower Age", ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"], index=1)

gender = st.selectbox("Gender", ["Male", "Female", "Joint", "Sex Not Available"])
loan_type = st.selectbox("Loan Type", ["type1", "type2", "type3"])
loan_purpose = st.selectbox("Loan Purpose", ["p1", "p2", "p3", "p4"])
occupancy_type = st.selectbox("Occupancy Type", ["ir", "pr", "sr"])
total_units = st.selectbox("Total Units", ["1U", "2U", "3U", "4U"])
credit_type = st.selectbox("Credit Type", ["CIB", "CRIF", "EQUI", "EXP"])
region = st.selectbox("Region", ["North", "North-East", "central", "south"])

loan_limit = st.selectbox("Loan Limit", ["cf", "ncf"])
approv_in_adv = st.selectbox("Approved in Advance", ["nopre", "pre"])
neg_ammortization = st.selectbox("Negative Amortization", ["neg_amm", "not_neg"])
submission = st.selectbox("Submission of Application", ["not_inst", "to_inst"])

credit_worthiness = st.selectbox("Credit Worthiness", ["l1", "l2"])
open_credit = st.selectbox("Open Credit", ["nopc", "opc"])
business_or_commercial = st.selectbox("Business or Commercial", ["b/c", "nob/c"])
interest_only = st.selectbox("Interest Only", ["int_only", "not_int"])
lump_sum_payment = st.selectbox("Lump Sum Payment", ["lpsm", "not_lpsm"])
construction_type = st.selectbox("Construction Type", ["mh", "sb"])
secured_by = st.selectbox("Secured By", ["home", "land"])
co_applicant_credit_type = st.selectbox("Co-applicant Credit Type", ["CIB", "EXP"])
security_type = st.selectbox("Security Type", ["Indriect", "direct"])

# -------------------
# Encoding maps (must match the training script's LabelEncoder order)
# -------------------
binary_map = {
    "l1": 0, "l2": 1,
    "nopc": 0, "opc": 1,
    "b/c": 0, "nob/c": 1,
    "int_only": 0, "not_int": 1,
    "lpsm": 0, "not_lpsm": 1,
    "mh": 0, "sb": 1,
    "home": 0, "land": 1,
    "CIB": 0, "EXP": 1,
    "Indriect": 0, "direct": 1,
    "cf": 0, "ncf": 1,
    "nopre": 0, "pre": 1,
    "neg_amm": 0, "not_neg": 1,
    "not_inst": 0, "to_inst": 1,
}


def age_to_number(age_str):
    if "-" in age_str:
        lo, hi = age_str.split("-")
        return (int(lo) + int(hi)) / 2
    elif "<" in age_str:
        return 20.0
    elif ">" in age_str:
        return 75.0
    return float(age_str)


# Convert to DataFrame (raw columns, exactly as the training script uses them)
input_df = pd.DataFrame([{
    "loan_limit": binary_map[loan_limit],
    "Gender": gender,
    "approv_in_adv": binary_map[approv_in_adv],
    "loan_type": loan_type,
    "loan_purpose": loan_purpose,
    "Credit_Worthiness": binary_map[credit_worthiness],
    "open_credit": binary_map[open_credit],
    "business_or_commercial": binary_map[business_or_commercial],
    "loan_amount": loan_amount,
    "rate_of_interest": rate_of_interest,
    "term": term,
    "Neg_ammortization": binary_map[neg_ammortization],
    "interest_only": binary_map[interest_only],
    "lump_sum_payment": binary_map[lump_sum_payment],
    "property_value": property_value,
    "construction_type": binary_map[construction_type],
    "occupancy_type": occupancy_type,
    "Secured_by": binary_map[secured_by],
    "total_units": total_units,
    "income": income,
    "credit_type": credit_type,
    "Credit_Score": credit_score,
    "co-applicant_credit_type": binary_map[co_applicant_credit_type],
    "age": age_to_number(age),
    "submission_of_application": binary_map[submission],
    "LTV": ltv,
    "Region": region,
    "Security_Type": binary_map[security_type],
    "dtir1": dtir1,
}])

# -------------------
# Prediction
# -------------------
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.write("---")
    st.write(f"**Prediction:** {'🔥 Likely to Default' if pred == 1 else '🟢 Not Likely to Default'}")
    st.write(f"**Probability:** {proba:.2f}")
    if proba >= 0.5:
        st.warning("This borrower is at higher risk of default.")
    else:
        st.success("This borrower is at low risk of default.")
