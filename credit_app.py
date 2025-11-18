import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# ------------------- Streamlit Page Config -------------------
st.set_page_config(page_title="Credit Approval Predictor", layout="wide")
st.title("✅ Credit Approval Predictor (UCI Australian Dataset)")

MODEL_PATH = "model_rf.joblib"

# ------------------- Train / Load Model -------------------
@st.cache_resource
def load_or_train_model():
    if not os.path.exists(MODEL_PATH):
        # Load dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat"
        df = pd.read_csv(url, sep=" ", header=None)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Categorical & numerical column indices
        cat_idx = [0, 3, 4, 5, 7, 8, 10, 11]
        num_idx = list(set(range(14)) - set(cat_idx))

        preprocessor = ColumnTransformer([
            ("num", SimpleImputer(strategy="mean"), num_idx),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("enc", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_idx)
        ])

        model = Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_proba),
            "f1": f1_score(y_test, y_pred),
        }

        joblib.dump(model, MODEL_PATH)
        return model, metrics
    else:
        model = joblib.load(MODEL_PATH)
        return model, None

model, metrics = load_or_train_model()

if metrics is not None:
    st.success(
        f"Model trained: Accuracy = {metrics['accuracy']:.2f}, "
        f"AUC = {metrics['auc']:.2f}, F1 = {metrics['f1']:.2f}"
    )
else:
    st.info("Loaded existing trained model from disk.")

# ------------------- Session State Setup -------------------
if "step" not in st.session_state:
    st.session_state.step = -1  # -1 = applicant info, 0..13 = features
if "answers" not in st.session_state:
    st.session_state.answers = [0.0] * 14
if "finished" not in st.session_state:
    st.session_state.finished = False
if "decision" not in st.session_state:
    st.session_state.decision = None
if "probability" not in st.session_state:
    st.session_state.probability = None
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "user_sex" not in st.session_state:
    st.session_state.user_sex = ""
if "user_email" not in st.session_state:
    st.session_state.user_email = ""

# ------------------- Reset Function -------------------
def reset_application():
    st.session_state.step = -1
    st.session_state.answers = [0.0] * 14
    st.session_state.finished = False
    st.session_state.decision = None
    st.session_state.probability = None
    st.session_state.user_name = ""
    st.session_state.user_sex = ""
    st.session_state.user_email = ""

# ------------------- Sidebar Controls -------------------
st.sidebar.header("Application Controls")
if st.sidebar.button("🔄 Start New Application"):
    reset_application()
    st.sidebar.success("New application started.")

if st.session_state.step >= 0:
    st.sidebar.markdown(f"**Current question:** {st.session_state.step + 1} / 14")
else:
    st.sidebar.markdown("**Current step:** Applicant Info")

# ------------------- Helper: UI for each feature -------------------
def ask_feature(step: int, current_value: float):
    """
    Returns the numeric value for the given step based on user input.
    Mapping from UI options -> numeric codes / representative values.
    """
    # 0: Married (A1)
    if step == 0:
        options = ["Married", "Not married"]
        mapping = {"Married": 1.0, "Not married": 0.0}
        default_idx = 0
        for i, o in enumerate(options):
            if mapping[o] == current_value:
                default_idx = i
        choice = st.selectbox(
            "Marital status",
            options,
            index=default_idx,
            help="Select if you are married or not."
        )
        return mapping[choice]

    # 1: Age bucket (A2)
    elif step == 1:
        options = ["18–25", "26–45", "46–80"]
        # Use midpoints as numeric values
        mapping = {
            "18–25": 21.0,
            "26–45": 35.0,
            "46–80": 60.0,
        }
        # Find default by nearest
        default_idx = 0
        for i, o in enumerate(options):
            if mapping[o] == current_value:
                default_idx = i
        choice = st.selectbox(
            "Age range",
            options,
            index=default_idx,
            help="Select your age range."
        )
        return mapping[choice]

    # 2: Existing Debt (A3)
    elif step == 2:
        return st.number_input(
            "Existing debt amount",
            value=float(current_value),
            min_value=0.0,
            step=100.0,
            help="Enter your total existing debt amount."
        )

    # 3: Employment Category (A4)
    elif step == 3:
        options = ["Student", "IT", "Healthcare", "Civil", "Others"]
        mapping = {o: float(i) for i, o in enumerate(options)}  # 0..4
        default_idx = 0
        for i, o in enumerate(options):
            if mapping[o] == current_value:
                default_idx = i
        choice = st.selectbox(
            "Employment category",
            options,
            index=default_idx,
            help="Select the category that best describes your employment."
        )
        return mapping[choice]

    # 4: Loan purpose (A5)
    elif step == 4:
        options = ["Education loan", "Personal loan", "Other reason not stated"]
        mapping = {o: float(i) for i, o in enumerate(options)}  # 0..2
        default_idx = 0
        for i, o in enumerate(options):
            if mapping[o] == current_value:
                default_idx = i
        choice = st.selectbox(
            "Loan purpose",
            options,
            index=default_idx,
            help="Select the primary purpose of this loan."
        )
        return mapping[choice]

    # 5: Account type (A6)
    elif step == 5:
        options = ["Savings account", "Current account", "Salary account", "Joint account", "Other"]
        mapping = {o: float(i) for i, o in enumerate(options)}  # 0..4
        default_idx = 0
        for i, o in enumerate(options):
            if mapping[o] == current_value:
                default_idx = i
        choice = st.selectbox(
            "Account type",
            options,
            index=default_idx,
            help="Select the main type of bank account you use."
        )
        return mapping[choice]

    # 6: Years of employment (A7)
    elif step == 6:
        return st.number_input(
            "Years of employment",
            value=float(current_value),
            min_value=0.0,
            step=1.0,
            help="Enter the number of years you have been employed."
        )

    # 7: Occupation / Job class (A8)
    elif step == 7:
        options = ["Entry-level", "Mid-level professional", "Senior professional", "Self-employed", "Other"]
        mapping = {o: float(i) for i, o in enumerate(options)}  # 0..4
        default_idx = 0
        for i, o in enumerate(options):
            if mapping[o] == current_value:
                default_idx = i
        choice = st.selectbox(
            "Occupation / Job class",
            options,
            index=default_idx,
            help="Select which best describes your occupation level."
        )
        return mapping[choice]

    # 8: Housing Status (A9)
    elif step == 8:
        options = ["Rent", "Own house property"]
        mapping = {"Rent": 0.0, "Own house property": 1.0}
        default_idx = 0
        for i, o in enumerate(options):
            if mapping[o] == current_value:
                default_idx = i
        choice = st.selectbox(
            "Housing status",
            options,
            index=default_idx,
            help="Select whether you rent or own a house."
        )
        return mapping[choice]

    # 9: Savings / Investment amount (A10)
    elif step == 9:
        return st.number_input(
            "Savings / Investment amount",
            value=float(current_value),
            min_value=0.0,
            step=500.0,
            help="Enter the approximate total amount of your savings or investments."
        )

    # 10: Dependents in household (A11)
    elif step == 10:
        options = ["2", "3", "4", "5 or more"]
        mapping = {
            "2": 2.0,
            "3": 3.0,
            "4": 4.0,
            "5 or more": 5.0,
        }
        default_idx = 0
        for i, o in enumerate(options):
            if mapping[o] == current_value:
                default_idx = i
        choice = st.selectbox(
            "Dependents in household",
            options,
            index=default_idx,
            help="Select how many dependents live in your household."
        )
        return mapping[choice]

    # 11: Additional loan (A12)
    elif step == 11:
        options = ["No other loan", "Car loan", "Home loan", "Credit card debt", "Other"]
        mapping = {o: float(i) for i, o in enumerate(options)}  # 0..4
        default_idx = 0
        for i, o in enumerate(options):
            if mapping[o] == current_value:
                default_idx = i
        choice = st.selectbox(
            "Additional loan",
            options,
            index=default_idx,
            help="Select if you have any other active loans or credit obligations."
        )
        return mapping[choice]

    # 12: Existing credit balance (A13) -> ranges to midpoints
    elif step == 12:
        options = [
            "0 – 1,000",
            "1,001 – 5,000",
            "5,001 – 10,000",
            "10,001 – 20,000",
            "20,001+",
        ]
        mapping = {
            "0 – 1,000": 500.0,
            "1,001 – 5,000": 3000.0,
            "5,001 – 10,000": 7500.0,
            "10,001 – 20,000": 15000.0,
            "20,001+": 25000.0,
        }
        default_idx = 0
        for i, o in enumerate(options):
            if mapping[o] == current_value:
                default_idx = i
        choice = st.selectbox(
            "Existing credit balance",
            options,
            index=default_idx,
            help="Select the approximate range of your existing total credit balance."
        )
        return mapping[choice]

    # 13: Credit score (A14)
    elif step == 13:
        options = [
            "600–625",
            "626–680",
            "681–720",
            "721–800",
        ]
        mapping = {
            "600–625": 612.0,
            "626–680": 653.0,
            "681–720": 700.0,
            "721–800": 760.0,
        }
        default_idx = 0
        for i, o in enumerate(options):
            if mapping[o] == current_value:
                default_idx = i
        choice = st.selectbox(
            "Credit score range",
            options,
            index=default_idx,
            help="Select your approximate credit score range."
        )
        return mapping[choice]

    # Fallback (should not happen)
    return current_value

# ------------------- APPLICANT INFO PAGE (step = -1) -------------------
if st.session_state.step == -1 and not st.session_state.finished:
    st.subheader("👤 Applicant Information")

    st.session_state.user_name = st.text_input(
        "Applicant Name", value=st.session_state.user_name
    )
    st.session_state.user_sex = st.selectbox(
        "Sex",
        ["Male", "Female", "Other"],
        index=["Male", "Female", "Other"].index(st.session_state.user_sex)
        if st.session_state.user_sex else 0
    )
    st.session_state.user_email = st.text_input(
        "Email Address", value=st.session_state.user_email
    )

    if st.button("Start Application ➡️"):
        if st.session_state.user_name.strip() == "" or st.session_state.user_email.strip() == "":
            st.error("Please enter both your name and email address.")
        else:
            st.session_state.step = 0
            st.rerun()

# ------------------- QUESTION FLOW (steps 0–13) -------------------
if st.session_state.step >= 0 and not st.session_state.finished:
    step = st.session_state.step

    st.subheader("📝 Application Questionnaire")
    st.write("Answer each question carefully. Click **Next** to move forward.")

    current_val = st.session_state.answers[step]
    new_value = ask_feature(step, current_val)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Next ➡️"):
            st.session_state.answers[step] = float(new_value)

            if step == 13:
                # Last question -> run prediction
                input_df = pd.DataFrame([st.session_state.answers])
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]
                decision = "✅ APPROVED" if prediction == 1 else "⛔ DENIED"

                st.session_state.decision = decision
                st.session_state.probability = float(probability)
                st.session_state.finished = True
            else:
                st.session_state.step += 1

            st.rerun()

    with col2:
        if step > 0:
            if st.button("⬅️ Previous"):
                st.session_state.answers[step] = float(new_value)
                st.session_state.step -= 1
                st.rerun()

# ------------------- RESULT PAGE -------------------
if st.session_state.finished:
    name = st.session_state.user_name
    sex = st.session_state.user_sex
    email = st.session_state.user_email
    decision = st.session_state.decision
    probability = st.session_state.probability

    st.subheader("📌 Final Decision")

    if "APPROVED" in decision:
        st.success(
            f"🎉 Congratulations {name}! Your credit application has been **APPROVED**.\n\n"
            f"**Approval Probability:** `{probability:.3f}`\n\n"
            f"An email confirmation will be sent to: **{email}**."
        )
    else:
        st.error(
            f"⛔ Sorry {name}, your credit application has been **DENIED**.\n\n"
            f"**Approval Probability:** `{probability:.3f}`"
        )

        st.write("### 💡 Tips to Improve / Maintain Your Credit Score")
        st.markdown("""
1. **Pay bills on time** – Payment history is the biggest factor in your score.  
2. **Reduce credit utilization below 30%** – Try to keep used credit well below your limit.  
3. **Avoid too many new applications** – Each hard inquiry can slightly reduce your score.  
4. **Keep older accounts open** – Older accounts help build a longer credit history.  
5. **Check your credit reports regularly** – Correct any errors or fraudulent entries early.
        """)

            # ---------------- MODEL EXPLANATION (Feature Importances) -------------------
    st.write("### 🔍 Top Feature Contributions (Model Feature Importances)")

    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]

    # Get transformed feature names
    try:
        feature_names = prep.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(clf.n_features_in_)]

    importances = clf.feature_importances_

    # Build a DataFrame for top features
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })
    fi = fi.sort_values("importance", ascending=False).head(5)

    fig, ax = plt.subplots()
    ax.barh(fi["feature"], fi["importance"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Top 5 features affecting approval decision")
    st.pyplot(fig)
