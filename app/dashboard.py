import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.fetch_data import fetch_data
from modeling.preprocess import clean_data, TARGET, SEED

st.set_page_config(
    page_title="Stroke Prediction App",
    page_icon="🧠",
    layout="wide"
)

page = st.sidebar.selectbox(
    "Navigation",
    ["EDA", "Model Results", "Risk Predictor"]
)

st.title("🧠 Stroke Prediction Dashboard")
st.markdown("---")


@st.cache_data
def load_data():
    df = fetch_data()
    df = clean_data(df)
    return df

@st.cache_resource
def load_model():
    return joblib.load("modeling/artifacts/best_model.pkl")

@st.cache_data
def get_test_data():
    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    return X_test, y_test

df = load_data()
model = load_model()
X_test_split, y_test_split = get_test_data()
y_prob = model.predict_proba(X_test_split)[:, 1]


if page == "EDA":
    st.header("Exploratory Data Analysis")

    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients", len(df))
    col2.metric("Stroke Cases", int(df[TARGET].sum()))
    col3.metric("Stroke Rate", f"{df[TARGET].mean()*100:.1f}%")
    st.markdown("""
    **📦 Data Source**  
    - **Dataset:** Stroke Prediction Dataset  
    - **Author:** fedesoriano  
    - **Source:** [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)  
    - **License:** CC BY-SA 4.0 — open for academic use  
    - **Collection:** Real-world clinical data, 5,110 patients  
    """)

    st.subheader("Target Distribution")
    df_pie = df.copy()
    df_pie[TARGET] = df_pie[TARGET].map({0: "No Stroke", 1: "Stroke"})
    fig = px.pie(df_pie, names=TARGET, title="Stroke Distribution",
                 color_discrete_sequence=["#4A90D9", "#E8E8E8"])
    st.plotly_chart(fig, use_container_width=True)
    st.info("⚠️ Only 4.9% of patients had a stroke — a severe class imbalance. A naive model always predicting 'no stroke' would achieve 95% accuracy but miss every real case.")
    st.markdown("""
**What we see:** 95.1% no stroke vs 4.9% stroke — extreme imbalance.

**Interesting finding:** A model that never predicts stroke would score 95% accuracy — yet be completely useless clinically.

**Implication:** Accuracy is a misleading metric here. We use **F1, Recall, and ROC-AUC** instead.
""")

    st.subheader("Age vs Stroke")
    fig = px.histogram(df, x="age", color=TARGET, barmode="overlay", opacity=0.7,
                       title="Age Distribution by Stroke",
                       labels={"age": "Age", "count": "Count", TARGET: "Stroke"})
    st.plotly_chart(fig, use_container_width=True)
    st.info("📊 Stroke cases are almost exclusively seen in patients over 40. Age is the strongest predictor in our dataset — confirmed later by feature importance (0.20).")
    st.markdown("""
**What we see:** Stroke cases concentrate heavily after age 40 and peak around 70-80.

**Interesting finding:** Almost zero stroke cases below age 40 — a near-perfect natural threshold.

**Open question:** Is the risk increase gradual or does it jump at certain age groups?
""")

    st.subheader("Stroke Rate by Age")
    age_stroke = df.groupby("age")[TARGET].mean().reset_index()
    fig = px.line(age_stroke, x="age", y=TARGET, title="Stroke Rate by Age",
                  labels={"age": "Age", TARGET: "Stroke Rate"})
    st.plotly_chart(fig, use_container_width=True)
    st.info("📈 Stroke risk increases dramatically after age 60, reaching 25% at age 80. This non-linear pattern confirms that raw age alone may not fully capture the risk.")
    st.markdown("""
**What we see:** Risk stays near zero until age 40, then rises sharply after 60, reaching ~25% at age 80.

**Interesting finding:** The relationship is non-linear — risk accelerates, it doesn't grow steadily.

**Implication:** Grouping age into Young (0-40) / Middle (40-60) / Senior (60+) captures this pattern better than treating age as a continuous number.
""")

    st.subheader("Numerical Variables vs Stroke")
    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.box(df, x=TARGET, y="age", title="Age vs Stroke")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(df, x=TARGET, y="avg_glucose_level", title="Glucose vs Stroke")
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        fig = px.box(df, x=TARGET, y="bmi", title="BMI vs Stroke")
        st.plotly_chart(fig, use_container_width=True)
    st.info("📦 Age shows the clearest separation (median 43 vs 70). Glucose is higher and more variable in stroke patients. BMI shows minimal difference (28 vs 30).")
    st.markdown("""
**What we see:** Age → 27-year median gap. Glucose → wider spread in stroke group. BMI → almost identical distributions.

**Interesting finding:** Glucose variability in stroke patients is striking — not just higher on average, but much more spread out, suggesting a diabetic subgroup.

**Open question:** Does BMI interact with age or glucose to become a stronger predictor in combination?
""")

    st.subheader("Categorical Variables vs Stroke")
    cat_cols = ["gender", "ever_married", "work_type", "smoking_status"]
    for i in range(0, len(cat_cols), 2):
        col1, col2 = st.columns(2)
        with col1:
            col = cat_cols[i]
            stroke_rate = df.groupby(col)[TARGET].mean().reset_index()
            fig = px.bar(stroke_rate, x=col, y=TARGET, title=f"Stroke Rate by {col}",
                         labels={TARGET: "Stroke Rate"})
            st.plotly_chart(fig, use_container_width=True)
        if i + 1 < len(cat_cols):
            with col2:
                col = cat_cols[i + 1]
                stroke_rate = df.groupby(col)[TARGET].mean().reset_index()
                fig = px.bar(stroke_rate, x=col, y=TARGET, title=f"Stroke Rate by {col}",
                             labels={TARGET: "Stroke Rate"})
                st.plotly_chart(fig, use_container_width=True)
    st.info("📊 Gender shows minimal difference. Ever-married appears higher but is likely a confounding age effect. Work type and smoking status show more meaningful variation.")
    st.markdown("""
**What we see:** Female 4.7% vs Male 5.1% — nearly identical. Ever-married 6.5% vs 1.6% — large gap. Self-employed has highest work-type rate (8%).

**Interesting finding:** Ever-married stroke rate is 4x higher — but older people are both more likely to be married and to have strokes. This is a **confounding effect**, not a causal relationship.

**Open question:** If we control for age, does ever-married still predict stroke?
""")

    st.subheader("Risk Factors vs Stroke")
    col1, col2 = st.columns(2)
    with col1:
        stroke_rate = df.groupby("hypertension")[TARGET].mean().reset_index()
        stroke_rate["hypertension"] = stroke_rate["hypertension"].map({0: "No", 1: "Yes"})
        fig = px.bar(stroke_rate, x="hypertension", y=TARGET,
                     title="Stroke Rate by Hypertension",
                     labels={TARGET: "Stroke Rate", "hypertension": "Hypertension"},
                     category_orders={"hypertension": ["No", "Yes"]})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        stroke_rate = df.groupby("heart_disease")[TARGET].mean().reset_index()
        stroke_rate["heart_disease"] = stroke_rate["heart_disease"].map({0: "No", 1: "Yes"})
        fig = px.bar(stroke_rate, x="heart_disease", y=TARGET,
                     title="Stroke Rate by Heart Disease",
                     labels={TARGET: "Stroke Rate", "heart_disease": "Heart Disease"},
                     category_orders={"heart_disease": ["No", "Yes"]})
        st.plotly_chart(fig, use_container_width=True)
    st.info("🏥 Hypertension triples stroke risk (4% → 13%) and heart disease quadruples it (4% → 17%). Both are well-established clinical risk factors. But what happens when they co-occur?")
    st.markdown("""
**What we see:** Hypertension: 4% → 13%. Heart disease: 4% → 17%.

**Interesting finding:** These are among the most clinically validated stroke risk factors — our data confirms the literature.

**Open question:** Do these two conditions amplify each other's effect when present together?
""")

    both = df.copy()
    both["both_conditions"] = ((both["hypertension"] == 1) & (both["heart_disease"] == 1)).astype(int)
    both_rate = both.groupby("both_conditions")[TARGET].mean().reset_index()
    both_rate["both_conditions"] = both_rate["both_conditions"].map({0: "No Comorbidity", 1: "Both Conditions"})
    fig = px.bar(both_rate, x="both_conditions", y=TARGET,
                 title="Compounding Effect: Hypertension + Heart Disease",
                 labels={TARGET: "Stroke Rate", "both_conditions": ""})
    st.plotly_chart(fig, use_container_width=True)
    st.info("⚡ Patients with both conditions show a 20.3% stroke rate — more than 4x higher than those with neither (4.7%). Only 64 patients (1.3%) had both, yet they are a critically high-risk group.")
    st.markdown("""
**What we see:** No comorbidity → 4.7%. Both conditions → 20.3%.

**Interesting finding:** The combined effect suggests a **compounding interaction**, not just additive risk.

**Implication:** Tree-based models (Random Forest, XGBoost) can naturally capture this interaction.
""")

    st.subheader("BMI Missing Value Analysis")
    col1, col2 = st.columns(2)
    with col1:
        bmi_missing = df.copy()
        bmi_missing["bmi_missing"] = bmi_missing["bmi"].isnull().astype(int)
        stroke_rate = bmi_missing.groupby("bmi_missing")[TARGET].mean().reset_index()
        stroke_rate["bmi_missing"] = stroke_rate["bmi_missing"].map({0: "BMI Present", 1: "BMI Missing"})
        fig = px.bar(stroke_rate, x="bmi_missing", y=TARGET,
                     title="Stroke Rate by BMI Missingness",
                     labels={TARGET: "Stroke Rate", "bmi_missing": "BMI Status"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        bmi_missing["age_group"] = pd.cut(
            bmi_missing["age"], bins=[0, 40, 60, 120],
            labels=["Young (0-40)", "Middle (40-60)", "Senior (60+)"]
        )
        missing_by_age = bmi_missing.groupby("age_group", observed=False)["bmi_missing"].mean().reset_index()
        fig = px.bar(missing_by_age, x="age_group", y="bmi_missing",
                     title="BMI Missing Rate by Age Group",
                     labels={"bmi_missing": "Missing Rate", "age_group": "Age Group"})
        st.plotly_chart(fig, use_container_width=True)
    st.info("🔍 BMI missing values are NOT random — patients with missing BMI have 5x higher stroke rate (19.9% vs 4.2%) and are 9 years older on average (52 vs 43).")
    st.markdown("""
**What we see:** BMI Present → 4.2% stroke. BMI Missing → 19.9% stroke. Senior group has the highest missing rate (6.7%).

**Interesting finding:** The missingness itself is informative — this is called **Missing Not At Random (MNAR)**.

**Implication:** Age-based median imputation is more appropriate than simple median. A `bmi_missing` flag carries predictive signal.
""")

    st.subheader("Glucose Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="avg_glucose_level", color=TARGET,
                           barmode="overlay", opacity=0.7,
                           title="Glucose Distribution by Stroke",
                           labels={"avg_glucose_level": "Avg Glucose Level", TARGET: "Stroke"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        df["glucose_cat"] = pd.cut(df["avg_glucose_level"], bins=[0, 100, 125, 500],
                                   labels=["Normal", "Pre-diabetic", "Diabetic"])
        stroke_rate = df.groupby("glucose_cat", observed=False)[TARGET].mean().reset_index()
        fig = px.bar(stroke_rate, x="glucose_cat", y=TARGET,
                     title="Stroke Rate by Glucose Category",
                     labels={TARGET: "Stroke Rate", "glucose_cat": "Glucose Category"})
        st.plotly_chart(fig, use_container_width=True)
    st.info("🩸 Glucose is right-skewed with a secondary peak at 200+ mg/dL. Diabetic patients (glucose > 125) show 3x higher stroke rate (10%) vs normal (3.5%).")
    st.markdown("""
**What we see:** Two peaks — normal (80-100 mg/dL) and diabetic subgroup (200+ mg/dL).

**Interesting finding:** Pre-diabetic and normal groups have nearly identical stroke rates — risk only jumps at the diabetic threshold (>125 mg/dL).

**Note:** Thresholds follow **American Diabetes Association** clinical standards — not arbitrary cuts.
""")

    st.subheader("Smoking Status Analysis")
    col1, col2 = st.columns(2)
    with col1:
        stroke_rate = df.groupby("smoking_status")[TARGET].mean().reset_index()
        fig = px.bar(stroke_rate, x="smoking_status", y=TARGET,
                     title="Stroke Rate by Smoking Status",
                     labels={TARGET: "Stroke Rate", "smoking_status": "Smoking Status"},
                     category_orders={"smoking_status": ["never smoked", "formerly smoked", "smokes", "Unknown"]})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        smoking_count = df["smoking_status"].value_counts().reset_index()
        smoking_count.columns = ["smoking_status", "count"]
        fig = px.bar(smoking_count, x="smoking_status", y="count",
                     title="Smoking Status Distribution",
                     labels={"count": "Count", "smoking_status": "Smoking Status"})
        st.plotly_chart(fig, use_container_width=True)
    st.info("🚬 'Formerly smoked' paradoxically has the highest stroke rate (8%) — even higher than current smokers (5%). The Unknown group (30% of data) has the lowest rate (3%).")
    st.markdown("""
**What we see:** Formerly smoked 8% > Smokes 5.3% > Never smoked 4.7% > Unknown 3%.

**Interesting finding:** Reverse causality — patients who already had health problems likely quit smoking. This is a well-known epidemiological phenomenon.

**Open question:** Are Unknown patients systematically younger/healthier? Future work could investigate this.
""")

    st.subheader("BMI Outlier Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(df, y="bmi", title="BMI Distribution (with outliers)", labels={"bmi": "BMI"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("Patients with BMI > 60", int((df["bmi"] > 60).sum()))
        st.metric("Max BMI", float(df["bmi"].max()))
        st.markdown("""
**What we see:** 13 patients have BMI > 60, with a maximum of 97.6.

**Interesting finding:** Extreme values appear predominantly in the non-stroke group — extreme BMI alone may not be the decisive factor.

**Why we kept them:**
- Medically possible (severe obesity exists clinically)
- Only 13 patients — removing them loses real data
- Tree-based models are robust to outliers by design
""")

    st.subheader("Correlation Matrix")
    corr_cols = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease", TARGET]
    corr_matrix = df[corr_cols].corr()
    fig = px.imshow(corr_matrix, title="Correlation Matrix",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=".2f")
    st.plotly_chart(fig, use_container_width=True)
    st.info("🔗 Age has the highest correlation with stroke (0.25). No feature pair exceeds 0.33 — no multicollinearity concern.")
    st.markdown("""
**What we see:** Age → stroke: 0.25 (highest). BMI → stroke: 0.04 (lowest).

**Interesting finding:** Despite age being the strongest predictor, its correlation (0.25) is still modest — stroke is a multi-factorial condition.

**Implication:** No multicollinearity issues — all features can safely enter the model together.
""")


elif page == "Model Results":
    st.header("Model Results")

    # ── 1. Preprocessing Pipeline ─────────────────────────────────────────────
    st.subheader("1. Preprocessing Pipeline")
    st.markdown("Before any model sees the data, it goes through a carefully designed pipeline:")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("""
**Step-by-step:**

1. **Fetch Data** → Kaggle API (automatic)
2. **Clean Data**
   - Drop `id` column (no predictive value)
   - Remove `gender = Other` (only 1 patient)
3. **Feature Engineering**
   - `age_group`: Young / Middle / Senior
   - `glucose_category`: Normal / Pre-diabetic / Diabetic
   - `bmi_missing`: flag for missing BMI (MNAR)
4. **Train/Test Split** (80/20, stratified)
5. **AgeBasedBMIImputer** (custom transformer)
   - Age-group median imputation
   - Prevents data leakage
6. **ColumnTransformer**
   - Numeric → StandardScaler
   - Binary → OrdinalEncoder
   - Categorical → OneHotEncoder
7. **TunedThresholdClassifierCV**
   - Optimizes decision threshold for F1
        """)
    with col2:
        pipeline_steps = {
            "Step": ["Fetch Data", "Drop id & Other", "Feature Engineering",
                     "Train/Test Split", "BMI Imputation", "Encoding & Scaling", "Threshold Tuning"],
            "Method": ["Kaggle API", "Rule-based", "Domain knowledge",
                       "Stratified 80/20", "Age-based median (custom)",
                       "OHE + StandardScaler", "TunedThresholdClassifierCV"],
            "Why": ["Automated, no login", "No signal, too rare", "Non-linear relationships",
                    "Preserve imbalance ratio", "MNAR pattern detected",
                    "Required for ML models", "Imbalanced classification"]
        }
        st.dataframe(pd.DataFrame(pipeline_steps), use_container_width=True)
    st.info("🔒 All transformations happen INSIDE the pipeline — fitted only on train data. This prevents data leakage.")

    # ── 1.5 Decision Chain ───────────────────────────────────────────────────
    st.subheader("1.5 From EDA to Model — Decision Chain")
    st.markdown("""
Every preprocessing and modeling decision was motivated by EDA findings:
    """)

    chain_data = {
        "EDA Finding": [
            "Age shows non-linear risk increase after 60",
            "BMI missingness correlates with age & stroke (MNAR)",
            "Glucose risk jumps only at diabetic threshold (>125)",
            "Hypertension + Heart Disease compound effect (20.3%)",
            "95/5 class imbalance — accuracy misleading",
            "BMI outliers (max=97.6) medically plausible",
            "Smoking 'Unknown' = 30% of data",
        ],
        "Decision Made": [
            "age_group feature: Young / Middle / Senior",
            "AgeBasedBMIImputer + bmi_missing flag",
            "glucose_category: Normal / Pre-diabetic / Diabetic",
            "Tree-based models (RF, XGBoost) capture interactions",
            "TunedThresholdClassifierCV + F1/Recall metrics",
            "Keep outliers — tree models are robust",
            "Keep Unknown as separate category",
        ],
        "Where Applied": [
            "preprocess.py → clean_data()",
            "preprocess.py → AgeBasedBMIImputer",
            "preprocess.py → clean_data()",
            "train.py → model selection",
            "train.py → TunedThresholdClassifierCV",
            "preprocess.py → no outlier removal",
            "preprocess.py → OHE keeps Unknown",
        ],
        "Validated By": [
            "Feature importance: age rank 1 (0.202)",
            "Feature importance: bmi_missing rank 4 (0.088)",
            "Stroke rate: Diabetic 10% vs Normal 3.5%",
            "Compounding effect confirmed in EDA",
            "DummyClassifier F1=0.000 proves baseline useless",
            "Outliers in non-stroke group — no removal needed",
            "Unknown stroke rate 3% — distinct behavior",
        ]
    }

    st.dataframe(pd.DataFrame(chain_data), use_container_width=True)
    st.info("🔗 Every decision has a clear trail: EDA observation → preprocessing/modeling choice → validation. This is the scientific method applied to data science.")

    # ── 2. Model Selection ────────────────────────────────────────────────────
    st.subheader("2. Why These Models?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Models we chose and why:**

✅ **Logistic Regression**
- Simple, interpretable baseline
- Good for understanding feature coefficients
- Common in medical literature

✅ **Random Forest**
- Handles non-linear relationships
- Robust to outliers (important for our BMI outliers)
- Captures feature interactions automatically
- Our best performing model

✅ **XGBoost**
- Industry standard for tabular data
- Gradient boosting corrects previous errors
- Optimized with Optuna (30 trials)
        """)
    with col2:
        st.markdown("""
**Models we rejected and why:**

❌ **SVM** — Slow on 5000+ rows, poor on imbalanced data

❌ **KNN** — Very slow at prediction, suffers in high dimensions

❌ **Neural Networks** — 5,110 rows too small (overfitting risk), black box for medical use

**Why TunedThresholdClassifierCV?**
- Default threshold (0.5) is wrong for 95/5 imbalance
- Automatically finds the optimal threshold for F1
- Clinically motivated: missing a stroke is worse than a false alarm
        """)

    # ── 3. Why These Metrics? ─────────────────────────────────────────────────
    st.subheader("3. Why These Metrics?")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.error("❌ Accuracy")
        st.markdown("A model always saying 'no stroke' gets **95% accuracy** — but misses every real stroke. Useless clinically.")
    with col2:
        st.success("✅ Recall")
        st.markdown("**Most important.** Of all real stroke patients, how many did we catch? Missing = no treatment = life-threatening.")
    with col3:
        st.success("✅ F1 Score")
        st.markdown("Balance between Recall and Precision. We optimize threshold for F1.")
    with col4:
        st.success("✅ ROC-AUC")
        st.markdown("Threshold-independent. Probability model ranks stroke patient above non-stroke. Our models ~0.85.")

    # ── 4. Cross-Validation Strategy ─────────────────────────────────────────
    st.subheader("4. Cross-Validation Strategy")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Why Stratified K-Fold (5 folds)?**
- Our data has 95/5 imbalance — random split could put all strokes in one fold
- Stratified ensures each fold has ~4.9% stroke cases
- 5 folds = 5 different evaluations → averaged for reliability
- Single split too noisy with only 249 stroke cases
        """)
    with col2:
        cv_data = {
            "Fold": ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Mean"],
            "Train Stroke %": ["~4.9%", "~4.9%", "~4.9%", "~4.9%", "~4.9%", "4.9%"],
            "Test Stroke %": ["~4.9%", "~4.9%", "~4.9%", "~4.9%", "~4.9%", "4.9%"],
        }
        st.dataframe(pd.DataFrame(cv_data), use_container_width=True)
        st.caption("Stratified split preserves the 4.9% stroke rate in every fold.")


    st.subheader("4.1 Cross-Validation Fold Results")
    import json
    try:
        with open("modeling/artifacts/results.json", "r") as f:
            cv_results = json.load(f)

        rf_folds = cv_results.get("RandomForest", {}).get("fold_f1_scores", [])
        lr_folds = cv_results.get("LogisticRegression", {}).get("fold_f1_scores", [])
        xgb_folds = cv_results.get("XGBoost", {}).get("fold_f1_scores", [])

        if rf_folds:
            fold_data = {
                "Fold": [f"Fold {i+1}" for i in range(5)] + ["Mean", "Std"],
                "Random Forest F1": [round(x, 3) for x in rf_folds] + [round(sum(rf_folds)/5, 3), round(pd.Series(rf_folds).std(), 3)],
                "LogReg F1": [round(x, 3) for x in lr_folds] + [round(sum(lr_folds)/5, 3), round(pd.Series(lr_folds).std(), 3)],
                "XGBoost F1": [round(x, 3) for x in xgb_folds] + [round(sum(xgb_folds)/5, 3), round(pd.Series(xgb_folds).std(), 3)],
            }
            st.dataframe(pd.DataFrame(fold_data), use_container_width=True)

            fig = px.line(
                x=[f"Fold {i+1}" for i in range(5)],
                y=[rf_folds, lr_folds, xgb_folds],
                title="F1 Score per Fold — All Models",
                labels={"x": "Fold", "value": "F1 Score"}
            )
            fig.data[0].name = "Random Forest"
            fig.data[1].name = "Logistic Regression"
            fig.data[2].name = "XGBoost"
            fig.update_layout(legend_title="Model")
            st.plotly_chart(fig, use_container_width=True)

            st.info(f"📊 Random Forest F1 ranges from {min(rf_folds):.3f} to {max(rf_folds):.3f} across folds (std={pd.Series(rf_folds).std():.3f}). Low variance confirms model stability.")
            st.markdown("""
**What we see:** All models show fold-to-fold variation — expected with only 249 stroke cases.

**Interesting finding:** Random Forest is the most consistent performer across all 5 folds.

**Implication:** Low standard deviation confirms the model is stable — not just lucky on one split.
            """)
    except FileNotFoundError:
        st.warning("Run train.py first to generate results.json")

    # ── 5. Model Comparison ───────────────────────────────────────────────────
    st.subheader("5. Model Comparison (Cross-Validation Results)")
    results_data = {
        "Model": ["DummyClassifier (Baseline 1)", "Logistic Regression (Baseline 2)",
                  "Random Forest ⭐", "XGBoost (Optuna)"],
        "F1": [0.000, 0.258, 0.293, 0.281],
        "Recall": [0.000, 0.503, 0.493, 0.453],
        "ROC-AUC": [0.500, 0.853, 0.848, 0.841],
        "Threshold": ["-", "0.115", "0.131", "0.129"],
        "Role": ["Naive baseline", "ML baseline", "Best model", "Optimized alternative"]
    }
    st.dataframe(pd.DataFrame(results_data), use_container_width=True)
    st.info("⭐ Random Forest wins on F1 (0.293). Logistic Regression has the highest Recall (0.503) — if maximizing stroke detection is the only goal, LogReg is competitive.")
    st.markdown("""
**What we see:** DummyClassifier F1=0.000 — confirms our models learn something real. All thresholds far below 0.5 (0.11–0.13).

**Interesting finding:** Default 0.5 threshold is completely inappropriate for this imbalanced problem.

**Limitation:** F1 scores ~0.29 reflect the genuine difficulty of predicting stroke from only 249 positive cases.
    """)

    # ── 6. Optuna HPO ─────────────────────────────────────────────────────────
    st.subheader("6. Optuna Hyperparameter Optimization (XGBoost)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**What is Optuna?**
Automatic hyperparameter optimization using Bayesian search — 30 trials × 5-fold CV = 150 model evaluations.

**What we optimized:** n_estimators, max_depth, learning_rate, subsample, colsample_bytree
        """)
    with col2:
        optuna_data = {
            "Parameter": ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"],
            "Default": [200, 4, 0.1, 1.0, 1.0],
            "Optuna Best": [103, 4, 0.029, 0.986, 0.880],
        }
        st.dataframe(pd.DataFrame(optuna_data), use_container_width=True)
        st.caption("Lower learning rate with fewer trees — more careful learning.")
    st.info("📊 XGBoost F1 improved from 0.204 (default) to 0.281 (Optuna) — 38% improvement. Yet Random Forest still wins, suggesting RF is a better fit for this small dataset.")

    # ── 7. Best Model Test Evaluation ────────────────────────────────────────
    st.subheader("7. Best Model — Test Set Evaluation")
    col1, col2, col3 = st.columns(3)
    col1.metric("Best Model", "Random Forest")
    col2.metric("CV F1 Score", "0.293")
    col3.metric("Test F1 Score", "0.290")
    st.success("✅ CV score (0.293) ≈ Test score (0.290) — No overfitting detected. The model generalizes well.")

    st.markdown("---")
    st.markdown("**Per-Class Performance:**")
    per_class_data = {
        "Class": ["No Stroke (0)", "Stroke (1)"],
        "Precision": [0.97, 0.20],
        "Recall": [0.89, 0.54],
        "F1-Score": [0.93, 0.29],
        "Support": [972, 50]
    }
    st.dataframe(pd.DataFrame(per_class_data), use_container_width=True)
    st.markdown("""
**What this means:**
- No Stroke → excellent performance (F1=0.93) ✅
- Stroke → modest performance (F1=0.29) ⚠️ — driven by severe imbalance
- High Recall (0.54) prioritized over Precision (0.20) — clinically motivated
    """)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Confusion Matrix**")
        cm_data = np.array([[864, 108], [23, 27]])
        fig = px.imshow(cm_data,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=["No Stroke", "Stroke"], y=["No Stroke", "Stroke"],
                        color_continuous_scale="Blues", text_auto=True,
                        title="Confusion Matrix (Test Set)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
- **864** → Correctly identified no stroke ✅
- **27** → Correctly identified stroke ✅
- **23** → Missed strokes ❌ (False Negatives)
- **108** → False alarms ⚠️ (False Positives)
        """)

    with col2:
        st.markdown("**ROC Curve**")
        fpr, tpr, _ = roc_curve(y_test_split, y_prob)
        auc_score = roc_auc_score(y_test_split, y_prob)
        fig = px.line(x=fpr, y=tpr,
                      title=f"ROC Curve (AUC = {auc_score:.3f})",
                      labels={"x": "False Positive Rate", "y": "True Positive Rate"})
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines",
                        line=dict(dash="dash", color="gray"), name="Random classifier")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
- **AUC = 0.821** → Model ranks stroke patients above non-stroke 82% of the time
- Significantly better than random (AUC=0.5) ✅
        """)

    with col3:
        st.markdown("**Precision-Recall Curve**")
        precision_vals, recall_vals, _ = precision_recall_curve(y_test_split, y_prob)
        avg_precision = average_precision_score(y_test_split, y_prob)
        fig = px.line(x=recall_vals, y=precision_vals,
                      title=f"PR Curve (AP = {avg_precision:.3f})",
                      labels={"x": "Recall", "y": "Precision"})
        fig.add_hline(y=0.049, line_dash="dash", line_color="gray",
                      annotation_text="Random (4.9%)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
- **AP = Average Precision** → area under PR curve
- More informative than ROC for imbalanced datasets ✅
- Dashed line → random classifier baseline (4.9%)
        """)

    # ── 8. Threshold vs Metrics Chart ────────────────────────────────────────
    st.subheader("8. Threshold vs Performance")
    thresholds_range = np.linspace(0.05, 0.35, 30)
    threshold_results = []
    for t in thresholds_range:
        y_pred_t = (y_prob >= t).astype(int)
        threshold_results.append({
            "Threshold": round(t, 3),
            "Recall": recall_score(y_test_split, y_pred_t),
            "Precision": precision_score(y_test_split, y_pred_t, zero_division=0),
            "F1": f1_score(y_test_split, y_pred_t)
        })
    thresh_df = pd.DataFrame(threshold_results)
    fig = px.line(thresh_df, x="Threshold", y=["Recall", "Precision", "F1"],
                  title="Threshold vs Performance Metrics",
                  labels={"value": "Score", "variable": "Metric"})
    fig.add_vline(x=0.131, line_dash="dash", line_color="red",
                  annotation_text="Selected (0.131)")
    st.plotly_chart(fig, use_container_width=True)
    st.info("🎯 The red dashed line shows the threshold selected by TunedThresholdClassifierCV (0.131) — the point that maximizes F1.")
    st.markdown("""
**What we see:** As threshold decreases, Recall increases but Precision drops. F1 peaks around 0.10–0.13.

**Interesting finding:** The optimal threshold (0.131) is far from the default 0.5 — confirming that imbalanced data requires threshold tuning.

**Clinical trade-off:** Lowering threshold catches more strokes but creates more false alarms. The right threshold depends on clinical context.
    """)

    # ── 9. Interactive Threshold Analysis ────────────────────────────────────
    st.subheader("9. Interactive Threshold Analysis")
    st.markdown("Adjust the slider to see how different thresholds affect performance in real time:")

    threshold = st.slider("Decision Threshold", min_value=0.05, max_value=0.35,
                          value=0.131, step=0.005,
                          help="Lower threshold → more stroke predictions → higher Recall, lower Precision")

    y_pred_thresh = (y_prob >= threshold).astype(int)
    recall = recall_score(y_test_split, y_pred_thresh)
    precision = precision_score(y_test_split, y_pred_thresh, zero_division=0)
    f1 = f1_score(y_test_split, y_pred_thresh)
    cm_thresh = confusion_matrix(y_test_split, y_pred_thresh)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Threshold", f"{threshold:.3f}")
    col2.metric("Recall", f"{recall:.3f}", delta=f"{recall - 0.54:.3f}")
    col3.metric("Precision", f"{precision:.3f}")
    col4.metric("F1 Score", f"{f1:.3f}")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.imshow(cm_thresh,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=["No Stroke", "Stroke"], y=["No Stroke", "Stroke"],
                        color_continuous_scale="Blues", text_auto=True,
                        title=f"Confusion Matrix (threshold={threshold:.3f})")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown(f"""
**At threshold = {threshold:.3f}:**
- Stroke cases caught: **{cm_thresh[1][1]}** out of 50
- Stroke cases missed: **{cm_thresh[1][0]}** (False Negatives)
- False alarms: **{cm_thresh[0][1]}** (False Positives)

**Clinical interpretation:**
- Lower threshold → catch more strokes but more false alarms
- Higher threshold → fewer false alarms but miss more strokes
- **For medical use: lower threshold is safer** — missing a stroke is life-threatening
        """)

    # ── 10. Feature Importance ────────────────────────────────────────────────
    st.subheader("10. Feature Importance (Random Forest)")
    try:
        rf_estimator = model.named_steps["model"].estimator_
        feature_names_out = model.named_steps["preprocessor"].named_steps["column_tf"].get_feature_names_out()
        clean_names = [name.split("__")[-1] for name in feature_names_out]
        importances = rf_estimator.feature_importances_

        fi_df = pd.DataFrame({
            "Feature": clean_names,
            "Importance": importances,
        }).sort_values("Importance", ascending=False).head(10)

        fi_df["Type"] = fi_df["Feature"].apply(
            lambda x: "Engineered" if any(k in x for k in ["age_group", "glucose_category", "bmi_missing"])
            else "Binary" if any(k in x for k in ["hypertension", "heart_disease", "ever_married", "Residence_type", "gender"])
            else "Numeric"
        )
    except Exception as e:
        st.warning(f"Could not load feature importance: {e}")
        fi_df = pd.DataFrame()
    fig = px.bar(fi_df, x="Importance", y="Feature", color="Type", orientation="h",
                 title="Top 10 Feature Importances",
                 labels={"Importance": "Importance Score", "Feature": ""})
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)
    st.info("🏆 Age is the most important feature (0.202), followed by glucose (0.178). bmi_missing ranks 4th (0.088) — confirming the MNAR pattern detected in EDA.")
    st.markdown("""
**What we see:** Age and glucose dominate. Three engineered features appear in top 10.

**Interesting finding:** `bmi_missing` (rank 4, 0.088) — the fact that BMI is missing carries more predictive signal than many clinical variables. This validates our MNAR analysis.

**Interesting finding:** `age_group_senior` and `age_group_young` both appear — non-linear age grouping adds information beyond raw age.

**Limitation:** `Residence_type` ranks last — rural/urban distinction has minimal predictive value for stroke.
    """)

   
    # ── 11. Error Analysis ───────────────────────────────────────────────────
    st.subheader("11. Error Analysis — Who Does the Model Miss?")
    st.markdown("Understanding **which patients** the model fails on is as important as overall metrics:")

    df_test = load_data()
    X_all = df_test.drop(columns=[TARGET])
    y_all = df_test[TARGET]
    _, X_test_ea, _, y_test_ea = train_test_split(
        X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all
    )

    y_pred_ea = model.predict(X_test_ea)
    X_test_ea = X_test_ea.copy()
    X_test_ea["actual"] = y_test_ea.values
    X_test_ea["predicted"] = y_pred_ea

    fn = X_test_ea[(X_test_ea["actual"] == 1) & (X_test_ea["predicted"] == 0)]
    tp = X_test_ea[(X_test_ea["actual"] == 1) & (X_test_ea["predicted"] == 1)]

    col1, col2 = st.columns(2)
    with col1:
        compare_df = pd.DataFrame({
            "Age": list(fn["age"]) + list(tp["age"]),
            "Group": ["Missed Stroke"] * len(fn) + ["Caught Stroke"] * len(tp)
        })
        fig = px.histogram(compare_df, x="Age", color="Group",
                           barmode="overlay", opacity=0.7,
                           title="Age: Missed vs Caught Strokes")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        compare_df2 = pd.DataFrame({
            "Glucose": list(fn["avg_glucose_level"]) + list(tp["avg_glucose_level"]),
            "Group": ["Missed Stroke"] * len(fn) + ["Caught Stroke"] * len(tp)
        })
        fig = px.histogram(compare_df2, x="Glucose", color="Group",
                           barmode="overlay", opacity=0.7,
                           title="Glucose: Missed vs Caught Strokes")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Missed Strokes (FN)", len(fn))
    col2.metric("Avg Age — Missed", f"{fn['age'].mean():.1f}")
    col3.metric("Avg Age — Caught", f"{tp['age'].mean():.1f}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Caught Strokes (TP)", len(tp))
    col2.metric("Avg Glucose — Missed", f"{fn['avg_glucose_level'].mean():.1f}")
    col3.metric("Avg Glucose — Caught", f"{tp['avg_glucose_level'].mean():.1f}")

    st.info("🔍 Missed strokes are significantly younger (avg 58.2 vs 76.3) "
            "and have lower glucose (avg 100.1 vs 168.7). "
            "The model struggles with atypical stroke presentations.")
    st.markdown("""
**What we see:** 
- Missed strokes → avg age 58.2, avg glucose 100.1 (normal range!)
- Caught strokes → avg age 76.3, avg glucose 168.7 (diabetic range)

**Interesting finding:** The model has learned to associate stroke with old age + high glucose. 
It misses younger patients with normal glucose — atypical but real stroke cases.

**Clinical implication:** Extra vigilance needed for patients aged 40-65 with normal glucose 
levels — the model's blind spot. Clinical judgment should not rely solely on this model 
for this patient group.
    """)


    
    # ── 11. Model Limitations ─────────────────────────────────────────────────
    st.subheader("11. Model Limitations & Honest Assessment")
    st.warning("""
⚠️ **What our model does well:**
- ROC-AUC of 0.821 — significantly better than random (0.5)
- Catches 54% of stroke cases at default threshold
- No overfitting (CV ≈ Test score)
- Properly handles class imbalance via threshold optimization
    """)
    st.error("""
❌ **Where our model struggles:**
- F1 score of 0.29 is modest — driven by severe class imbalance (only 249 stroke cases)
- Still misses 46% of stroke cases at optimal threshold
- Precision of 0.20 means 80% of "stroke" predictions are false alarms
- Performance on very young patients (<20) is unreliable due to very few cases
    """)
    st.markdown("""
**What could improve the model:**
- More data — especially more stroke cases
- Additional clinical features (blood pressure, cholesterol)
- Ensemble of models
- Class-weighted training (class_weight='balanced')
    """)

