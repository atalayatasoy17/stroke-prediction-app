import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
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

df = load_data()
model = load_model()


if page == "EDA":
    st.header("Exploratory Data Analysis")

    # ── Dataset Overview ──────────────────────────────────────────────────────
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

    # ── Target Distribution ───────────────────────────────────────────────────
    st.subheader("Target Distribution")
    df_pie = df.copy()
    df_pie[TARGET] = df_pie[TARGET].map({0: "No Stroke", 1: "Stroke"})
    fig = px.pie(
        df_pie,
        names=TARGET,
        title="Stroke Distribution",
        color_discrete_sequence=["#4A90D9", "#E8E8E8"]
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("⚠️ Only 4.9% of patients had a stroke — a severe class imbalance. A naive model always predicting 'no stroke' would achieve 95% accuracy but miss every real case.")
    st.markdown("""
**What we see:** 95.1% no stroke vs 4.9% stroke — extreme imbalance.

**Interesting finding:** A model that never predicts stroke would score 95% accuracy — yet be completely useless clinically.

**Implication:** Accuracy is a misleading metric here. We use **F1, Recall, and ROC-AUC** instead.
""")

    # ── Age Distribution ──────────────────────────────────────────────────────
    st.subheader("Age vs Stroke")
    fig = px.histogram(
        df,
        x="age",
        color=TARGET,
        barmode="overlay",
        opacity=0.7,
        title="Age Distribution by Stroke",
        labels={"age": "Age", "count": "Count", TARGET: "Stroke"}
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("📊 Stroke cases are almost exclusively seen in patients over 40. Age is the strongest predictor in our dataset — confirmed later by feature importance (0.20).")
    st.markdown("""
**What we see:** Stroke cases (dark blue) concentrate heavily after age 40 and peak around 70-80.

**Interesting finding:** Almost zero stroke cases below age 40 — a near-perfect natural threshold.

**Open question:** Is the risk increase gradual or does it jump at certain age groups?
""")

    # ── Stroke Rate by Age ────────────────────────────────────────────────────
    st.subheader("Stroke Rate by Age")
    age_stroke = df.groupby("age")[TARGET].mean().reset_index()
    fig = px.line(
        age_stroke,
        x="age",
        y=TARGET,
        title="Stroke Rate by Age",
        labels={"age": "Age", TARGET: "Stroke Rate"}
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("📈 Stroke risk increases dramatically after age 60, reaching 25% at age 80. This non-linear pattern confirms that raw age alone may not fully capture the risk.")
    st.markdown("""
**What we see:** Risk stays near zero until age 40, then rises sharply after 60, reaching ~25% at age 80.

**Interesting finding:** The relationship is non-linear — risk accelerates, it doesn't grow steadily.

**Implication:** Grouping age into Young (0-40) / Middle (40-60) / Senior (60+) captures this pattern better than treating age as a continuous number.
""")

    # ── Numerical Variables ───────────────────────────────────────────────────
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

    # ── Categorical Variables ─────────────────────────────────────────────────
    st.subheader("Categorical Variables vs Stroke")
    cat_cols = ["gender", "ever_married", "work_type", "smoking_status"]
    for i in range(0, len(cat_cols), 2):
        col1, col2 = st.columns(2)
        with col1:
            col = cat_cols[i]
            stroke_rate = df.groupby(col)[TARGET].mean().reset_index()
            fig = px.bar(stroke_rate, x=col, y=TARGET,
                         title=f"Stroke Rate by {col}",
                         labels={TARGET: "Stroke Rate"})
            st.plotly_chart(fig, use_container_width=True)
        if i + 1 < len(cat_cols):
            with col2:
                col = cat_cols[i + 1]
                stroke_rate = df.groupby(col)[TARGET].mean().reset_index()
                fig = px.bar(stroke_rate, x=col, y=TARGET,
                             title=f"Stroke Rate by {col}",
                             labels={TARGET: "Stroke Rate"})
                st.plotly_chart(fig, use_container_width=True)
    st.info("📊 Gender shows minimal difference. Ever-married appears higher but is likely a confounding age effect. Work type and smoking status show more meaningful variation.")
    st.markdown("""
**What we see:** Female 4.7% vs Male 5.1% — nearly identical. Ever-married 6.5% vs 1.6% — large gap. Self-employed has highest work-type rate (8%).

**Interesting finding:** Ever-married stroke rate is 4x higher — but older people are both more likely to be married and to have strokes. This is a **confounding effect**, not a causal relationship.

**Open question:** If we control for age, does ever-married still predict stroke?
""")

    # ── Risk Factors ──────────────────────────────────────────────────────────
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

    # ── Compounding Effect ────────────────────────────────────────────────────
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

**Interesting finding:** The combined effect (20.3%) exceeds what simple addition would predict (13% + 17% - 4% = 26% minus overlap). This suggests a **compounding interaction**, not just additive risk.

**Implication:** Tree-based models (Random Forest, XGBoost) can naturally capture this interaction — they don't need manual feature engineering for this.
""")

    # ── BMI Missing Value Analysis ────────────────────────────────────────────
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

**Interesting finding:** The missingness itself is informative — it correlates strongly with age and stroke risk. This is called **Missing Not At Random (MNAR)**.

**Implication:** We cannot use simple median imputation — it would ignore the age pattern. Age-based median imputation is more appropriate. Also, a `bmi_missing` flag was added as a feature since the missingness carries predictive signal.
""")

    # ── Glucose Distribution ──────────────────────────────────────────────────
    st.subheader("Glucose Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="avg_glucose_level", color=TARGET,
                           barmode="overlay", opacity=0.7,
                           title="Glucose Distribution by Stroke",
                           labels={"avg_glucose_level": "Avg Glucose Level", TARGET: "Stroke"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        df["glucose_cat"] = pd.cut(df["avg_glucose_level"],
                                   bins=[0, 100, 125, 500],
                                   labels=["Normal", "Pre-diabetic", "Diabetic"])
        stroke_rate = df.groupby("glucose_cat", observed=False)[TARGET].mean().reset_index()
        fig = px.bar(stroke_rate, x="glucose_cat", y=TARGET,
                     title="Stroke Rate by Glucose Category",
                     labels={TARGET: "Stroke Rate", "glucose_cat": "Glucose Category"})
        st.plotly_chart(fig, use_container_width=True)
    st.info("🩸 Glucose is right-skewed with a secondary peak at 200+ mg/dL. Diabetic patients (glucose > 125) show 3x higher stroke rate (10%) vs normal (3.5%).")
    st.markdown("""
**What we see:** Two peaks in the distribution — one around 80-100 mg/dL (normal), one around 200+ mg/dL (diabetic subgroup). Diabetic category has a dramatically higher stroke rate.

**Interesting finding:** Pre-diabetic and normal groups have nearly identical stroke rates (3.5% vs 3.8%) — the risk only jumps significantly at the diabetic threshold (>125 mg/dL).

**Note:** Thresholds (Normal < 100, Pre-diabetic 100–125, Diabetic > 125 mg/dL) follow **American Diabetes Association** clinical standards — not arbitrary cuts.
""")

    # ── Smoking Status ────────────────────────────────────────────────────────
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

**Interesting finding:** Reverse causality — patients who already had health problems likely quit smoking, rather than quitting causing strokes. This is a well-known epidemiological phenomenon.

**Open question:** The Unknown group (30% of data, lowest stroke rate 3%) raises an important question: are these systematically younger or healthier patients whose smoking history was simply not recorded? Future work could investigate whether Unknown status correlates with age or health metrics.
""")

    # ── BMI Outlier Analysis ──────────────────────────────────────────────────
    st.subheader("BMI Outlier Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(df, y="bmi", title="BMI Distribution (with outliers)",
                     labels={"bmi": "BMI"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("Patients with BMI > 60", int((df["bmi"] > 60).sum()))
        st.metric("Max BMI", float(df["bmi"].max()))
        st.markdown("""
**What we see:** 13 patients have BMI > 60, with a maximum of 97.6.

**Interesting finding:** These extreme values appear predominantly in the non-stroke group — suggesting extreme BMI alone may not be the decisive risk factor.

**Why we kept them:**
- Medically possible (severe obesity exists clinically)
- Only 13 patients — removing them loses real data
- Tree-based models (Random Forest, XGBoost) are robust to outliers by design
""")

    # ── Correlation Matrix ────────────────────────────────────────────────────
    st.subheader("Correlation Matrix")
    corr_cols = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease", TARGET]
    corr_matrix = df[corr_cols].corr()
    fig = px.imshow(corr_matrix, title="Correlation Matrix",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=".2f")
    st.plotly_chart(fig, use_container_width=True)
    st.info("🔗 Age has the highest correlation with stroke (0.25). No feature pair exceeds 0.33 — no multicollinearity concern.")
    st.markdown("""
**What we see:** Age → stroke: 0.25 (highest). BMI → stroke: 0.04 (lowest). No pair of features exceeds 0.33 correlation.

**Interesting finding:** Despite age being the strongest individual predictor, its correlation (0.25) is still relatively modest — stroke is a multi-factorial condition.

**Implication:** No multicollinearity issues — all features can safely enter the model together without redundancy concerns.
""")