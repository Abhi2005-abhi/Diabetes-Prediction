import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

# ==================================================================================
# PAGE CONFIG
# ==================================================================================
st.set_page_config(
    page_title="GlucoScope | Diabetes Risk Dashboard",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "Diabetes.csv"
MODEL_PATH = "models/logistic_reg.sav"

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

# Clinically-informed reference ranges used purely to give the user a friendly
# "where does this number sit" cue -- not a medical diagnosis.
REFERENCE_RANGES = {
    "Pregnancies": (0, 17, "count"),
    "Glucose": (70, 140, "mg/dL"),
    "BloodPressure": (60, 80, "mm Hg"),
    "SkinThickness": (10, 40, "mm"),
    "Insulin": (16, 166, "mu U/mL"),
    "BMI": (18.5, 24.9, "kg/m²"),
    "DiabetesPedigreeFunction": (0.08, 0.6, "score"),
    "Age": (18, 45, "years"),
}

# ==================================================================================
# THEME / CSS
# ==================================================================================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root{
        --bg:#F3F8F7;
        --surface:#FFFFFF;
        --ink:#132226;
        --muted:#5C7377;
        --line:#DCEAE8;
        --teal:#0B5563;
        --teal-light:#12818F;
        --teal-pale:#E4F3F1;
        --coral:#E8604C;
        --coral-pale:#FCE7E3;
        --good:#1E9E76;
        --good-pale:#E1F5EC;
    }

    html, body, [class*="css"]  { font-family:'Inter', sans-serif; }
    .stApp { background:var(--bg); }

    h1,h2,h3,h4 { font-family:'Sora', sans-serif !important; color:var(--ink); }

    #MainMenu, footer, header {visibility:hidden;}

    section[data-testid="stSidebar"]{
        background:linear-gradient(180deg, #0B5563 0%, #0A3F49 100%);
        border-right:none;
    }
    section[data-testid="stSidebar"] * { color:#EAF6F4 !important; }
    section[data-testid="stSidebar"] .stRadio label { font-family:'Inter'; }
    section[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,0.15); }

    /* Hero banner */
    .hero{
        background:linear-gradient(120deg, #0B5563 0%, #12818F 55%, #1BA6A0 100%);
        border-radius:20px;
        padding:34px 40px;
        color:white;
        margin-bottom:26px;
        box-shadow:0 10px 30px rgba(11,85,99,0.25);
        position:relative;
        overflow:hidden;
    }
    .hero::after{
        content:"";
        position:absolute; right:-60px; top:-60px;
        width:220px; height:220px; border-radius:50%;
        background:rgba(255,255,255,0.08);
    }
    .hero h1{ color:white !important; font-size:2.1rem; margin:0 0 6px 0; font-weight:800; }
    .hero p{ color:#DFF4F1; font-size:1rem; margin:0; max-width:640px; }
    .hero .badge{
        display:inline-block; background:rgba(255,255,255,0.16); border:1px solid rgba(255,255,255,0.35);
        padding:4px 12px; border-radius:999px; font-size:0.78rem; letter-spacing:.03em;
        margin-bottom:14px; font-family:'IBM Plex Mono', monospace;
    }

    /* Cards */
    .card{
        background:var(--surface);
        border:1px solid var(--line);
        border-radius:16px;
        padding:22px 24px;
        box-shadow:0 2px 10px rgba(20,60,60,0.04);
    }
    .stat-card{
        background:var(--surface); border:1px solid var(--line); border-radius:14px;
        padding:16px 18px; text-align:left;
    }
    .stat-card .label{ font-size:0.76rem; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; font-family:'IBM Plex Mono', monospace;}
    .stat-card .value{ font-size:1.6rem; font-weight:700; color:var(--teal); font-family:'Sora', sans-serif; }

    .chip{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:0.74rem; font-weight:600; font-family:'IBM Plex Mono', monospace;}
    .chip-low{ background:#EAF1FA; color:#3562A6; }
    .chip-normal{ background:var(--good-pale); color:var(--good); }
    .chip-high{ background:var(--coral-pale); color:var(--coral); }

    .verdict{
        border-radius:16px; padding:22px 26px; margin-top:6px;
    }
    .verdict-risk{ background:linear-gradient(120deg,#FCE7E3,#FBD9D3); border:1px solid #F3B7AB; }
    .verdict-ok{ background:linear-gradient(120deg,#E1F5EC,#D3F0E2); border:1px solid #A9E1C4; }
    .verdict h3{ margin-top:0; }

    div[data-testid="stMetric"]{
        background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:14px 16px;
    }

    .stButton>button{
        background:linear-gradient(120deg,#0B5563,#12818F);
        color:white; border:none; border-radius:12px; padding:0.6em 1.4em;
        font-weight:600; font-family:'Sora', sans-serif; letter-spacing:.01em;
        box-shadow:0 6px 16px rgba(11,85,99,0.25);
    }
    .stButton>button:hover{ filter:brightness(1.08); }

    .stTabs [data-baseweb="tab-list"]{ gap:6px; }
    .stTabs [data-baseweb="tab"]{
        background:var(--surface); border-radius:10px 10px 0 0; padding:10px 18px;
        font-family:'Sora', sans-serif; font-weight:600; color:var(--muted);
        border:1px solid var(--line); border-bottom:none;
    }
    .stTabs [aria-selected="true"]{ color:var(--teal) !important; background:var(--teal-pale) !important; }

    .footnote{ color:var(--muted); font-size:0.8rem; margin-top:24px; }
    </style>
    """, unsafe_allow_html=True)


# ==================================================================================
# MODEL
# ==================================================================================
@st.cache_resource(show_spinner="Loading prediction model…")
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    df = pd.read_csv(DATA_PATH)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    return pipeline


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def get_test_metrics(_model, df):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = _model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "cm": confusion_matrix(y_test, y_pred),
    }


def status_chip(feature, value):
    lo, hi, _ = REFERENCE_RANGES[feature]
    if value < lo:
        return '<span class="chip chip-low">below range</span>'
    elif value > hi:
        return '<span class="chip chip-high">above range</span>'
    return '<span class="chip chip-normal">in range</span>'


def make_gauge(proba):
    color = "#E8604C" if proba >= 0.5 else "#1E9E76"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={"suffix": "%", "font": {"size": 42, "family": "Sora", "color": "#132226"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#5C7377"},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "#E1F5EC"},
                {"range": [30, 60], "color": "#FFF3D6"},
                {"range": [60, 100], "color": "#FCE7E3"},
            ],
            "threshold": {"line": {"color": "#132226", "width": 3}, "thickness": 0.8, "value": proba * 100},
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=10, b=10),
                       paper_bgcolor="rgba(0,0,0,0)", font={"family": "Inter"})
    return fig


# ==================================================================================
# APP
# ==================================================================================
inject_css()
model = load_or_train_model()
df_data = load_dataset()

with st.sidebar:
    st.markdown("### 🩸 GlucoScope")
    st.caption("Diabetes Risk Intelligence")
    st.markdown("---")
    page = st.radio("Navigate", ["🔍 Single Prediction", "📂 Batch Prediction", "📊 Model Insights"],
                     label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Model**")
    st.caption("Logistic Regression · scikit-learn pipeline")
    st.markdown("**Trained on**")
    st.caption(f"{len(df_data)} patient records — PIMA Indian Diabetes Dataset")
    st.markdown("---")
    st.caption("⚠️ Educational tool only — not a medical diagnosis.")

st.markdown("""
<div class="hero">
    <div class="badge">🩺 CLINICAL ML DASHBOARD</div>
    <h1>GlucoScope — Diabetes Risk Predictor</h1>
    <p>Estimate diabetes risk from routine diagnostic measurements using a logistic
    regression model trained on the PIMA Indian Diabetes dataset. Enter a single
    patient's numbers, or score a whole batch in one upload.</p>
</div>
""", unsafe_allow_html=True)

page = page.split(" ", 1)[1]

# ----------------------------------------------------------------------------------
# SINGLE PREDICTION
# ----------------------------------------------------------------------------------
if page == "Single Prediction":
    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("#### Patient measurements")
        st.caption("Adjust the sliders to match the diagnostic report.")

        c1, c2 = st.columns(2)
        with c1:
            pregnancies = st.slider("Pregnancies", 0, 17, 1)
            glucose = st.slider("Glucose (mg/dL)", 0, 200, 120)
            blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 140, 70)
            skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
        with c2:
            insulin = st.slider("Insulin (mu U/mL)", 0, 850, 80)
            bmi = st.slider("BMI (kg/m²)", 0.0, 67.0, 25.0, step=0.1)
            dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, step=0.01)
            age = st.slider("Age (years)", 1, 100, 30)

        values = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
        input_data = pd.DataFrame([values], columns=FEATURES)

        predict_clicked = st.button("🔮 Run Prediction", type="primary", use_container_width=True)

        st.markdown("###### Where each value sits vs. a typical healthy range")
        chip_cols = st.columns(4)
        for i, feat in enumerate(FEATURES):
            with chip_cols[i % 4]:
                st.markdown(f"**{feat}**<br>{status_chip(feat, values[i])}", unsafe_allow_html=True)

    with right:
        st.markdown("#### Risk result")
        if predict_clicked:
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0][1]

            st.plotly_chart(make_gauge(proba), use_container_width=True, config={"displayModeBar": False})

            if prediction == 1:
                st.markdown(f"""
                <div class="verdict verdict-risk">
                    <h3>⚠️ Elevated diabetes risk</h3>
                    <p>The model estimates a <b>{proba:.0%}</b> probability of diabetes based on
                    the values entered. This flags the profile for closer medical review —
                    it is not a diagnosis.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict verdict-ok">
                    <h3>✅ Low diabetes risk</h3>
                    <p>The model estimates a <b>{proba:.0%}</b> probability of diabetes based on
                    the values entered. Keep up routine checkups and a healthy lifestyle.</p>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("What drove this prediction?"):
                classifier = model.named_steps["classifier"]
                scaler = model.named_steps["scaler"]
                scaled_input = scaler.transform(input_data)[0]
                contributions = scaled_input * classifier.coef_[0]
                contrib_df = pd.DataFrame({"Feature": FEATURES, "Contribution": contributions})
                contrib_df = contrib_df.sort_values("Contribution")
                fig = px.bar(contrib_df, x="Contribution", y="Feature", orientation="h",
                             color="Contribution", color_continuous_scale=["#1E9E76", "#EDEDED", "#E8604C"],
                             color_continuous_midpoint=0)
                fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   coloraxis_showscale=False, font={"family": "Inter"})
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.caption("Positive bars pushed the prediction toward *diabetic*, negative bars pulled it toward *not diabetic*.")
        else:
            st.info("Set the patient's values on the left and click **Run Prediction** to see the risk gauge here.")

# ----------------------------------------------------------------------------------
# BATCH PREDICTION
# ----------------------------------------------------------------------------------
elif page == "Batch Prediction":
    st.markdown("#### Score a batch of patients")
    st.caption("Upload a CSV with the same 8 feature columns as the training set (no `Outcome` column needed).")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    with st.expander("📄 Need a template?"):
        template = pd.DataFrame([values if 'values' in dir() else [1, 120, 70, 20, 80, 25.0, 0.5, 30]], columns=FEATURES)
        st.dataframe(template, use_container_width=True)
        st.download_button("⬇️ Download empty template", template.iloc[0:0].to_csv(index=False).encode("utf-8"),
                            "template.csv", "text/csv")

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.markdown("###### Preview")
        st.dataframe(df_upload.head(), use_container_width=True)

        if all(col in df_upload.columns for col in FEATURES):
            predictions = model.predict(df_upload[FEATURES])
            probabilities = model.predict_proba(df_upload[FEATURES])[:, 1]

            df_upload["Prediction"] = predictions
            df_upload["Probability (Diabetic)"] = probabilities.round(3)
            df_upload["Result"] = df_upload["Prediction"].apply(lambda x: "Diabetic" if x == 1 else "Not Diabetic")

            total = len(df_upload)
            diabetic_count = int(df_upload["Prediction"].sum())

            m1, m2, m3 = st.columns(3)
            m1.markdown(f'<div class="stat-card"><div class="label">Total Patients</div><div class="value">{total}</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="stat-card"><div class="label">Flagged Diabetic</div><div class="value">{diabetic_count}</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="stat-card"><div class="label">Flagged Rate</div><div class="value">{diabetic_count/total*100:.1f}%</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns([1, 1.4])
            with c1:
                pie = px.pie(names=["Not Diabetic", "Diabetic"],
                             values=[total - diabetic_count, diabetic_count],
                             color=["Not Diabetic", "Diabetic"],
                             color_discrete_map={"Not Diabetic": "#1E9E76", "Diabetic": "#E8604C"},
                             hole=0.55)
                pie.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                                   paper_bgcolor="rgba(0,0,0,0)", font={"family": "Inter"},
                                   legend=dict(orientation="h", y=-0.1))
                st.plotly_chart(pie, use_container_width=True, config={"displayModeBar": False})
            with c2:
                hist = px.histogram(df_upload, x="Probability (Diabetic)", nbins=20,
                                     color_discrete_sequence=["#12818F"])
                hist.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    font={"family": "Inter"}, bargap=0.05,
                                    xaxis_title="Predicted probability", yaxis_title="Patients")
                st.plotly_chart(hist, use_container_width=True, config={"displayModeBar": False})

            st.markdown("###### Full results")
            st.dataframe(df_upload, use_container_width=True)

            csv_output = df_upload.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download predictions as CSV", csv_output, "predictions.csv", "text/csv",
                                type="primary")
        else:
            st.error(f"CSV must contain these columns: {', '.join(FEATURES)}")

# ----------------------------------------------------------------------------------
# MODEL INSIGHTS / ABOUT
# ----------------------------------------------------------------------------------
else:
    metrics = get_test_metrics(model, df_data)

    st.markdown("#### Model performance")
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip(
        [c1, c2, c3, c4],
        ["Accuracy", "Precision", "Recall", "F1 Score"],
        [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]],
    ):
        col.markdown(f'<div class="stat-card"><div class="label">{label}</div><div class="value">{val:.0%}</div></div>',
                      unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("###### Confusion matrix (test set)")
        cm = metrics["cm"]
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Teal",
                         labels=dict(x="Predicted", y="Actual", color="Count"),
                         x=["Not Diabetic", "Diabetic"], y=["Not Diabetic", "Diabetic"])
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                           paper_bgcolor="rgba(0,0,0,0)", font={"family": "Inter"}, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        st.markdown("###### What the model weighs most")
        classifier = model.named_steps["classifier"]
        coef_df = pd.DataFrame({"Feature": FEATURES, "Weight": classifier.coef_[0]}).sort_values("Weight")
        fig2 = px.bar(coef_df, x="Weight", y="Feature", orientation="h",
                      color="Weight", color_continuous_scale=["#1E9E76", "#EDEDED", "#E8604C"],
                      color_continuous_midpoint=0)
        fig2.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            coloraxis_showscale=False, font={"family": "Inter"})
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### About this project")
    a1, a2 = st.columns([1.2, 1])
    with a1:
        st.markdown("""
        <div class="card">
        GlucoScope uses a <b>Logistic Regression</b> model, wrapped in a scaling
        pipeline, to estimate diabetes likelihood from eight routine diagnostic
        measurements. It's trained on the <b>PIMA Indian Diabetes Dataset</b> from
        the National Institute of Diabetes and Digestive and Kidney Diseases —
        768 records of female patients aged 21 and up.
        <br><br>
        <b>How to use it</b><br>
        • <b>Single Prediction</b> — enter one patient's numbers and get an instant risk gauge.<br>
        • <b>Batch Prediction</b> — upload a CSV of many patients and download scored results.<br>
        • <b>Model Insights</b> — see how well the model performs and which features matter most.
        </div>
        """, unsafe_allow_html=True)
    with a2:
        st.markdown("###### Dataset snapshot")
        st.dataframe(df_data.describe().T[["mean", "min", "max"]].round(1), use_container_width=True)

    st.markdown('<p class="footnote">Built with Streamlit, scikit-learn &amp; Plotly · Educational demo, not a substitute for professional medical advice.</p>', unsafe_allow_html=True)
