import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go

st.set_page_config(page_title = "SensorGuard", page_icon = "🛠️", layout = "wide")

ROOT = Path(__file__).resolve().parent
if ROOT.name == "src":
    ROOT = ROOT.parent
MODEL_PATH = ROOT / "model" / "sensorguard.joblib"
DATA_PATH = ROOT / "data" / "sensor.csv"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family:'IBM Plex Mono', monospace; }
h1, h2, h3 { font-family:'Chakra Petch', sans-serif; letter-spacing:1px; }
[data-testid="stMetricValue"] { font-family:'Chakra Petch', sans-serif; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

b = load_model()
model, scaler, FEATS, SAVED_THR, NAME = (
    b["model"], b["scaler"], b["feature_names"], b["threshold"], b["name"]
)

def score_one(values):
    X = scaler.transform(np.asarray(values, dtype="float32").reshape(1, -1))
    return float(-model.decision_function(X)[0])

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None, None, None, None
    raw = pd.read_csv(DATA_PATH)
    status = raw["machine_status"].values
    clean = raw.reindex(columns=FEATS).ffill().bfill().fillna(0).astype("float32")
    s = -model.decision_function(scaler.transform(clean.values))
    return clean, status, float(s.min()), float(s.max())

clean, status, SMIN, SMAX = load_data()

# ---------- helpers ----------
def make_gauge(score, thr):
    color = "#ff4d56" if score >= thr else "#34e5a3"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"valueformat": ".4f"},
        gauge={
            "axis": {"range": [SMIN, SMAX]},
            "bar": {"color": color, "thickness": 0.3},
            "threshold": {"line": {"color": "#ffffff", "width": 3},
                          "thickness": 0.9, "value": thr},
            "steps": [{"range": [SMIN, thr], "color": "#10331f"},
                      {"range": [thr, SMAX], "color": "#3a1418"}],
        }))
    fig.update_layout(height = 240, margin = dict(l = 20, r = 20, t = 10, b = 10),
                      paper_bgcolor="rgba(0,0,0,0)", font_color="#cdd7e3")
    return fig

def top_abnormal(values, n=6):
    z = (np.asarray(values) - scaler.mean_) / scaler.scale_   # how many std-devs off normal
    return pd.Series(np.abs(z), index=FEATS).sort_values(ascending=False).head(n)

# ---------- header ----------
st.title("🛠️ SensorGuard")
st.caption(f"Predictive-maintenance anomaly detector · model: {NAME} · {len(FEATS)} sensors")

# ---------- sidebar ----------
with st.sidebar:
    st.header("⚙️ Controls")
    st.metric("Model", NAME)
    st.metric("Sensors watched", len(FEATS))
    thr = st.slider("Alarm sensitivity", float(SMIN), float(SMAX), float(SAVED_THR)) \
        if SMIN is not None else SAVED_THR
    st.caption(f"Saved threshold: {SAVED_THR:.4f}")
    st.divider()
    st.caption("Trained only on healthy readings. Anything unfamiliar gets flagged.")

# ---------- session memory ----------
st.session_state.setdefault("history", [])
st.session_state.setdefault("row", None)

tab1, tab2, tab3 = st.tabs(["🔍 Live check", "📁 Batch score", "📊 Model"])

# ===== TAB 1 =====
with tab1:
    if clean is None:
        st.warning("No data/sensor.csv found. Use the Batch tab instead.")
    else:
        c1, c2, c3 = st.columns(3)
        if c1.button("🎲 Random reading", use_container_width=True):
            st.session_state.row = int(np.random.randint(len(clean)))
        if c2.button("⚠️ Known failure", use_container_width=True):
            st.session_state.row = int(np.random.choice(np.where(status != "NORMAL")[0]))
        if c3.button("✅ Healthy reading", use_container_width=True):
            st.session_state.row = int(np.random.choice(np.where(status == "NORMAL")[0]))

        if st.session_state.row is not None:
            i = st.session_state.row
            vals = clean.iloc[i].values
            score = score_one(vals)
            st.session_state.history = (st.session_state.history + [score])[-25:]
            anomaly = score >= thr

            left, right = st.columns([1.1, 1])
            with left:
                st.plotly_chart(make_gauge(score, thr), use_container_width = True)
            with right:
                st.metric("Verdict", "🔴 ANOMALY" if anomaly else "🟢 NORMAL")
                st.metric("Anomaly score", f"{score:.4f}")
                st.metric("Margin to alarm", f"{score - thr:+.4f}")
                st.write(f"Actual label in data: **{status[i]}**")

            st.markdown("**Sensors that look most unusual right now**")
            st.bar_chart(top_abnormal(vals))

            if len(st.session_state.history) > 1:
                st.markdown("**Your recent checks (score over time)**")
                st.line_chart(pd.DataFrame({"score": st.session_state.history}))

            with st.expander("See all sensor values"):
                st.dataframe(pd.Series(vals, index = FEATS, name = "reading"))

with tab2:
    up = st.file_uploader("Upload a CSV with the sensor columns", type = "csv")
    if up is not None:
        new = pd.read_csv(up)
        X = new.reindex(columns=FEATS).ffill().bfill().fillna(0).astype("float32")
        new["anomaly_score"] = -model.decision_function(scaler.transform(X.values))
        new["alert"] = (new["anomaly_score"] >= thr).astype(int)
        a, bb = st.columns(2)
        a.metric("Rows flagged", f"{int(new['alert'].sum())} / {len(new)}")
        bb.metric("Alert rate", f"{100 * new['alert'].mean():.1f}%")
        st.line_chart(new["anomaly_score"])
        st.dataframe(new.head(100))
        st.download_button("⬇️ Download scored CSV",
                           new.to_csv(index=False).encode(), "scored.csv", "text/csv")

with tab3:
    st.subheader("How this model was chosen")
    st.write(
        "I benchmarked eight models on the same data. The supervised ones look "
        "almost perfect, but that is because spotting a machine already in recovery "
        "is easy and is not really a prediction. The honest, deployable choice is an "
        "unsupervised detector that never sees a failure label, so SensorGuard runs "
        "on Isolation Forest."
    )
    bench = pd.DataFrame({
        "Model": ["Random Forest", "HistGradientBoosting", "KNN", "Decision Tree",
                  "Logistic Regression", "Isolation Forest ⭐", "Elliptic Envelope",
                  "Local Outlier Factor"],
        "Type": ["supervised"] * 5 + ["unsupervised"] * 3,
        "F1": [1.000, 1.000, 0.997, 0.989, 0.975, 0.681, 0.673, 0.576],
        "Recall": [1.000, 1.000, 0.997, 0.999, 0.999, 0.994, 1.000, 0.844],
    })
    st.dataframe(bench, use_container_width = True, hide_index = True)
    st.caption("Isolation Forest catches ~99% of failures in under 2 seconds, trained only on healthy data.")