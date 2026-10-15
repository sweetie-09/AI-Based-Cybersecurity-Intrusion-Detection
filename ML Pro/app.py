import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os
import time
import subprocess
import sys
import psutil
import pickle
import random
import numpy as np
from datetime import datetime
from io import BytesIO

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Cybersecurity IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="auto"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIVE_RESULTS = os.path.join(BASE_DIR, "results", "live_results.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "multiclass_model.pkl")
DEMO_SAMPLES_PATH = os.path.join(BASE_DIR, "models", "demo_samples.pkl")

# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>
html, body, [class*="css"] { font-family: Arial, sans-serif; }
.stApp { background: #080d18; color: #e8f0ff; }
#MainMenu { visibility: hidden; }
header { visibility: hidden; }
footer { visibility: hidden; }
.block-container { padding-top: 20px; padding-bottom: 40px; max-width: 1600px; }

.header {
    background: #0d1628; border: 1px solid #18304a;
    border-radius: 16px; padding: 20px; margin-bottom: 20px;
}
.logo { font-size: 38px; }
.main-title { color: #f1f7ff; font-size: 23px; font-weight: 800; }
.sub-title { color: #70849e; font-size: 11px; letter-spacing: 1px; margin-top: 5px; }

.online { color: #22c55e; font-weight: bold; }
.offline { color: #ef4444; font-weight: bold; }
.demo-badge { color: #f59e0b; font-weight: bold; }

.card {
    background: #0d1628; border: 1px solid #1b3048;
    border-radius: 15px; padding: 20px; min-height: 125px;
}
.card-title { color: #71839d; font-size: 12px; font-weight: 600; }
.card-value { color: #f5f9ff; font-size: 31px; font-weight: 800; margin-top: 12px; }
.card-icon { font-size: 22px; }

.panel {
    background: #0d1628; border: 1px solid #1b3048;
    border-radius: 15px; padding: 20px; margin-top: 20px;
}
.panel-title { color: #f1f7ff; font-size: 17px; font-weight: 800; margin-bottom: 15px; }

.pred-normal { border: 1px solid #22c55e; }
.pred-threat { border: 1px solid #ef4444; }
.prediction { font-size: 28px; font-weight: 800; margin-top: 5px; }
.green { color: #22c55e; }
.red { color: #ef4444; }
.info-label { color: #71839d; font-size: 11px; }
.info-value { color: #e8f0ff; font-size: 13px; font-weight: 600; }

.pipeline { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.stage {
    flex: 1; text-align: center; background: #111d31;
    border: 1px solid #203750; border-radius: 12px; padding: 15px 5px;
}
.stage-icon { font-size: 22px; }
.stage-text { color: #9eb0c8; font-size: 11px; margin-top: 7px; }
.arrow { color: #00e5ff; font-size: 22px; }

.alert {
    background: #24151b; border-left: 4px solid #ef4444;
    border-radius: 8px; padding: 12px; margin-bottom: 8px;
}
.alert-title { color: #ff6262; font-weight: 800; }
.alert-info { color: #8fa2ba; font-size: 12px; }

.hero {
    background: linear-gradient(135deg, #0d1628 0%, #101e35 100%);
    border: 1px solid #18304a; border-radius: 18px;
    padding: 45px 40px; margin-bottom: 25px;
}
.hero-title { color: #f1f7ff; font-size: 34px; font-weight: 800; margin-bottom: 10px; }
.hero-sub { color: #9eb0c8; font-size: 16px; line-height: 1.6; }
.feature-card {
    background: #0d1628; border: 1px solid #1b3048;
    border-radius: 14px; padding: 22px; height: 100%;
}
.feature-icon { font-size: 30px; margin-bottom: 10px; }
.feature-title { color: #f1f7ff; font-size: 16px; font-weight: 700; margin-bottom: 8px; }
.feature-text { color: #9eb0c8; font-size: 13px; line-height: 1.6; }

div[data-testid="stButton"] button {
    background: #101d31; color: #dcecff;
    border: 1px solid #23415d; border-radius: 10px; font-weight: 700;
}
div[data-testid="stButton"] button:hover { border-color: #00e5ff; color: #00e5ff; }

div[data-testid="stDownloadButton"] button {
    background: #101d31 !important; color: #00e5ff !important;
    border: 1px solid #00e5ff !important;
    border-radius: 10px !important; font-weight: 700 !important;
}
div[data-testid="stDownloadButton"] button:hover {
    background: #00e5ff !important; color: #080d18 !important;
}
</style>
""",
    unsafe_allow_html=True
)
# Force sidebar to always be visible
st.markdown(
    """
<style>
/* Force sidebar open */
section[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    transform: none !important;
    min-width: 250px !important;
    width: 250px !important;
    margin-left: 0 !important;
}

/* Hide the collapse button so nobody hides it accidentally */
button[data-testid="stSidebarCollapseButton"],
button[kind="header"] {
    display: none !important;
}

/* Hide Streamlit header (menu bar) */
header[data-testid="stHeader"] {
    display: none !important;
}

/* Remove the collapsed-sidebar padding on mobile/tablet */
[data-testid="stSidebarNav"] {
    display: block !important;
}
</style>
""",
    unsafe_allow_html=True
)

# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"
if "mode" not in st.session_state:
    st.session_state.mode = "Live Monitoring"
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False
if "monitor_process" not in st.session_state:
    st.session_state.monitor_process = None
if "previous_prediction" not in st.session_state:
    st.session_state.previous_prediction = None
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "activity" not in st.session_state:
    st.session_state.activity = []
if "traffic_history" not in st.session_state:
    st.session_state.traffic_history = []
if "start_time" not in st.session_state:
    st.session_state.start_time = None

# Demo mode state
if "demo_running" not in st.session_state:
    st.session_state.demo_running = False
if "demo_alerts" not in st.session_state:
    st.session_state.demo_alerts = []
if "demo_activity" not in st.session_state:
    st.session_state.demo_activity = []
if "demo_traffic_history" not in st.session_state:
    st.session_state.demo_traffic_history = []
if "demo_packets" not in st.session_state:
    st.session_state.demo_packets = 0
if "demo_flows" not in st.session_state:
    st.session_state.demo_flows = 0
if "demo_normal" not in st.session_state:
    st.session_state.demo_normal = 0
if "demo_threats" not in st.session_state:
    st.session_state.demo_threats = 0
if "demo_start_time" not in st.session_state:
    st.session_state.demo_start_time = None
if "demo_last_pred" not in st.session_state:
    st.session_state.demo_last_pred = None
if "demo_last_conf" not in st.session_state:
    st.session_state.demo_last_conf = 0
if "demo_last_src" not in st.session_state:
    st.session_state.demo_last_src = "-"
if "demo_last_dst" not in st.session_state:
    st.session_state.demo_last_dst = "-"
if "demo_last_proto" not in st.session_state:
    st.session_state.demo_last_proto = "-"
if "demo_last_time" not in st.session_state:
    st.session_state.demo_last_time = "-"
if "demo_samples" not in st.session_state:
    st.session_state.demo_samples = None
if "ml_model" not in st.session_state:
    st.session_state.ml_model = None

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def read_results():
    if not os.path.exists(LIVE_RESULTS):
        return None
    try:
        with open(LIVE_RESULTS, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def start_monitor():
    if st.session_state.monitor_process is not None:
        return
    script = os.path.join(BASE_DIR, "live_monitor.py")
    if not os.path.exists(script):
        st.error("live_monitor.py was not found.")
        return
    try:
        process = subprocess.Popen(
            [sys.executable, script],
            cwd=BASE_DIR
        )
        st.session_state.monitor_process = process
        st.session_state.monitoring = True
        st.session_state.start_time = time.time()
        st.session_state.alerts = []
        st.session_state.activity = []
        st.session_state.traffic_history = []
        st.session_state.previous_prediction = None
    except Exception as error:
        st.error(f"Could not start monitor: {error}")


def stop_monitor():
    process = st.session_state.monitor_process
    if process is not None:
        try:
            process.terminate()
        except Exception:
            pass
    st.session_state.monitor_process = None
    st.session_state.monitoring = False


def get_attack_category(name):
    if not name:
        return "Normal"
    name = str(name).lower()
    if "ddos" in name: return "DDoS"
    if "dos" in name: return "DoS"
    if "patator" in name: return "Brute Force"
    if "portscan" in name or "port scan" in name: return "Port Scan"
    if "bot" in name: return "Botnet"
    if "web attack" in name: return "Web Attack"
    if "heartbleed" in name: return "Heartbleed"
    if "infiltration" in name: return "Infiltration"
    if "benign" in name or "normal" in name: return "Normal"
    return "Other"


def load_demo_samples():
    if st.session_state.demo_samples is not None:
        return st.session_state.demo_samples
    if not os.path.exists(DEMO_SAMPLES_PATH):
        return None
    try:
        with open(DEMO_SAMPLES_PATH, "rb") as f:
            st.session_state.demo_samples = pickle.load(f)
        return st.session_state.demo_samples
    except Exception as e:
        st.error(f"Could not load demo samples: {e}")
        return None


def load_model():
    if st.session_state.ml_model is not None:
        return st.session_state.ml_model
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import joblib
        st.session_state.ml_model = joblib.load(MODEL_PATH)
        return st.session_state.ml_model
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None


def process_demo_mode():
    if not st.session_state.demo_running:
        return

    samples = load_demo_samples()
    if samples is None:
        st.session_state.demo_running = False
        return

    model = load_model()
    if model is None:
        st.session_state.demo_running = False
        return

    if random.random() < 0.10 and len(samples.get("attacks", [])) > 0:
        flow_df = random.choice(samples["attacks"])
    else:
        if len(samples.get("benign", [])) > 0:
            flow_df = random.choice(samples["benign"])
        else:
            return

    try:
        prediction = model.predict(flow_df)[0]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(flow_df)[0]
            confidence = float(np.max(probs)) * 100
        else:
            confidence = 95.0
    except Exception as e:
        st.error(f"Prediction error in demo mode: {e}")
        st.session_state.demo_running = False
        return

    src_ip = f"192.168.1.{random.randint(2, 254)}"
    dst_ip = f"10.0.0.{random.randint(2, 254)}"
    protocol = random.choice(["TCP", "UDP", "ICMP"])
    now_str = datetime.now().strftime("%H:%M:%S")

    st.session_state.demo_packets += random.randint(1, 5)
    st.session_state.demo_flows += 1

    is_benign = str(prediction).upper() == "BENIGN"
    if is_benign:
        st.session_state.demo_normal += 1
    else:
        st.session_state.demo_threats += 1

    st.session_state.demo_last_pred = str(prediction)
    st.session_state.demo_last_conf = confidence
    st.session_state.demo_last_src = src_ip
    st.session_state.demo_last_dst = dst_ip
    st.session_state.demo_last_proto = protocol
    st.session_state.demo_last_time = now_str

    entry = {
        "Time": now_str,
        "Source IP": src_ip,
        "Destination IP": dst_ip,
        "Protocol": protocol,
        "Attack Type": str(prediction),
        "Severity": "Safe" if is_benign else "Critical",
        "Confidence": f"{round(confidence, 2)}%",
    }
    st.session_state.demo_activity.insert(0, entry)
    st.session_state.demo_activity = st.session_state.demo_activity[:100]

    if not is_benign:
        st.session_state.demo_alerts.insert(0, entry)
        st.session_state.demo_alerts = st.session_state.demo_alerts[:50]


# ============================================================
# SIDEBAR — NAVIGATION (uses session_state.page directly)
# ============================================================

with st.sidebar:
    st.markdown("## 🛡️ AI Cybersecurity IDS")
    st.markdown("---")

    if st.button("🏠 Home", width="stretch", key="nav_home"):
        st.session_state.page = "Home"
        st.rerun()

    if st.button("📊 Dashboard", width="stretch", key="nav_dash"):
        st.session_state.page = "Dashboard"
        st.rerun()

    st.markdown("---")

    if st.session_state.page == "Dashboard":
        st.markdown("### ⚙️ Data Source")

        if st.button(
            "🟢 Live Monitoring",
            width="stretch",
            key="mode_live",
            type="primary" if st.session_state.mode == "Live Monitoring" else "secondary"
        ):
            st.session_state.mode = "Live Monitoring"
            st.rerun()

        if st.button(
            "🧪 Demo Mode",
            width="stretch",
            key="mode_demo",
            type="primary" if st.session_state.mode == "Demo Mode" else "secondary"
        ):
            st.session_state.mode = "Demo Mode"
            st.rerun()

        if st.session_state.mode == "Demo Mode":
            st.warning(
                "Demo Mode replays real CICIDS2017 samples "
                "through the SAME Random Forest model."
            )
        else:
            st.success("Capturing real network traffic.")


# ============================================================
# HOME PAGE
# ============================================================

def render_home():
    st.markdown(
        """
<div class="hero">
<div class="hero-title">🛡️ AI-Based Cybersecurity Intrusion Detection & Threat Prediction System</div>
<div class="hero-sub">
A Machine Learning-powered Security Operations Center (SOC) that monitors live network traffic,
detects intrusions, and classifies attack types in real time using a Random Forest model trained on the CICIDS2017 dataset.
</div>
</div>
""",
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    features = [
        ("📡", "Real-Time Monitoring", "Captures live network packets and aggregates them into flows for analysis."),
        ("🤖", "ML-Powered Detection", "Random Forest model trained on 78 network features from CICIDS2017."),
        ("🚨", "Attack Classification", "Identifies DoS, DDoS, Brute Force, Port Scan, Botnet, and more."),
        ("📊", "Live SOC Dashboard", "Real-time metrics, charts, alert log, activity table, and exportable reports."),
    ]
    for col, (icon, title, text) in zip([c1, c2, c3, c4], features):
        with col:
            st.markdown(
                f"""
<div class="feature-card">
<div class="feature-icon">{icon}</div>
<div class="feature-title">{title}</div>
<div class="feature-text">{text}</div>
</div>
""",
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
<div class="panel">
<div class="panel-title">🧭 How to Use</div>
<div style="color:#9eb0c8;font-size:14px;line-height:1.9;">
<b style="color:#00e5ff;">1. Live Monitoring</b> — Captures real packets from your network interface and runs them through the trained model.<br>
<b style="color:#00e5ff;">2. Demo Mode</b> — Replays real attack samples from the CICIDS2017 dataset through the SAME model, useful for demonstrations.<br>
<b style="color:#00e5ff;">3. Dashboard</b> — Shows live metrics, traffic charts, attack distribution, alert log, and recent activity.<br>
<b style="color:#00e5ff;">4. Exports</b> — Download alert logs, activity tables, and session reports as CSV/PDF.
</div>
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("➡ Go to Dashboard", width="stretch", key="home_goto_dash"):
        st.session_state.page = "Dashboard"
        st.rerun()


# ============================================================
# DASHBOARD PAGE
# ============================================================

def render_dashboard():
    demo_mode = (st.session_state.mode == "Demo Mode")

    if demo_mode:
        process_demo_mode()

    data = read_results()
    packets = flows = normal = threats = 0
    last_prediction = None
    confidence = 0
    source = destination = protocol = last_time = "-"

    if data:
        packets = int(data.get("packets", 0))
        flows = int(data.get("flows", 0))
        normal = int(data.get("normal", 0))
        threats = int(data.get("attacks", 0))
        last_prediction = data.get("last_prediction")
        confidence = float(data.get("last_confidence", 0))
        source = data.get("last_source", "-")
        destination = data.get("last_destination", "-")
        protocol = data.get("last_protocol", "-")
        last_time = data.get("last_time", "-")

    if demo_mode:
        packets = st.session_state.demo_packets
        flows = st.session_state.demo_flows
        normal = st.session_state.demo_normal
        threats = st.session_state.demo_threats
        last_prediction = st.session_state.demo_last_pred
        confidence = st.session_state.demo_last_conf
        source = st.session_state.demo_last_src
        destination = st.session_state.demo_last_dst
        protocol = st.session_state.demo_last_proto
        last_time = st.session_state.demo_last_time

    if (not demo_mode) and last_prediction:
        sig = (str(last_prediction), str(source), str(destination), str(last_time))
        if sig != st.session_state.previous_prediction:
            is_threat = (last_prediction != "BENIGN")
            entry = {
                "Time": last_time,
                "Source IP": source,
                "Destination IP": destination,
                "Protocol": protocol,
                "Attack Type": last_prediction,
                "Severity": "Critical" if is_threat else "Safe",
                "Confidence": f"{round(confidence, 2)}%",
            }
            st.session_state.activity.insert(0, entry)
            st.session_state.activity = st.session_state.activity[:100]
            if is_threat:
                st.session_state.alerts.insert(0, entry)
                st.session_state.alerts = st.session_state.alerts[:50]
            st.session_state.previous_prediction = sig

    if demo_mode:
        status = '<span class="demo-badge">🧪 DEMO MODE</span>'
    elif st.session_state.monitoring:
        status = '<span class="online">● SYSTEM ONLINE</span>'
    else:
        status = '<span class="offline">● SYSTEM OFFLINE</span>'

    st.markdown(
        f"""
<div class="header">
<div style="display:flex;align-items:center;justify-content:space-between;">
<div style="display:flex;align-items:center;gap:18px;">
<div class="logo">🛡️</div>
<div>
<div class="main-title">AI-Based Cybersecurity Intrusion Detection & Threat Prediction System</div>
<div class="sub-title">SECURITY OPERATIONS CENTER • MACHINE LEARNING IDS</div>
</div>
</div>
<div style="text-align:right;">
<div style="color:#71839d;font-size:12px;">
{datetime.now().strftime("%d %b %Y • %H:%M:%S")}
</div>
<div style="margin-top:8px;">{status}</div>
</div>
</div>
</div>
""",
        unsafe_allow_html=True
    )

    if demo_mode:
        if not os.path.exists(DEMO_SAMPLES_PATH):
            st.error(
                f"⚠️ Demo samples file not found: `{DEMO_SAMPLES_PATH}`\n\n"
                "Run `python generate_demo_samples.py` first to create it."
            )
        else:
            st.info(
                "🧪 **Demo Mode active** — replaying real CICIDS2017 samples through the "
                "same Random Forest model. Predictions are genuine model output."
            )

    b1, b2, b3 = st.columns([1, 1, 3])

    with b1:
        if demo_mode:
            if not st.session_state.demo_running:
                if st.button("▶ START DEMO", width="stretch", key="start_demo_btn"):
                    st.session_state.demo_running = True
                    st.session_state.demo_start_time = time.time()
                    st.session_state.demo_alerts = []
                    st.session_state.demo_activity = []
                    st.session_state.demo_traffic_history = []
                    st.session_state.demo_packets = 0
                    st.session_state.demo_flows = 0
                    st.session_state.demo_normal = 0
                    st.session_state.demo_threats = 0
                    st.rerun()
            else:
                if st.button("⏹ STOP DEMO", width="stretch", key="stop_demo_btn"):
                    st.session_state.demo_running = False
                    st.rerun()
        else:
            if not st.session_state.monitoring:
                if st.button("▶ START MONITORING", width="stretch", key="start_mon_btn"):
                    start_monitor()
                    st.rerun()
            else:
                if st.button("⏹ STOP MONITORING", width="stretch", key="stop_mon_btn"):
                    stop_monitor()
                    st.rerun()

    with b2:
        if st.button("↻ REFRESH", width="stretch", key="refresh_btn"):
            st.rerun()

    with b3:
        if demo_mode:
            if st.session_state.demo_running:
                st.success("Demo mode running — replaying CICIDS2017 samples through the trained model.")
            else:
                st.warning("Demo mode ready. Click START DEMO to begin.")
        else:
            if st.session_state.monitoring:
                st.success("Live network monitoring is active.")
            else:
                st.warning("System offline. Click START MONITORING to begin.")

    st.markdown('<div class="panel-title">LIVE SECURITY METRICS</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    for col, icon, title, value in [
        (m1, "📦", "Packets Analyzed", packets),
        (m2, "🔗", "Flows Detected", flows),
        (m3, "🟢", "Normal Traffic", normal),
        (m4, "🚨", "Threats Detected", threats),
    ]:
        with col:
            st.markdown(
                f"""
<div class="card">
<div class="card-icon">{icon}</div>
<div class="card-title">{title}</div>
<div class="card-value">{value:,}</div>
</div>
""",
                unsafe_allow_html=True
            )

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="panel"><div class="panel-title">📈 Live Network Traffic</div>', unsafe_allow_html=True)
        hist_key = "demo_traffic_history" if demo_mode else "traffic_history"
        st.session_state[hist_key].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "normal": normal,
            "threat": threats,
        })
        st.session_state[hist_key] = st.session_state[hist_key][-60:]

        traffic_df = pd.DataFrame(st.session_state[hist_key])
        fig = go.Figure()
        if not traffic_df.empty:
            fig.add_trace(go.Scatter(
                x=traffic_df["time"], y=traffic_df["normal"],
                mode="lines+markers", name="Normal",
                line=dict(color="#00e5ff", width=3)
            ))
            fig.add_trace(go.Scatter(
                x=traffic_df["time"], y=traffic_df["threat"],
                mode="lines+markers", name="Threats",
                line=dict(color="#ef4444", width=3)
            ))
        fig.update_layout(
            height=330,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9eb0c8"),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="rgba(255,255,255,.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,.05)"),
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel"><div class="panel-title">🍩 Attack Distribution</div>', unsafe_allow_html=True)
        categories = {"Normal": normal, "DoS": 0, "DDoS": 0, "Brute Force": 0, "Port Scan": 0}
        alert_src = st.session_state.demo_alerts if demo_mode else st.session_state.alerts
        for alert in alert_src:
            cat = get_attack_category(alert.get("Attack Type", ""))
            if cat in categories:
                categories[cat] += 1

        labels = list(categories.keys())
        values = list(categories.values())
        if sum(values) == 0:
            values = [1, 0, 0, 0, 0]

        fig2 = go.Figure(data=[go.Pie(
            labels=labels, values=values, hole=0.60,
            textinfo="label+percent",
            marker=dict(colors=["#22c55e", "#f59e0b", "#ef4444", "#facc15", "#a855f7"])
        )])
        fig2.update_layout(
            height=330,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#dcecff"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig2, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    if last_prediction:
        threat = (str(last_prediction).upper() != "BENIGN")
        panel_class = "pred-threat" if threat else "pred-normal"
        text_class = "red" if threat else "green"
        title = "🚨 THREAT DETECTED" if threat else "🟢 NORMAL TRAFFIC"

        st.markdown(
            f"""
<div class="panel {panel_class}">
<div class="panel-title">🔍 Current ML Prediction</div>
<div style="color:#71839d;font-size:12px;font-weight:bold;">{title}</div>
<div class="prediction {text_class}">{last_prediction}</div>
<div style="margin-top:15px;">
<div class="info-label">MODEL CONFIDENCE: {confidence:.2f}%</div>
<div style="background:#172338;height:9px;border-radius:10px;margin-top:6px;">
<div style="width:{min(confidence, 100)}%;height:9px;border-radius:10px;background:{'#ef4444' if threat else '#22c55e'};"></div>
</div>
</div>
<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:15px;margin-top:20px;">
<div><div class="info-label">SOURCE IP</div><div class="info-value">{source}</div></div>
<div><div class="info-label">DESTINATION IP</div><div class="info-value">{destination}</div></div>
<div><div class="info-label">PROTOCOL</div><div class="info-value">{protocol}</div></div>
<div><div class="info-label">ATTACK TYPE</div><div class="info-value">{last_prediction}</div></div>
<div><div class="info-label">TIME</div><div class="info-value">{last_time}</div></div>
</div>
</div>
""",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
<div class="panel">
<div class="panel-title">🔍 Current ML Prediction</div>
<div style="color:#71839d;padding:20px 0;">Waiting for network traffic...</div>
</div>
""",
            unsafe_allow_html=True
        )

    st.markdown(
        """
<div class="panel">
<div class="panel-title">⚙️ Detection Pipeline</div>
<div class="pipeline">
<div class="stage"><div class="stage-icon">🌐</div><div class="stage-text">Network Traffic</div></div>
<div class="arrow">→</div>
<div class="stage"><div class="stage-icon">📡</div><div class="stage-text">Packet Capture</div></div>
<div class="arrow">→</div>
<div class="stage"><div class="stage-icon">🔧</div><div class="stage-text">Feature Extraction</div></div>
<div class="arrow">→</div>
<div class="stage"><div class="stage-icon">🤖</div><div class="stage-text">Random Forest</div></div>
<div class="arrow">→</div>
<div class="stage"><div class="stage-icon">🚨</div><div class="stage-text">Threat Classification</div></div>
</div>
</div>
""",
        unsafe_allow_html=True
    )

    left2, right2 = st.columns([1.5, 1])

    with left2:
        st.markdown('<div class="panel"><div class="panel-title">🚨 Alert Log</div>', unsafe_allow_html=True)
        alerts = st.session_state.demo_alerts if demo_mode else st.session_state.alerts
        if not alerts:
            st.info("No threats detected yet.")
        else:
            for alert in alerts[:10]:
                st.markdown(
                    f"""
<div class="alert">
<div class="alert-title">{alert["Attack Type"]}</div>
<div class="alert-info">
{alert["Time"]} &nbsp;|&nbsp; {alert["Source IP"]} → {alert["Destination IP"]} &nbsp;|&nbsp; {alert["Protocol"]} &nbsp;|&nbsp; Confidence: {alert["Confidence"]}
</div>
</div>
""",
                    unsafe_allow_html=True
                )
        st.markdown("</div>", unsafe_allow_html=True)

    with right2:
        st.markdown('<div class="panel"><div class="panel-title">🖥️ System Status</div>', unsafe_allow_html=True)
        model_status = "ONLINE" if os.path.exists(MODEL_PATH) else "NOT FOUND"
        cpu = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory().percent

        if demo_mode and st.session_state.demo_start_time:
            uptime_seconds = int(time.time() - st.session_state.demo_start_time)
        elif st.session_state.start_time:
            uptime_seconds = int(time.time() - st.session_state.start_time)
        else:
            uptime_seconds = 0

        st.markdown(
            f"""
<div style="line-height:2.1;">
<div><span class="info-label">MODEL</span><br><b style="color:#22c55e;">Random Forest • {model_status}</b></div>
<div><span class="info-label">DATASET</span><br><b>CICIDS2017</b></div>
<div><span class="info-label">FEATURES</span><br><b>78 Network Features</b></div>
<div><span class="info-label">CLASSES</span><br><b>15 Attack Classes</b></div>
<div><span class="info-label">CPU</span><br><b>{cpu:.1f}%</b></div>
<div><span class="info-label">MEMORY</span><br><b>{memory:.1f}%</b></div>
<div><span class="info-label">UPTIME</span><br><b>{uptime_seconds // 60}m {uptime_seconds % 60}s</b></div>
<div><span class="info-label">THREAT BLOCKING</span><br><b style="color:#f59e0b;">DETECTION ONLY</b></div>
</div>
""",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">📋 Recent Network Activity</div>', unsafe_allow_html=True)
    activity = st.session_state.demo_activity if demo_mode else st.session_state.activity
    if activity:
        activity_df = pd.DataFrame(activity)
        st.dataframe(activity_df, width="stretch", hide_index=True)
    else:
        st.info("Waiting for analyzed network flows...")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">📤 Export Reports</div>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)

    with e1:
        alert_df = pd.DataFrame(alerts)
        alert_csv = alert_df.to_csv(index=False) if not alert_df.empty else "No threats detected"
        st.download_button("⬇ Alert Log CSV", alert_csv, "alert_log.csv", "text/csv", width="stretch")

    with e2:
        activity_df2 = pd.DataFrame(activity)
        activity_csv = activity_df2.to_csv(index=False) if not activity_df2.empty else "No activity recorded"
        st.download_button("⬇ Activity CSV", activity_csv, "network_activity.csv", "text/csv", width="stretch")

    with e3:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas

            buffer = BytesIO()
            pdf = canvas.Canvas(buffer, pagesize=A4)
            width_pdf, height_pdf = A4
            pdf.setFont("Helvetica-Bold", 16)
            pdf.drawString(40, height_pdf - 50, "AI Cybersecurity IDS Session Report")
            pdf.setFont("Helvetica", 10)
            y = height_pdf - 90
            report_lines = [
                f"Mode: {'Demo (CICIDS2017 replay)' if demo_mode else 'Live Monitoring'}",
                "Dataset: CICIDS2017",
                "Model: Random Forest",
                "Features: 78",
                "Classes: 15",
                "",
                f"Packets Analyzed: {packets}",
                f"Flows Detected: {flows}",
                f"Normal Traffic: {normal}",
                f"Threats Detected: {threats}",
                "",
                "System Type: Intrusion Detection System",
                "Threat Blocking: Disabled",
                "Purpose: Detection and Classification",
            ]
            for line in report_lines:
                pdf.drawString(40, y, line)
                y -= 18
            pdf.save()
            buffer.seek(0)
            st.download_button(
                "⬇ Session PDF",
                buffer,
                "ids_session_report.pdf",
                "application/pdf",
                width="stretch"
            )
        except ImportError:
            st.warning("Install reportlab: pip install reportlab")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# ROUTER
# ============================================================

if st.session_state.page == "Home":
    render_home()
else:
    render_dashboard()


# ============================================================
# AUTO REFRESH
# ============================================================

if st.session_state.mode == "Demo Mode" and st.session_state.demo_running:
    time.sleep(1)
    st.rerun()

if st.session_state.mode == "Live Monitoring" and st.session_state.monitoring:
    time.sleep(1)
    st.rerun()