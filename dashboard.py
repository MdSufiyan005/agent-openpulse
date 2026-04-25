import streamlit as st
import pandas as pd
import json
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --- Configuration & Paths ---
st.set_page_config(
    page_title="Chiseled Dashboard",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

ARTIFACTS_DIR = "artifacts"
RESULTS_DIR = os.path.join(ARTIFACTS_DIR, "results")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.jsonl")
AGENT_LOG_PATH = os.path.join(RESULTS_DIR, "agent_log.json")
ENV_PATH = ".env"

# --- Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4150;
    }
    .agent-log {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        font-family: 'Courier New', Courier, monospace;
        color: #d1d1d1;
        max-height: 400px;
        overflow-y: auto;
        border-left: 5px solid #4e8cff;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return pd.DataFrame()
    
    data = []
    with open(METRICS_PATH, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    return pd.DataFrame(data)

def load_agent_log():
    if not os.path.exists(AGENT_LOG_PATH):
        return []
    with open(AGENT_LOG_PATH, "r") as f:
        return json.load(f)

def load_env():
    env = {}
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    env[k] = v
    return env

def save_env(env_updates):
    current_env = load_env()
    current_env.update(env_updates)
    with open(ENV_PATH, "w") as f:
        for k, v in current_env.items():
            f.write(f"{k}={v}\n")

# --- Sidebar: Controls ---
st.sidebar.title("🛠️ Chiseled Controls")
env = load_env()

st.sidebar.subheader("Model Settings")
model_id = st.sidebar.text_input("MODEL_ID", env.get("MODEL_ID", "Qwen/Qwen2.5-VL-2B-Instruct"))
llm_provider = st.sidebar.selectbox("LLM_PROVIDER", ["openrouter", "groq", "openai"], index=["openrouter", "groq", "openai"].index(env.get("LLM_PROVIDER", "openrouter")))

st.sidebar.subheader("Optimization Params")
kl_threshold = st.sidebar.slider("KL_THRESHOLD", 0.01, 0.20, float(env.get("KL_THRESHOLD", 0.05)))
max_images = st.sidebar.number_input("MAX_IMAGES", 1, 10, int(env.get("MAX_IMAGES", 3)))

if st.sidebar.button("🚀 Apply Changes & Restart"):
    save_env({
        "MODEL_ID": model_id,
        "LLM_PROVIDER": llm_provider,
        "KL_THRESHOLD": str(kl_threshold),
        "MAX_IMAGES": str(max_images)
    })
    st.sidebar.success("Settings saved! Restart the container to apply.")
    # In a real setup, we could trigger a restart via docker-compose restart
    # But for now, we just inform the user.

# --- Main Dashboard ---
st.title("🛠️ Chiseled Optimization Dashboard")
st.markdown("---")

df = load_metrics()
logs = load_agent_log()

if df.empty:
    st.warning("No metrics found yet. Start the pipeline to see real-time optimization!")
else:
    # Top Row: Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2] if len(df) > 1 else last_row
    
    def delta(curr, prev):
        try:
            return float(curr) - float(prev)
        except:
            return 0

    col1.metric("Current TPS", f"{last_row.get('tps', 'N/A')}", delta(last_row.get('tps', 0), prev_row.get('tps', 0)))
    col2.metric("Latency (ms)", f"{last_row.get('latency', 'N/A')}", delta(last_row.get('latency', 0), prev_row.get('latency', 0)), delta_color="inverse")
    col3.metric("RAM (MB)", f"{last_row.get('ram', 'N/A')}", delta(last_row.get('ram', 0), prev_row.get('ram', 0)), delta_color="inverse")
    col4.metric("Iteration", len(df))

    st.markdown("### 📈 Performance Trends")
    
    # Charts Row
    c1, c2 = st.columns(2)
    
    # TPS Chart
    fig_tps = px.line(df, y="tps", title="Tokens Per Second (Higher is better)", markers=True)
    fig_tps.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)')
    c1.plotly_chart(fig_tps, use_container_width=True)
    
    # Latency/RAM Chart
    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Scatter(y=df['latency'], name="Latency (ms)", mode='lines+markers'))
    fig_metrics.add_trace(go.Scatter(y=df['ram'], name="RAM (MB)", mode='lines+markers'))
    fig_metrics.update_layout(title="Latency & RAM Trends (Lower is better)", template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)')
    c2.plotly_chart(fig_metrics, use_container_width=True)

# Row: Agent Reasoning & Comparison Table
st.markdown("---")
left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown("### 🧠 Agent Reasoning Log")
    if logs:
        for log in reversed(logs):
            st.markdown(f'<div class="agent-log">{log}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info("Waiting for agent to store memory...")

with right_col:
    st.markdown("### 📋 Detailed Metrics History")
    if not df.empty:
        st.dataframe(df.style.highlight_max(axis=0, subset=['tps']).highlight_min(axis=0, subset=['latency', 'ram']), use_container_width=True)
    else:
        st.info("No table data available yet.")

# Auto-refresh
time.sleep(5)
st.rerun()
