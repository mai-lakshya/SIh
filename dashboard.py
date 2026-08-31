import streamlit as st
import optuna
import pandas as pd
import sqlite3
import os
import time
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="God Mode Training Dashboard", layout="wide", page_icon="🚀")

# Custom CSS for dark theme and animations
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .metric-card {
        background: linear-gradient(145deg, #1E232E, #2B3240);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    h1, h2, h3 {
        color: #00F0FF;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌌 God Mode Training Dashboard")
st.markdown("### Real-time 12-Hour Optimization Tracker")

# Check if DB exists
db_path = "god_mode_study.db"
if not os.path.exists(db_path):
    st.warning("Database not found yet. The training script might still be initializing...")
    st.stop()

# Connect to DB and fetch studies
try:
    conn = sqlite3.connect(db_path)
    studies_df = pd.read_sql("SELECT * FROM studies", conn)
    conn.close()
    
    if studies_df.empty:
        st.info("Database initialized, but no studies found yet. Waiting for first fold to start...")
        st.stop()
        
    study_name = studies_df.iloc[-1]['study_name']
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")
    
    trials = study.trials
    completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><h3>Active Study</h3><h2>{study_name.split('_')[2] if len(study_name.split('_'))>2 else 'Fold 0'}</h2></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h3>Total Trials</h3><h2>{len(trials)}</h2></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h3>Completed</h3><h2>{len(completed_trials)}</h2></div>", unsafe_allow_html=True)
    with col4:
        # Calculate time elapsed
        if len(trials) > 0 and trials[0].datetime_start is not None:
            elapsed = pd.Timestamp.now(tz=trials[0].datetime_start.tzinfo) - trials[0].datetime_start
            hours, remainder = divmod(elapsed.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{hours}h {minutes}m {seconds}s"
        else:
            time_str = "0h 0m 0s"
        st.markdown(f"<div class='metric-card'><h3>Time Elapsed</h3><h2>{time_str}</h2></div>", unsafe_allow_html=True)

    if len(completed_trials) > 0:
        st.markdown("---")
        st.subheader("Performance Metrics (Pareto Front Approximation)")
        
        # Multi-objective Pareto extraction
        pareto_trials = study.best_trials
        
        if len(pareto_trials) > 0:
            best_t = pareto_trials[0]
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Best Recall (Max)", f"{best_t.values[0]:.4f}")
            m_col2.metric("Best ECE (Min)", f"{best_t.values[1]:.4f}")
            m_col3.metric("Best CRS RMSE (Min)", f"{best_t.values[2]:.2f}")
            m_col4.metric("Best C-Index (Max)", f"{best_t.values[3]:.4f}")
            
        # History Plot
        df = study.trials_dataframe()
        if not df.empty and 'values_0' in df.columns:
            st.subheader("Optimization History")
            fig = px.scatter(df, x='number', y='values_0', color='state',
                             title="Recall (Objective 0) over Trials",
                             labels={'number': 'Trial Number', 'values_0': 'Recall'})
            fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            
except Exception as e:
    st.error(f"Error loading dashboard: {str(e)}")
    
st.markdown("---")
st.subheader("Live Logs (training_12hr.log)")
try:
    with open('training_12hr.log', 'r') as f:
        # Read last 20 lines
        lines = f.readlines()
        log_text = "".join(lines[-20:])
        st.code(log_text, language='bash')
except FileNotFoundError:
    st.write("Log file not found yet.")

# Auto-refresh
time.sleep(5)
st.rerun()
