import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.markdown('<p class="cyber-title">TELEMETRY & ANALYTICS</p>', unsafe_allow_html=True)
st.markdown('<p class="cyber-subtitle">Advanced Neural Inference Statistics & Detection Insights</p>', unsafe_allow_html=True)

# Generate simulated rich telemetry data for visualization
np.random.seed(42)
n_samples = 150
timestamps = pd.date_range(start="2026-10-02 08:00:00", periods=n_samples, freq="10s")
confidences = np.random.uniform(0.75, 0.99, n_samples)
bottle_counts = np.random.poisson(lam=4, size=n_samples)
defect_status = np.random.choice(["Pristine", "Minor Scratch", "Cap Mismatch", "Underfilled"], size=n_samples, p=[0.7, 0.15, 0.1, 0.05])

df_telemetry = pd.DataFrame({
    "Timestamp": timestamps,
    "Confidence": confidences,
    "BottleCount": bottle_counts,
    "Status": defect_status
})

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 Confidence Score Distribution")
    fig_hist = px.histogram(
        df_telemetry, x="Confidence", nbins=20,
        color_discrete_sequence=["#00f2fe"],
        template="plotly_dark"
    )
    fig_hist.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.markdown("### 🍾 Bottle Detection Volume Over Time")
    fig_line = px.line(
        df_telemetry, x="Timestamp", y="BottleCount",
        color_discrete_sequence=["#4facfe"],
        template="plotly_dark"
    )
    fig_line.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_line, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.markdown("### 🏷️ Quality Control Classification")
    status_counts = df_telemetry["Status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]
    fig_pie = px.pie(
        status_counts, names="Status", values="Count",
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Teal,
        template="plotly_dark"
    )
    fig_pie.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col4:
    st.markdown("### 📊 Raw Telemetry Logs")
    st.dataframe(df_telemetry.tail(10), use_container_width=True)
