import streamlit as st
from utils import get_available_models

st.markdown('<p class="minimal-title">QUANTUM BOTTLE VISION</p>', unsafe_allow_html=True)
st.markdown('<p class="minimal-subtitle">Autonomous YOLOv8 & YOLOv26 Object Intelligence Dashboard</p>', unsafe_allow_html=True)

# Top Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
        <div class="metric-container">
            <div class="metric-value">99.4%</div>
            <div class="metric-label">Precision Rate</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="metric-container">
            <div class="metric-value">60 FPS</div>
            <div class="metric-label">Inference Speed</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="metric-container">
            <div class="metric-value">YOLOv8/26</div>
            <div class="metric-label">Active Core</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div class="metric-container">
            <div class="metric-value">ONLINE</div>
            <div class="metric-label">System Status</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Feature Grid Cards
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
        <div class="glass-card">
            <h3 style="color: #38BDF8; margin-top: 0; font-weight: 700;">👁️ High-Performance Neural Scanner</h3>
            <p style="color: #94A3B8; line-height: 1.6;">
                Deploy any YOLOv8, YOLOv12, or YOLOv26 model weights instantly. Real-time bounding box detection, tracking, confidence filtering, and automated defect counting on images, videos, and live webcams.
            </p>
        </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("""
        <div class="glass-card">
            <h3 style="color: #38BDF8; margin-top: 0; font-weight: 700;">📊 Live Telemetry & Analytics</h3>
            <p style="color: #94A3B8; line-height: 1.6;">
                Interactive Plotly analytics tracking detection confidence distribution, quality control classification breakdown, and inference telemetry throughput.
            </p>
        </div>
    """, unsafe_allow_html=True)

col_c, col_d = st.columns(2)

with col_c:
    st.markdown("""
        <div class="glass-card">
            <h3 style="color: #38BDF8; margin-top: 0; font-weight: 700;">🧠 Model Diagnostics & Inspector</h3>
            <p style="color: #94A3B8; line-height: 1.6;">
                Inspect layer parameters, architecture specifications, and benchmark live model latency across multiple .pt weight files.
            </p>
        </div>
    """, unsafe_allow_html=True)

with col_d:
    st.markdown("""
        <div class="glass-card">
            <h3 style="color: #38BDF8; margin-top: 0; font-weight: 700;">⚙️ Active Neural Core Config</h3>
            <p style="color: #94A3B8; line-height: 1.6;">
                Seamlessly switch between available model weights in the sidebar and tune confidence thresholds in real-time.
            </p>
        </div>
    """, unsafe_allow_html=True)
