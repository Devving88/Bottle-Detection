import streamlit as st
from utils import inject_custom_css

st.set_page_config(
    page_title="QUANTUM BOTTLE VISION // AI Cyber-Matrix",
    page_icon="🍾",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()

# Define multi-page setup
home_page = st.Page("pages/home.py", title="Command Center", icon="⚡")
detect_page = st.Page("pages/bottle_detection_app.py", title="Neural Scanner", icon="👁️")
analytics_page = st.Page("pages/analytics.py", title="Telemetry & Analytics", icon="📊")
diagnostics_page = st.Page("pages/model_diagnostics.py", title="Model Diagnostics", icon="🧠")
settings_page = st.Page("pages/settings.py", title="Neural Config", icon="⚙️")

pg = st.navigation({
    "MAIN OVERVIEW": [home_page],
    "DETECTION ENGINE": [detect_page],
    "INTELLIGENCE": [analytics_page, diagnostics_page],
    "SYSTEM": [settings_page]
})

# Sidebar branding footer
with st.sidebar:
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #8b9bb4; font-size: 0.8rem;">
            <p style="font-family: 'Orbitron'; color: #00f2fe; font-weight: 700;">QUANTUM BOTTLE VISION v3.0</p>
            <p>Powered by YOLO Neural Networks & Streamlit</p>
            <p style="color: #4facfe;">🟢 SYSTEM ONLINE</p>
        </div>
        """,
        unsafe_allow_html=True
    )

pg.run()
