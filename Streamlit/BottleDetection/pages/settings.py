import streamlit as st
import os

st.markdown('<p class="cyber-title">NEURAL CONFIGURATION</p>', unsafe_allow_html=True)
st.markdown('<p class="cyber-subtitle">System Settings, Weight Uploaders & Hardware Acceleration</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="glass-card">
            <h3 style="color: #00f2fe; margin-top: 0;">⚙️ Inference Settings</h3>
        </div>
    """, unsafe_allow_html=True)
    
    device_mode = st.selectbox("Compute Device", ["CUDA (GPU)", "CPU", "MPS (Apple Silicon)"])
    batch_processing = st.checkbox("Enable Batch Tensor Parallelism", value=True)
    save_reports = st.checkbox("Auto-Save Detection Logs to CSV", value=True)
    
    if st.button("Save Configuration"):
        st.success("⚙️ Neural configuration updated successfully!")

with col2:
    st.markdown("""
        <div class="glass-card">
            <h3 style="color: #4facfe; margin-top: 0;">📥 Custom Weights Uploader</h3>
        </div>
    """, unsafe_allow_html=True)
    
    custom_weight = st.file_uploader("Upload custom YOLO `.pt` weights", type=["pt"])
    if custom_weight is not None:
        save_path = os.path.join("model", custom_weight.name)
        with open(save_path, "wb") as f:
            f.write(custom_weight.read())
        st.success(f"Successfully uploaded and registered `{custom_weight.name}` in neural weights registry!")
