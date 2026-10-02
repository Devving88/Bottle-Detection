import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
from utils import load_selected_model, get_available_models

UPLOAD_DIR = "upload"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.markdown('<p class="minimal-title">NEURAL SCANNER</p>', unsafe_allow_html=True)
st.markdown('<p class="minimal-subtitle">High-Precision YOLO Object Detection & Real-Time Tracking Engine</p>', unsafe_allow_html=True)

# Sidebar controls for detection
with st.sidebar:
    st.markdown("### 🎛️ Active Neural Core")
    available_models = get_available_models()
    selected_model_name = st.selectbox("Select Model Weights", list(available_models.keys()))
    model_path = available_models[selected_model_name]
    
    st.markdown("---")
    st.markdown("### ⚙️ Inference Parameters")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.40, 0.05)
    iou_threshold = st.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.05)
    
    box_color_hex = st.color_picker("Bounding Box Accent", "#38BDF8")
    hex_c = box_color_hex.lstrip('#')
    rgb_c = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
    BOX_COLOR = (rgb_c[2], rgb_c[1], rgb_c[0])  # BGR

model = load_selected_model(model_path)

# Tabs for input modes
tab_img, tab_vid, tab_cam = st.tabs(["🖼️ Image Scanner", "🎬 Video Tracking", "🔴 Live Webcam"])

with tab_img:
    st.markdown("""
        <div class="glass-card">
            <h3 style="color: #38BDF8; margin-top: 0;">Still Image Object Detection</h3>
            <p style="color: #94A3B8; font-size: 0.95rem;">Upload high-resolution bottle imagery for instantaneous neural classification and bounding box localization.</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="img_uploader")
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        img_np = np.array(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📥 Original Source")
            st.image(image, use_container_width=True)
            
        with col2:
            st.markdown("#### ⚡ Neural Inference Result")
            with st.spinner("Executing neural inference layers..."):
                results = model(img_np, conf=conf_threshold, iou=iou_threshold)
                res_plotted = results[0].plot()
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                st.image(res_rgb, use_container_width=True)
                
                boxes = results[0].boxes
                count = len(boxes) if boxes is not None else 0
                st.success(f"🎯 Detection Complete! Detected **{count}** object(s).")
                
                if count > 0:
                    data = []
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        cls_name = model.names[cls_id]
                        data.append({"Class": cls_name, "Confidence Score": f"{conf:.2%}"})
                    df_det = pd.DataFrame(data)
                    st.dataframe(df_det, use_container_width=True)

with tab_vid:
    st.markdown("""
        <div class="glass-card">
            <h3 style="color: #38BDF8; margin-top: 0;">Video Stream Object Tracking</h3>
            <p style="color: #94A3B8; font-size: 0.95rem;">Upload video recordings to run frame-by-frame YOLO tracking across temporal sequences.</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"], key="vid_uploader")
    
    if uploaded_video is not None:
        video_path = os.path.join(UPLOAD_DIR, uploaded_video.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())
            
        st.video(video_path)
        
        if st.button("🚀 Launch Video Neural Tracker"):
            cap = cv2.VideoCapture(video_path)
            frame_placeholder = st.empty()
            metric_placeholder = st.empty()
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                
                results = model.track(frame, persist=True, conf=conf_threshold, iou=iou_threshold, verbose=False)
                res_plotted = results[0].plot()
                
                boxes = results[0].boxes
                current_count = len(boxes) if boxes is not None else 0
                
                frame_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                metric_placeholder.info(f"📊 Processing Frame: {frame_count} | Active Bottles Tracked: {current_count}")
                
            cap.release()
            st.success(f"🎬 Video tracking successfully finished across {frame_count} frames!")

with tab_cam:
    st.markdown("""
        <div class="glass-card">
            <h3 style="color: #38BDF8; margin-top: 0;">Live Camera Stream</h3>
            <p style="color: #94A3B8; font-size: 0.95rem;">Connect to local camera feed for real-time edge AI object recognition.</p>
        </div>
    """, unsafe_allow_html=True)
    
    run_cam = st.checkbox("🟢 Enable Live Webcam Feed")
    
    if run_cam:
        cam_placeholder = st.empty()
        stop_cam = st.button("🔴 Stop Webcam Stream")
        
        cap = cv2.VideoCapture(0)
        while cap.isOpened() and not stop_cam:
            ret, frame = cap.read()
            if not ret:
                st.error("Unable to access local camera device.")
                break
                
            results = model.track(frame, persist=True, conf=conf_threshold, iou=iou_threshold, verbose=False)
            res_plotted = results[0].plot()
            frame_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            cam_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
        cap.release()
