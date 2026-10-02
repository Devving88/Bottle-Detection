import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
from utils import load_selected_model, get_available_models
import time

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    import av
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False

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
            <h3 style="color: #38BDF8; margin-top: 0;">🖼️ Still Image Object Detection</h3>
            <p style="color: #94A3B8; font-size: 0.95rem;">Upload high-resolution bottle imagery for instantaneous neural classification, bounding box localization, and defect inspection.</p>
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
                start_time = time.time()
                results = model(img_np, conf=conf_threshold, iou=iou_threshold)
                inference_time = (time.time() - start_time) * 1000

                res_plotted = results[0].plot()
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                st.image(res_rgb, use_container_width=True)

                boxes = results[0].boxes
                count = len(boxes) if boxes is not None else 0

                # Metric Cards
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.markdown(f'<div class="metric-container"><div class="metric-value">{count}</div><div class="metric-label">Detected Objects</div></div>', unsafe_allow_html=True)
                with m_col2:
                    avg_conf = float(boxes.conf.mean()) * 100 if count > 0 else 0
                    st.markdown(f'<div class="metric-container"><div class="metric-value">{avg_conf:.1f}%</div><div class="metric-label">Avg Confidence</div></div>', unsafe_allow_html=True)
                with m_col3:
                    st.markdown(f'<div class="metric-container"><div class="metric-value">{inference_time:.0f}ms</div><div class="metric-label">Latency</div></div>', unsafe_allow_html=True)

                if count > 0:
                    data = []
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        cls_name = model.names[cls_id]
                        data.append({"Class": cls_name, "Confidence Score": f"{conf:.2%}"})
                    df_det = pd.DataFrame(data)
                    st.markdown("#### 📋 Detected Instances Breakdown")
                    st.dataframe(df_det, use_container_width=True)

                    # Download annotated image button
                    buf = cv2.imencode('.png', res_plotted)[1].tobytes()
                    st.download_button(
                        label="📥 Download Annotated Image",
                        data=buf,
                        file_name="detected_bottle_result.png",
                        mime="image/png"
                    )

with tab_vid:
    st.markdown("""
        <div class="glass-card">
            <h3 style="color: #38BDF8; margin-top: 0;">🎬 Video Stream Object Tracking</h3>
            <p style="color: #94A3B8; font-size: 0.95rem;">Upload video recordings to run frame-by-frame YOLO tracking across temporal sequences with persistent ID assignment.</p>
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
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30

            frame_placeholder = st.empty()
            metric_placeholder = st.empty()
            progress_bar = st.progress(0)
            
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
                metric_placeholder.info(f"📊 Processing Frame: {frame_count} / {total_frames} | Active Objects Tracked: {current_count}")
                
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                
            cap.release()
            progress_bar.empty()
            st.success(f"🎬 Video tracking successfully finished across {frame_count} frames!")

with tab_cam:
    st.markdown("""
        <div class="glass-card">
            <h3 style="color: #38BDF8; margin-top: 0;">🔴 Live Camera Stream</h3>
            <p style="color: #94A3B8; font-size: 0.95rem;">Connect to local camera feed or cloud camera input for real-time edge AI object recognition and tracking.</p>
        </div>
    """, unsafe_allow_html=True)
    
    stream_mode = st.radio("Stream Mode", ["📸 Snapshot Capture Mode (Recommended for Cloud)", "⚡ Real-Time WebRTC Stream", "🟢 Local Webcam (Localhost Only)"], horizontal=True)
    
    if stream_mode == "📸 Snapshot Capture Mode (Recommended for Cloud)":
        st.info("💡 **Snapshot Mode:** Click **'Take Photo'** below to instantly capture and run YOLO neural detection on your bottle. Works 100% reliably on Streamlit Cloud!")
        cam_image = st.camera_input("Take a snapshot with your device camera")
        if cam_image is not None:
            image = Image.open(cam_image)
            img_np = np.array(image)
            
            with st.spinner("⚡ Running Neural Inference..."):
                results = model(img_np, conf=conf_threshold, iou=iou_threshold)
                res_plotted = results[0].plot()
                boxes = results[0].boxes
                count = len(boxes) if boxes is not None else 0
                
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            st.image(res_rgb, use_container_width=True)
            st.success(f"🎯 Detection Complete! Found {count} objects.")
    elif stream_mode == "⚡ Real-Time WebRTC Stream":
        if not HAS_WEBRTC:
            st.warning("⚠️ `streamlit-webrtc` is not installed.")
        else:
            st.info("ℹ️ WebRTC streaming requires P2P network traversal. If it hangs, please use **Snapshot Capture Mode** above.")
            class BottleVideoTransformer:
                def __init__(self, model, conf, iou):
                    self.model = model
                    self.conf = conf
                    self.iou = iou

                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    img = frame.to_ndarray(format="bgr24")
                    results = self.model(img, conf=self.conf, iou=self.iou, verbose=False)
                    res_plotted = results[0].plot()
                    return av.VideoFrame.from_ndarray(res_plotted, format="bgr24")

            RTC_CONFIGURATION = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )

            webrtc_streamer(
                key="bottle-detection-webrtc",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_transformer_factory=lambda: BottleVideoTransformer(model, conf_threshold, iou_threshold),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
    else:
        run_cam = st.toggle("🟢 Start Local Webcam", value=False)
        if run_cam:
            cam_placeholder = st.empty()
            stats_placeholder = st.empty()
            
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("⚠️ Local webcam (index 0) is not available in cloud hosting environments (like Streamlit Cloud).")
            else:
                while run_cam and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Unable to access local camera device.")
                        break
                    
                    start_t = time.time()
                    results = model.track(frame, persist=True, conf=conf_threshold, iou=iou_threshold, verbose=False)
                    fps_infer = 1.0 / max(time.time() - start_t, 1e-5)

                    res_plotted = results[0].plot()
                    boxes = results[0].boxes
                    current_count = len(boxes) if boxes is not None else 0

                    frame_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                    cam_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    stats_placeholder.markdown(f"⚡ **Real-time FPS:** {fps_infer:.1f} | 🎯 **Active Detections:** {current_count}")
                    
                cap.release()
