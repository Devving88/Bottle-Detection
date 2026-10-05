import streamlit as st
import os
import cv2
import av
import tempfile

from ultralytics import YOLO
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ===========================================================================
# MODEL SETTINGS
# ===========================================================================

MODEL_DIR = Path("model")

model_files = sorted(
    MODEL_DIR.glob("*.pt")
)

if not model_files:

    st.error(
        "ไม่พบไฟล์โมเดล .pt ในโฟลเดอร์ model/"
    )

    st.stop()


# ===========================================================================
# MAIN HEADER
# ===========================================================================

st.title(
    "💧 ระบบตรวจจับขวดน้ำด้วย YOLO"
)

st.caption(
    "ระบบตรวจจับและจำแนกประเภทขวดน้ำจากวิดีโอและเว็บแคม"
)

st.markdown("---")


col_model, col_info, col_f1 = st.columns(
    [2, 1, 1]
)

with col_model:

    st.markdown(
        "### 🤖 เลือกโมเดล"
    )

    selected_model = st.selectbox(
        "Model",
        model_files,
        format_func=lambda x: x.stem,
        label_visibility="collapsed",
    )


with col_info:

    st.markdown(
        "### 📦 โมเดลที่เลือก"
    )

    st.info(
        selected_model.name
    )


# ===========================================================================
# F1 CONFIDENCE STATISTICS
# ===========================================================================

F1_STATS = {
    "yolo8n": (0.93, 0.279),
    "yolo11n": (0.93, 0.577),
    "yolo12n": (0.93, 0.649),
    "yolo26n": (0.93, 0.674),
}


with col_f1:

    model_key = selected_model.stem.lower()

    if model_key in F1_STATS:

        f1, confidence = F1_STATS[model_key]

        st.markdown(
            "### 📊 F1 Confidence"
        )

        st.info(
            f"All classes {f1:.2f} at {confidence:.3f}"
        )



# ===========================================================================
# LOAD MODEL
# ===========================================================================

@st.cache_resource(
    show_spinner="กำลังโหลดโมเดล YOLO..."
)
def load_model(model_path):

    return YOLO(
        str(model_path)
    )


model = load_model(
    selected_model
)

class_names = model.model.names


# ===========================================================================
# CLASS NAME HELPER
# ===========================================================================

def get_class_name(class_id):

    if isinstance(class_names, dict):

        return class_names.get(
            class_id,
            str(class_id)
        )

    return class_names[class_id]


# ===========================================================================
# SIDEBAR
# ===========================================================================

with st.sidebar:

    st.markdown(
        "## 💧 Water Bottle Detection"
    )

    st.markdown(
        "ระบบตรวจจับขวดน้ำด้วย **YOLO**"
    )

    st.markdown("---")

    st.markdown(
        "### 🏷️ ประเภทขวดที่รองรับ"
    )

    if isinstance(class_names, dict):

        for cls_name in class_names.values():

            st.markdown(
                f"- {cls_name}"
            )

    else:

        for cls_name in class_names:

            st.markdown(
                f"- {cls_name}"
            )

    st.markdown("---")

    st.caption(
        f"โมเดล: `{selected_model.name}`"
    )

    st.caption(
        "ไฟล์อัปโหลดจะถูกเก็บไว้ที่โฟลเดอร์ `upload/`"
    )


# ===========================================================================
# UPLOAD DIRECTORY
# ===========================================================================

UPLOAD_DIR = "upload"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# ===========================================================================
# DETECTION SETTINGS
# ===========================================================================

BOX_COLOR = (
    216,
    138,
    61
)

LABEL_TEXT_COLOR = (
    11,
    15,
    25
)


# ===========================================================================
# DRAW DETECTIONS
# ===========================================================================

def draw_detections(
    frame,
    results
):

    if (
        not results
        or results[0].boxes is None
    ):

        return frame


    for box in results[0].boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        class_id = int(
            box.cls[0]
        )

        conf = float(
            box.conf[0]
        )

        class_name = get_class_name(
            class_id
        )


        text = f"{class_name} {conf:.2f}"


        if "bottle" in class_name.lower():

            color = (
                255,
                144,
                30
            )

            text_y = y1 + 20

        else:

            color = (
                0,
                255,
                0
            )

            text_y = max(
                y1 - 10,
                20
            )


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )


        (
            text_w,
            text_h
        ), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )


        cv2.rectangle(
            frame,
            (
                x1,
                text_y - text_h - 5
            ),
            (
                x1 + text_w,
                text_y + 5
            ),
            color,
            -1
        )


        cv2.putText(
            frame,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )


    return frame


# ===========================================================================
# WEBCAM VIDEO PROCESSOR
# ===========================================================================

class YOLOVideoProcessor(
    VideoProcessorBase
):

    def __init__(self):

        self.model = model

        self.conf = 0.25

        # Prevent multiple frames from being
        # processed at the same time.
        self.processing = False


    def recv(
        self,
        frame
    ):

        # ---------------------------------------------------------------
        # Browser laptop webcam -> OpenCV BGR
        # ---------------------------------------------------------------

        img = frame.to_ndarray(
            format="bgr24"
        )


        # ---------------------------------------------------------------
        # Prevent frame-processing overload
        # ---------------------------------------------------------------

        if self.processing:

            return av.VideoFrame.from_ndarray(
                img,
                format="bgr24"
            )


        self.processing = True


        try:

            # -----------------------------------------------------------
            # YOLO Tracking
            # -----------------------------------------------------------

            results = self.model.track(
                img,
                persist=True,
                conf=self.conf,
                imgsz=640,
                verbose=False,
            )


            # -----------------------------------------------------------
            # Draw detections
            # -----------------------------------------------------------

            img = draw_detections(
                img,
                results
            )

        finally:

            self.processing = False


        # ---------------------------------------------------------------
        # OpenCV BGR -> Browser WebRTC
        # ---------------------------------------------------------------

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# ===========================================================================
# TABS
# ===========================================================================

tab_video, tab_webcam = st.tabs(
    [
        "📹  อัปโหลดวิดีโอ",
        "🎥  เว็บแคมเรียลไทม์",
    ]
)


# ===========================================================================
# VIDEO TAB
# ===========================================================================

with tab_video:

    st.subheader(
        "อัปโหลดวิดีโอเพื่อตรวจจับขวดน้ำ"
    )


    col_upload, col_setting = st.columns(
        [2, 1]
    )


    # -----------------------------------------------------------------------
    # Upload
    # -----------------------------------------------------------------------

    with col_upload:

        uploaded_file = st.file_uploader(
            "เลือกไฟล์วิดีโอ (mp4, avi, mov, mkv)",
            type=[
                "mp4",
                "avi",
                "mov",
                "mkv",
            ],
        )


    # -----------------------------------------------------------------------
    # Settings
    # -----------------------------------------------------------------------

    with col_setting:

        conf_video = st.slider(
            "Confidence threshold",
            0.0,
            1.0,
            0.65,
            0.05,
            key="conf_video",
        )


        skip_frame = st.checkbox(
            "ข้ามเฟรมเพื่อเพิ่มความเร็ว",
            value=True,
            key="skip_video",
        )


    # -----------------------------------------------------------------------
    # Process uploaded video
    # -----------------------------------------------------------------------

    if uploaded_file is not None:

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:

            st.video(uploaded_file)


        start_video = st.button(
            "▶️  เริ่มประมวลผลวิดีโอ",
            key="start_video",
            use_container_width=True,
        )


        if start_video:

            with st.spinner(
                "กำลังประมวลผล... อาจใช้เวลาสักครู่ กรุณารอจนกว่าจะเสร็จสิ้น"
            ):

                tfile_in = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4",
                )

                tfile_in.write(
                    uploaded_file.read()
                )

                tfile_in.close()


                tfile_out = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".webm",
                )

                output_path = tfile_out.name

                tfile_out.close()


                cap = cv2.VideoCapture(
                    tfile_in.name
                )


                if not cap.isOpened():

                    st.error(
                        "ไม่สามารถเปิดไฟล์วิดีโอได้"
                    )

                    st.stop()


                orig_width = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_WIDTH
                    )
                )

                orig_height = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_HEIGHT
                    )
                )

                orig_fps = int(
                    cap.get(
                        cv2.CAP_PROP_FPS
                    )
                )

                if orig_fps <= 0:

                    orig_fps = 30.0


                total_frames = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_COUNT
                    )
                )

                if total_frames <= 0:

                    total_frames = 1


                target_width = 640

                target_height = int(
                    orig_height
                    * (
                        target_width
                        / orig_width
                    )
                )

                target_height = (
                    target_height // 2
                ) * 2


                skip_frames = (
                    2
                    if skip_frame
                    else 1
                )

                out_fps = max(
                    1,
                    int(
                        orig_fps
                        // skip_frames
                    ),
                )


                fourcc = cv2.VideoWriter_fourcc(
                    *'vp80'
                )

                out = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    out_fps,
                    (
                        target_width,
                        target_height,
                    ),
                )


                progress_bar = (
                    st.progress(0)
                )

                status_text = (
                    st.empty()
                )


                frame_count = 0

                processed_count = (
                    0
                )


                while cap.isOpened():

                    ret, frame = (
                        cap.read()
                    )

                    if not ret:

                        break


                    frame_count += (
                        1
                    )


                    if (
                        frame_count
                        % skip_frames
                        != 0
                    ):

                        continue


                    frame_resized = (
                        cv2.resize(
                            frame,
                            (
                                target_width,
                                target_height,
                            ),
                        )
                    )


                    results = (
                        model.track(
                            frame_resized,
                            persist=True,
                            conf=conf_video,
                            imgsz=640,
                            verbose=False,
                        )
                    )


                    res_plotted = (
                        draw_detections(
                            frame_resized.copy(),
                            results,
                        )
                    )


                    out.write(
                        res_plotted
                    )


                    processed_count += (
                        1
                    )


                    if (
                        total_frames
                        > 0
                    ):

                        progress = (
                            min(
                                frame_count
                                / total_frames,
                                1.0,
                            )
                        )

                        progress_bar.progress(
                            progress
                        )

                        status_text.text(
                            f"กำลังประมวลผล: {frame_count}/{total_frames} เฟรม ({(progress*100):.1f}%)"
                        )


                cap.release()

                out.release()


                status_text.text(
                    "ประมวลผลเสร็จสิ้น! กำลังเตรียมวิดีโอแสดงผล..."
                )


                with open(
                    output_path,
                    "rb",
                ) as video_file:

                    video_bytes = (
                        video_file.read()
                    )


                st.success("สำเร็จ!")

                col1, col2, col3 = st.columns([1, 2, 1])

                with col2:

                    st.video(
                        video_bytes,
                        format="video/webm",
                    )


                output_filename = f"output_{Path(uploaded_file.name).stem}.webm"


                st.download_button(
                    label="📥 ดาวน์โหลดวิดีโอผลลัพธ์",
                    data=video_bytes,
                    file_name=output_filename,
                    mime="video/webm",
                    use_container_width=True,
                )


                os.remove(
                    tfile_in.name
                )

                os.remove(
                    output_path
                )


# ===========================================================================
# WEBCAM TAB
# ===========================================================================

with tab_webcam:

    st.subheader(
        "ตรวจจับขวดน้ำแบบเรียลไทม์จากเว็บแคม"
    )


    st.markdown(
        """
        เว็บแคมส่วนนี้ใช้ **WebRTC** แทน `cv2.VideoCapture(0)`
        ดังนั้นกล้องจะเป็นกล้องของ Browser ที่กำลังเปิด Streamlit
        """
    )


    # -----------------------------------------------------------------------
    # Confidence
    # -----------------------------------------------------------------------

    conf_cam = st.slider(
        "Confidence threshold",
        0.0,
        1.0,
        0.25,
        0.05,
        key="conf_cam",
    )


    st.markdown("---")


    # =========================================================================
    # WEBRTC
    # =========================================================================

    webrtc_ctx = webrtc_streamer(
        key="water-bottle-webcam",
        video_processor_factory=YOLOVideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 30},
            },
            "audio": False,
        },
        async_processing=False,
    )


    # =========================================================================
    # UPDATE CONFIDENCE
    # =========================================================================

    if webrtc_ctx.video_processor:

        webrtc_ctx.video_processor.conf = conf_cam


    # =========================================================================
    # CAMERA STATUS
    # =========================================================================

    if webrtc_ctx.state.playing:

        st.success(
            "🟢 กล้องกำลังทำงาน"
        )

    else:

        st.info(
            "กดปุ่ม START ด้านบนเพื่อเปิดเว็บแคม "
            "และกด Allow เมื่อ Browser ขอสิทธิ์ใช้กล้อง"
        )


    st.caption(
        "หมายเหตุ: Browser ต้องได้รับอนุญาตให้เข้าถึงกล้อง "
        "และหากนำไปใช้งานออนไลน์ แนะนำให้เปิดผ่าน HTTPS"
    )
