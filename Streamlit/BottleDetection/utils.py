import streamlit as st
import os
import glob
from ultralytics import YOLO

def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

        /* Global Minimalist Luxury Theme */
        .stApp {
            background: #06080F;
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: #E2E8F0;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', sans-serif !important;
            letter-spacing: -0.5px;
        }

        /* Animated Header */
        @keyframes fadeInSlide {
            0% { opacity: 0; transform: translateY(-15px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        .minimal-title {
            font-family: 'Outfit', sans-serif;
            font-weight: 800;
            font-size: 3rem;
            background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: fadeInSlide 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
            margin-bottom: 0.2rem;
        }

        .minimal-subtitle {
            color: #64748B;
            font-size: 1.05rem;
            font-weight: 400;
            animation: fadeInSlide 1s cubic-bezier(0.16, 1, 0.3, 1) forwards;
            margin-bottom: 2.5rem;
        }

        /* Floating Glass Cards with Smooth Hover Animations */
        @keyframes floatCard {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-6px); }
            100% { transform: translateY(0px); }
        }

        .glass-card {
            background: rgba(15, 23, 42, 0.6);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 135, 0.08);
            border-radius: 20px;
            padding: 28px;
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
            transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            margin-bottom: 20px;
            position: relative;
            overflow: hidden;
        }

        .glass-card:hover {
            border-color: rgba(56, 189, 248, 0.4);
            box-shadow: 0 30px 60px -20px rgba(56, 189, 248, 0.15);
            transform: translateY(-4px);
        }

        /* Modern File Uploader Dropzone */
        [data-testid="stFileUploader"] {
            background: rgba(15, 23, 42, 0.5);
            border: 2px dashed rgba(56, 189, 248, 0.3);
            border-radius: 16px;
            padding: 24px;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        }

        [data-testid="stFileUploader"]:hover {
            border-color: #38BDF8;
            background: rgba(56, 189, 248, 0.05);
            box-shadow: 0 10px 30px -10px rgba(56, 189, 248, 0.3);
        }

        /* Modern Selectbox & Inputs */
        [data-baseweb="select"] > div {
            background-color: rgba(15, 23, 42, 0.8) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
            color: #FFFFFF !important;
            transition: all 0.3s ease;
        }

        [data-baseweb="select"] > div:hover {
            border-color: #38BDF8 !important;
            box-shadow: 0 0 15px rgba(56, 189, 248, 0.2);
        }

        /* Pulsing Glow Metric Cards */
        @keyframes pulseGlow {
            0% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.2); }
            70% { box-shadow: 0 0 0 15px rgba(56, 189, 248, 0); }
            100% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }
        }

        .metric-container {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            transition: all 0.3s ease;
            animation: pulseGlow 4s infinite;
        }

        .metric-container:hover {
            border-color: #38BDF8;
            transform: scale(1.02);
        }

        .metric-value {
            font-family: 'Outfit', sans-serif;
            font-size: 2.4rem;
            font-weight: 700;
            color: #38BDF8;
            margin-bottom: 4px;
        }

        .metric-label {
            font-size: 0.85rem;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 600;
        }

        /* Sleek Minimalist Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #38BDF8 100%, #2563EB 0%);
            color: #FFFFFF;
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            letter-spacing: 0.5px;
            border: none;
            border-radius: 12px;
            padding: 12px 28px;
            box-shadow: 0 10px 25px -5px rgba(56, 189, 248, 0.4);
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            width: 100%;
        }

        .stButton>button:hover {
            background: linear-gradient(135deg, #2563EB 0%, #38BDF8 100%);
            box-shadow: 0 15px 30px -5px rgba(56, 189, 248, 0.6);
            transform: translateY(-2px);
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background: #090D16;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* Tabs Styling Enhancement */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            background-color: rgba(15, 23, 42, 0.7);
            padding: 10px;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }

        .stTabs [data-baseweb="tab"] {
            background-color: transparent !important;
            border-radius: 12px !important;
            color: #94A3B8 !important;
            font-family: 'Outfit', sans-serif !important;
            padding: 12px 24px !important;
            font-weight: 600 !important;
            border: none !important;
            box-shadow: none !important;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: #FFFFFF !important;
            background-color: rgba(56, 189, 248, 0.1) !important;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #38BDF8 0%, #2563EB 100%) !important;
            color: #FFFFFF !important;
            box-shadow: 0 8px 25px -5px rgba(56, 189, 248, 0.5) !important;
        }

        /* Hide Streamlit default tab highlight underline / border lines */
        .stTabs [data-baseweb="tab-highlight"], 
        .stTabs div[data-baseweb="tab-border"] {
            display: none !important;
        }
    </style>
    """,old_string: unsafe_allow_html=True)

@st.cache_resource(show_spinner="⚡ Loading Neural Model Weights...")
def load_selected_model(model_path):
    if not os.path.exists(model_path):
        model_path = "model/best.pt"
    return YOLO(model_path)

def get_available_models():
    """Discover all .pt files across workspace for YOLOv8/26 selection."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    pt_pattern = os.path.join(root_dir, "**", "*.pt")
    found_files = glob.glob(pt_pattern, recursive=True)
    
    models = {}
    for fpath in found_files:
        rel_path = os.path.relpath(fpath, os.path.dirname(__file__))
        filename = os.path.basename(fpath)
        parent_dir = os.path.basename(os.path.dirname(fpath))
        
        # Categorize nice labels for YOLOv8, YOLOv12, YOLOv26
        if "yolov8" in filename.lower() or "best" in filename.lower():
            label = f"🚀 YOLOv8 Core -> {filename}"
        elif "yolo12" in filename.lower():
            label = f"⚡ YOLOv12 Engine -> {filename}"
        elif "yolo26" in filename.lower() or "ft" in filename.lower():
            label = f"✨ YOLOv26 Quantum -> {filename}"
        else:
            label = f"📦 Custom Model -> {filename}"
            
        models[label] = rel_path
        
    if not models:
        models["Default Model (best.pt)"] = "model/best.pt"
        
    return models
