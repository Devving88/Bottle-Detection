import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Water Bottle Detection",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define the pages
main_page = st.Page("pages/home.py", title="About", icon="📃")
bottle_OBJ_DETECT = st.Page("pages/test.py", title="Bottle-check", icon="📷")
# Set up navigation
pg = st.navigation([main_page, bottle_OBJ_DETECT])

# Run the selected page
pg.run()
