"""
app.py — Clean entry point for the Healthkaton Streamlit app.

Dispatches to individual page modules under streamlit/pages/.
Run with:  streamlit run app.py
"""

import sys
from pathlib import Path
import importlib.util
import streamlit as st

# ── Lazy-loaded resources (cached so they load only once) ────────────────────
from configuration import constants
from utils.data.cleaning import data_cleaning_food_dataset

# Ensure root directory is on sys.path so all package imports resolve
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ── Page config (must be the first Streamlit command) ────────────────────────
st.set_page_config(page_title="Diabetes & Health Analysis", layout="centered")

@st.cache_resource
def load_model():
    """Load the ML model once and share across reruns."""
    return constants.model


@st.cache_data
def load_food_dataset():
    """Load and clean the food-calorie dataset once."""
    return data_cleaning_food_dataset("data/dataset/food_calories_dataset.csv")


# Store shared resources in session_state so page modules can access them
# without re-importing heavy objects.
if "_model" not in st.session_state:
    st.session_state._model = load_model()

if "_food_data" not in st.session_state:
    st.session_state._food_data = load_food_dataset()


# ── Session-state defaults ───────────────────────────────────────────────────
_DEFAULTS = {
    "page": "Prediction",
    "bmr": None,
    "uploaded_file": None,
    "reasons_markdown_conclusion": None,
    "reasons_markdown_conclusion_food": None,
    "reasons_markdown_conclusion_image_food": None,
    "image": None,
    "jenis_kelamin": None,
    "Pregnancies": 0,
    "Glucose": 0,
    "BloodPressure": 0,
    "SkinThickness": 0,
    "Insulin": 0,
    "Weight": 0.0,
    "Height": 0.0,
    "DiabetesPedigreeFunction": 0,
    "Age": 0,
    "BMI": None,
    "prediction": None,
}

for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Page routing ─────────────────────────────────────────────────────────────
# NOTE: We can't use `from streamlit.pages.X import ...` because "streamlit"
# collides with the installed streamlit package.  Load by file path instead.


def _import_page(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT_DIR / filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.render

render_prediction   = _import_page("pages.prediction",     "streamlit/pages/prediction.py")
render_recommendation = _import_page("pages.recommendation", "streamlit/pages/recommendation.py")
render_upload       = _import_page("pages.upload",          "streamlit/pages/upload.py")

_PAGES = {
    "Prediction": render_prediction,
    "Food recommendation": render_recommendation,
    "Upload File": render_upload,
}

current_page = st.session_state.get("page", "Prediction")
page_fn = _PAGES.get(current_page, render_prediction)
page_fn()
