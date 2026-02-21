"""
Page 3 — Upload File (Food-label image analysis)

Lets the user upload a food-product photo, runs OCR,
and provides AI-driven nutritional safety analysis.
"""

import streamlit as st
import pandas as pd
import math
import traceback
from PIL import UnidentifiedImageError

from pipelines.image_analysis import (
    image_ocr_nutrition,
    image_analysis_composition,
)
from utils.helpers.common import switch_page
import utils.styling.css as streamlit_css


def render():
    """Render the Upload File page."""

    ss = st.session_state

    st.markdown(
        "<h2 style='text-align: center; color: black; border: 2px solid red;'>"
        "<strong>📸🍕 Food Composition Photo Nutrition Analysis</strong></h2>",
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("<h3>Your health data:</h3>", unsafe_allow_html=True)

    # ── Health summary cards (reused pattern) ────────────────────────────────
    _show_health_cards(ss)

    # ── File uploader ────────────────────────────────────────────────────────
    st.write("")
    st.subheader("Upload Product Composition Photo")

    ss.uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )

    # ── Analyse button ───────────────────────────────────────────────────────
    with st.container():
        if st.button("Analyze product image"):
            try:
                with st.spinner("🔄 Please wait, processing photo and content..."):
                    _analyse_image(ss)
            except UnidentifiedImageError:
                st.error(
                    "The uploaded file is not a valid image. Please upload a valid JPG, JPEG, or PNG file. "
                    "Make sure the image contains nutritional composition of a food/beverage product."
                )
            except Exception as e:
                st.error("Make sure the image contains clear nutritional composition of a food/beverage product.")
                st.error(f"⚠️ failed:\n{''.join(traceback.format_exception(type(e), e, e.__traceback__))}")

    # ── Display results ──────────────────────────────────────────────────────
    if ss.reasons_markdown_conclusion_image_food is not None:
        try:
            _show_results(ss)
        except UnidentifiedImageError:
            st.error(
                "The uploaded file is not a valid image. Please upload a valid JPG, JPEG, or PNG file. "
                "Make sure the image contains nutritional composition of a food/beverage product."
            )
        except Exception as e:
            st.error("Make sure the image contains clear nutritional composition of a food/beverage product.")
            st.error(f"⚠️ failed:\n{''.join(traceback.format_exception(type(e), e, e.__traceback__))}")
    else:
        st.info("Upload an image to analyze nutritional composition.", icon="ℹ️")

    # ── Navigation ───────────────────────────────────────────────────────────
    st.markdown("<h3>Choose page:</h3>", unsafe_allow_html=True)
    st.markdown(streamlit_css.choose_page_option, unsafe_allow_html=True)
    st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to health page"):
            switch_page("Prediction")
    with col2:
        if st.button("Food recommendation"):
            switch_page("Food recommendation")


# ── Private helpers ──────────────────────────────────────────────────────────


def _show_health_cards(ss):
    """Render glucose / BMR summary cards."""

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"<div style='background-color: #cccccc; padding: 10px 10px 0px; "
            f"border-radius: 5px; margin-bottom: 20px; text-align: center;'>"
            f"<h6 style='margin: 0; font-weight: normal;'>Blood sugar level</h6>"
            f"<h2 style='margin: -30px 0 0 0;'>{ss.Glucose} mg/dL</h2></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div style='background-color: #cccccc; padding: 10px 10px 0px; "
            f"border-radius: 5px; margin-bottom: 20px; text-align: center;'>"
            f"<h6 style='margin: 0; font-weight: normal;'>Daily Calories (BMR)</h6>"
            f"<h2 style='margin: -30px 0 0 0;'>{ss.bmr} Calories/day</h2></div>",
            unsafe_allow_html=True,
        )

    col1c, col2c, col3c = st.columns(3)
    for col, label, val in zip(
        [col1c, col2c, col3c],
        ["Breakfast (Calories)", "Lunch (Calories)", "Dinner (Calories)"],
        [ss.bmr_1, ss.bmr_2, ss.bmr_3],
    ):
        with col:
            st.markdown(
                f"<div style='background-color: #cccccc; padding: 10px 10px 0px; "
                f"border-radius: 5px; text-align: center;'>"
                f"<h6 style='margin: 0; font-weight: normal;'>{label}</h6>"
                f"<h2 style='margin: -30px 0 0 0;'>{val} Calories</h2></div>",
                unsafe_allow_html=True,
            )


def _analyse_image(ss):
    """Run OCR + AI analysis on the uploaded image."""

    ss.image, ss.komposisi = image_ocr_nutrition(ss.uploaded_file)

    # Parse composition lines (validate OCR output to avoid IndexError)
    lines = [l.strip() for l in ss.komposisi.splitlines() if l.strip()]

    if not lines:
        raise ValueError("No text detected from OCR output.")

    # First line may be like 'Product: <name>' or just a product name
    first_line = lines[0]
    if ":" in first_line:
        parts = first_line.split(":", 1)
        ss.product_type = parts[1].strip() if len(parts) > 1 else parts[0].strip()
    else:
        ss.product_type = first_line.strip()

    data_upload = []
    for line in lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            data_upload.append({
                "Nutritional Content": key.split(". ", 1)[-1].strip(),
                "Unit": value.strip(),
            })
        else:
            # skip malformed lines
            continue

    ss.df_upload = pd.DataFrame(data_upload)
    ss.table_html_upload = ss.df_upload.to_html(index=False, classes="dataframe", border=0)

    # AI analysis
    ss.response_text = image_analysis_composition(ss.komposisi, ss.kesehatan_info_list)

    if not ss.response_text:
        raise ValueError("AI analysis is currently unavailable. Please try again later.")

    lines_pic = ss.response_text.split("\n\n")
    ss.safe_question = lines_pic[0].strip()
    reasons_raw = lines_pic[1].strip()
    nutrition_details = lines_pic[2].strip()
    conclusion_raw = lines_pic[3].strip()

    # Reasons
    reasons = reasons_raw.split("Detailed reasons:")[1]
    reasons_list = [
        item.strip()
        for item in reasons.split("\n            - ")
        if item.strip()
        and not (
            item.strip().lower() == "nan"
            or (
                math.isnan(float(item.strip()))
                if item.strip().replace(".", "", 1).isdigit()
                else False
            )
        )
    ]
    ss.reasons_markdown = "\n".join(reasons_list)

    # Nutrition detail
    detail_lines = nutrition_details.split("\n")
    ss.pola_food_upload = detail_lines[1].split("- Consumption pattern: ")[1]
    ss.kalori_food_upload = detail_lines[2].split("- Daily calories: ")[1]
    ss.nutrisi_food_upload = detail_lines[3].split("- Nutrition: ")[1]
    ss.saran_penyajian_food_upload = detail_lines[4].split("- Serving suggestion: ")[1]

    ss.df_food_upload_output = pd.DataFrame({
        "Nutrition Info": ["Consumption pattern", "Daily calories", "Nutrition", "Serving suggestion"],
        "Description": [
            ss.pola_food_upload,
            ss.kalori_food_upload,
            ss.nutrisi_food_upload,
            ss.saran_penyajian_food_upload,
        ],
    })

    # Conclusion
    conclusion_text = conclusion_raw.split("Conclusion:")[1]
    conclusion_list = [
        item.strip()
        for item in conclusion_text.split("\n            - ")
        if item.strip()
        and not (
            item.strip().lower() == "nan"
            or (
                math.isnan(float(item.strip()))
                if item.strip().replace(".", "", 1).isdigit()
                else False
            )
        )
    ]
    ss.reasons_markdown_conclusion_image_food = "\n".join(conclusion_list)


def _show_results(ss):
    """Display image analysis results."""

    st.image(ss.image, caption="Uploaded Image", use_column_width=True)

    st.markdown(
        f"<h2 style='border: 2px solid #ff9900; text-align: center;'>"
        f"🍔 Product: {ss.product_type}</h2>",
        unsafe_allow_html=True,
    )
    st.subheader("Nutritional Content:")
    st.markdown(streamlit_css.responsive_table_styling, unsafe_allow_html=True)
    st.markdown(ss.table_html_upload, unsafe_allow_html=True)

    st.subheader("Is the product recommended?")

    product_safe = ss.safe_question.split(": ")[1]
    warna = "#3cb371" if product_safe.lower() == "yes" else "#ff2b47"
    recommendation_label = "Product is recommended" if product_safe.lower() == "yes" else "Product is not recommended"

    st.markdown(
        f'<div style="background-color: {warna}; color: white; padding: 5px; '
        f'border-radius: 5px; text-align: center;">'
        f"<h3 style='margin: 0; color: white;'>{recommendation_label}</h3></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<h3>Detailed reasons:</h3>", unsafe_allow_html=True)
    st.markdown(ss.reasons_markdown, unsafe_allow_html=True)

    st.markdown("<h3>Nutrition information:</h3>", unsafe_allow_html=True)
    st.markdown(
        ss.df_food_upload_output.to_html(index=False, classes="dataframe", border=0),
        unsafe_allow_html=True,
    )

    st.markdown("<h3>Product information conclusion:</h3>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="background-color: {warna}; color: white; padding: 5px; '
        f'border-radius: 5px; text-align: center; margin-bottom: 20px;">'
        f"<h3 style='margin: 0; color: white; font-size: 20px;'>"
        f"{ss.reasons_markdown_conclusion_image_food}</h3></div>",
        unsafe_allow_html=True,
    )
