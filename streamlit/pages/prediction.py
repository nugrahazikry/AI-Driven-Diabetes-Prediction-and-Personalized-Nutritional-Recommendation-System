"""
Page 1 — Diabetes Prediction

Collects health inputs, runs the ML model, calculates BMI/BMR,
and shows AI-generated health advice.
"""

import streamlit as st
import numpy as np
import pandas as pd
import math
import time

from pipelines.diabetes_prediction import (
    calculate_bmr,
    calculate_daily_calories,
    bmi_calculator,
    diabetes_advice_prompt_process,
)
from utils.helpers.common import switch_page
import utils.styling.css as streamlit_css
import utils.styling.html as streamlit_html
from utils.styling.html import (
    patient_diabetes_condition_html,
    weight_result_html,
    calories_daily_html,
    bmi_info_html,
)


def render():
    """Render the Prediction page."""

    model = st.session_state._model  # loaded once in app.py

    st.markdown(streamlit_html.initial_page, unsafe_allow_html=True)
    st.subheader("Enter your health data")
    st.markdown(streamlit_css.initial_page, unsafe_allow_html=True)

    # ── Input form ───────────────────────────────────────────────────────────
    with st.container():
        row1_1, row1_2 = st.columns(2)

        with row1_1:
            st.session_state.Glucose = st.number_input(
                "Blood Sugar Level (mg/dL)", min_value=0, max_value=200, value=st.session_state.Glucose
            )
            st.session_state.BloodPressure = st.number_input(
                "Diastolic Blood Pressure (mmHg)", min_value=0, max_value=200, value=st.session_state.BloodPressure
            )
            st.session_state.Age = st.number_input(
                "Age", min_value=0, max_value=100, value=st.session_state.Age
            )

        with row1_2:
            st.session_state.Weight = st.number_input(
                "Weight (kg)", min_value=0.0, max_value=200.0, value=st.session_state.Weight
            )
            st.session_state.Height = st.number_input(
                "Height (cm)", min_value=0.0, max_value=250.0, value=st.session_state.Height
            )
            st.markdown(streamlit_css.select_box, unsafe_allow_html=True)
            st.session_state.Jenis_Kelamin = st.selectbox("Gender", ["Male", "Female"])

        st.markdown("<h4>Your activity plan:</h4>", unsafe_allow_html=True)
        st.session_state.aktivitas = st.select_slider(
            "Activity level",
            options=[
                "Sedentary (No exercise)",
                "Light (exercise 1-2 times per week)",
                "Moderate (exercise 3-4 times per week)",
                "Active (exercise 3-5 times per week)",
                "Very Active (exercise 6-7 times per week)",
                "Intense",
            ],
            label_visibility="collapsed",
        )

    # ── Predict button ───────────────────────────────────────────────────────
    with st.container():
        if st.button("Predict your health data"):
            with st.spinner("🔄 Please wait, your health data is being processed..."):
                time.sleep(3)
                if st.session_state.Height > 0:
                    _run_prediction(model)

    # ── Display results ──────────────────────────────────────────────────────
    if st.session_state.reasons_markdown_conclusion:
        _show_results()


# ── Private helpers ──────────────────────────────────────────────────────────


def _run_prediction(model):
    """Run ML prediction and AI advice, storing everything in session_state."""

    ss = st.session_state

    # BMI
    ss.BMI = round(ss.Weight / (ss.Height / 100) ** 2, 2)

    # Feature vector — use a DataFrame with the same feature names the model was trained with
    data = pd.DataFrame(
        [[ss.Glucose, ss.BloodPressure, ss.BMI, ss.Age]],
        columns=["Glucose", "BloodPressure", "BMI", "Age"],
    )

    # BMR / BMI helpers
    bmr_raw = calculate_bmr(ss.Weight, ss.Height / 100, ss.Age, ss.Jenis_Kelamin)
    ss.bmr = int(calculate_daily_calories(bmr_raw, ss.aktivitas))
    ss.bmr_light = int(calculate_daily_calories(bmr_raw, "Sedentary (No exercise)"))
    ss.bmi_string, ss.category, ss.color = bmi_calculator(ss.Weight, ss.Height / 100)
    ss.bmr_1 = int(0.35 * ss.bmr)
    ss.bmr_2 = int(0.40 * ss.bmr)
    ss.bmr_3 = int(0.25 * ss.bmr)
    ss.bmr_string = f"{ss.bmr} Calories/day"

    # ML prediction
    ss.prediction = model.predict(data)
    if ss.prediction == 1:
        ss.kondisi_pasien = "Patient has diabetes"
        ss.warna_kotak = "#ff2b47"
    else:
        ss.kondisi_pasien = "Patient does not have diabetes"
        ss.warna_kotak = "#3cb371"

    # Health-data string for AI prompts
    ss.data_kesehatan = (
        f"glucose level: {ss.Glucose} mg/dL\n"
        f"diastolic blood pressure: {ss.BloodPressure} mmHg\n"
        f"BMI: {ss.BMI} kg/m²\n"
        f"BMR: {ss.bmr} calories per day\n"
        f"age: {ss.Age} years old\n"
        f"gender: {ss.Jenis_Kelamin}\n"
        f"daily activity: {ss.aktivitas}\n"
        f"diabetes status: {ss.kondisi_pasien}\n"
    )

    # AI advice
    response_text = diabetes_advice_prompt_process(ss.data_kesehatan)
    if not response_text:
        st.error("AI advice is currently unavailable. Please try again later.")
        ss.reasons_markdown_conclusion = None
        return
    _parse_advice_response(response_text)


def _parse_advice_response(response_text: str):
    """Parse the AI response into session_state variables."""

    ss = st.session_state
    lines = response_text.split("\n\n")

    health_info_output = lines[0].strip()
    advices_output = lines[1].strip()
    conclusion_output = lines[2].strip()

    # Health-info table
    health_info = health_info_output.split("Health data information:")[1]
    health_info_list = health_info.split("\n            - ")
    health_info_list = [
        item.strip()
        for item in health_info_list
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
    reasons_markdown_health = "\n".join(health_info_list)

    reasons_list = [
        item.strip()[2:] if item.startswith("- ") else item.strip()
        for item in reasons_markdown_health.split("\n")
        if item.strip()
        and not (
            item.strip().lower() == "nan"
            or (
                math.isnan(float(item.split(":")[1].strip()))
                if item.split(":")[1].strip().replace(".", "", 1).isdigit()
                else False
            )
        )
    ]
    data_rows = [reason.split(": ", 1) for reason in reasons_list]
    df = pd.DataFrame(data_rows, columns=["Health Data", "Description"])
    ss.table_html = df.to_html(index=False, classes="dataframe", border=0)

    # Advice
    advices = advices_output.split("Healthy lifestyle guidelines:")[1]
    advice_list = advices.split("\n            - ")
    advice_list = [
        item.strip()
        for item in advice_list
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
    ss.reasons_markdown_advice = "\n".join(advice_list)

    # Conclusion
    conclusion = conclusion_output.split("Conclusion:")[1]
    conclusion_list = conclusion.split("\n            - ")
    conclusion_list = [
        item.strip()
        for item in conclusion_list
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
    ss.reasons_markdown_conclusion = "\n".join(conclusion_list)


def _show_results():
    """Display prediction results and AI advice."""

    ss = st.session_state

    st.markdown(
        patient_diabetes_condition_html(ss.warna_kotak, ss.kondisi_pasien),
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown("<h3>Your health information:</h3>", unsafe_allow_html=True)

    # BMI & BMR cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(weight_result_html(ss.bmi_string, "Body Mass Index (BMI)"), unsafe_allow_html=True)
    with col2:
        st.markdown(weight_result_html(ss.bmr_string, "Daily Calories (BMR)"), unsafe_allow_html=True)

    col1c, col2c, col3c = st.columns(3)
    with col1c:
        st.markdown(calories_daily_html(ss.bmr_1, "Breakfast (Calories)"), unsafe_allow_html=True)
    with col2c:
        st.markdown(calories_daily_html(ss.bmr_2, "Lunch (Calories)"), unsafe_allow_html=True)
    with col3c:
        st.markdown(calories_daily_html(ss.bmr_3, "Dinner (Calories)"), unsafe_allow_html=True)

    st.markdown(bmi_info_html(ss.color, ss.category), unsafe_allow_html=True)
    st.write("")
    st.markdown("<h3>Your health details:</h3>", unsafe_allow_html=True)

    st.markdown(streamlit_css.responsive_table_styling, unsafe_allow_html=True)
    st.markdown(ss.table_html, unsafe_allow_html=True)

    st.markdown("<h3>Healthy lifestyle recommendations:</h3>", unsafe_allow_html=True)
    st.markdown(ss.reasons_markdown_advice, unsafe_allow_html=True)

    st.markdown("<h3>Your health conclusion:</h3>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="background-color: {ss.warna_kotak}; color: white; padding: 5px;
             border-radius: 5px; text-align: center; margin-bottom: 20px;">
            <h3 style='margin: 0; color: white; font-size: 20px;'>{ss.reasons_markdown_conclusion}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    ss.kesehatan_info_list = ss.data_kesehatan

    # ── Navigation buttons ───────────────────────────────────────────────────
    st.markdown("<h3>Choose page:</h3>", unsafe_allow_html=True)
    st.markdown(streamlit_css.choose_page_option, unsafe_allow_html=True)
    st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Food recommendation"):
            switch_page("Food recommendation")
    with col2:
        if st.button("Analyze product photo"):
            switch_page("Upload File")
