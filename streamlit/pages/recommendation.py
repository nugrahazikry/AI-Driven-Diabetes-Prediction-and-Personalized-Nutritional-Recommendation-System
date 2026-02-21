"""
Page 2 — Food Recommendation

Shows nutrition data, lets user pick meals, and displays
AI-generated nutritional analysis per meal.
"""

import streamlit as st
import pandas as pd
import math
import time

from pipelines.food_recommendation import (
    generate_nutrisi,
    food_recommendation_prompt_process,
)
from utils.helpers.common import switch_page
import utils.styling.css as streamlit_css


# ── Nutrition value keys used throughout ─────────────────────────────────────
_NUTRITION_KEYS = [
    "Kalori", "lemak", "lemakJenuh", "kolesterol", "sodium",
    "karbohidrat", "serat", "gula", "protein",
]


def render():
    """Render the Food Recommendation page."""

    ss = st.session_state
    data = ss._food_data  # loaded once in app.py

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="background-color: #ffffff; padding: 0px; margin-top: 20px; '
        'text-align: center; border-radius: 5px;">'
        "<h1 style='margin: 0; color: black; text-align: center; border: 2px solid red;'>"
        "<strong>🔍🍔 Food Recommendations for You!</strong></h1></div>",
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("<h3>Your health data:</h3>", unsafe_allow_html=True)

    # ── Health summary cards ─────────────────────────────────────────────────
    _show_health_cards(ss)

    # ── Generate meal recommendations ────────────────────────────────────────
    ss.recommendations = {"Breakfast": 0.35, "Lunch": 0.4, "Dinner": 0.25}
    diabet_status = ss.prediction
    data_part = data.copy()

    if diabet_status == 1:
        data_part = data_part[data_part.diabetic_friendly == 1]

    if "recommended_nutrition" not in ss:
        ss.recommended_nutrition = generate_nutrisi(data_part, ss.recommendations, ss.bmr)
        ss.gender = ss.Jenis_Kelamin
    elif ss.gender != ss.Jenis_Kelamin:
        ss.recommended_nutrition = generate_nutrisi(data_part, ss.recommendations, ss.bmr)
        ss.gender = ss.Jenis_Kelamin

    # ── Display recommended recipes ──────────────────────────────────────────
    st.write("")
    st.subheader("Recommended food list:")

    col1d, col2d, col3d = st.columns(3)
    for cold, pecahin, part_nutri in zip(
        [col1d, col2d, col3d], ss.recommendations, ss.recommended_nutrition
    ):
        with cold:
            st.markdown(f"#### {pecahin} Menu")
            for recipe in part_nutri:
                with st.expander(recipe["makanan"]):
                    st.markdown(
                        '<h5 style="text-align: center; font-family: sans-serif;">'
                        "Nutritional Values (g):</h5>",
                        unsafe_allow_html=True,
                    )
                    nutri_df = pd.DataFrame(
                        {v: [recipe[v]] for v in _NUTRITION_KEYS}
                    ).T.rename(columns={0: "Composition"})
                    st.dataframe(nutri_df)

    # ── User meal selection ──────────────────────────────────────────────────
    st.subheader("Choose your meal:")

    breakfast_col, lunch_col, dinner_col = st.columns(3)
    with breakfast_col:
        breakfast_choice = st.selectbox(
            "Choose your breakfast:",
            [r["makanan"] for r in ss.recommended_nutrition[0]],
        )
    with lunch_col:
        lunch_choice = st.selectbox(
            "Choose your lunch:",
            [r["makanan"] for r in ss.recommended_nutrition[1]],
        )
    with dinner_col:
        dinner_choice = st.selectbox(
            "Choose your dinner:",
            [r["makanan"] for r in ss.recommended_nutrition[2]],
        )

    choices = [breakfast_choice, lunch_choice, dinner_choice]

    # ── Aggregate nutrition values ───────────────────────────────────────────
    total_nutrition = {k: 0 for k in _NUTRITION_KEYS}
    sarapan_string = siang_string = malam_string = ""

    for choice, meals_ in zip(choices, ss.recommended_nutrition):
        for meal in meals_:
            if meal["makanan"] == choice:
                for k in _NUTRITION_KEYS:
                    total_nutrition[k] += meal[k]

                if choice == breakfast_choice:
                    meal["meal time"] = "breakfast"
                    sarapan_string = ", ".join(f"{k}: {v}" for k, v in meal.items())
                elif choice == lunch_choice:
                    meal["meal time"] = "lunch"
                    siang_string = ", ".join(f"{k}: {v}" for k, v in meal.items())
                elif choice == dinner_choice:
                    meal["meal time"] = "dinner"
                    malam_string = ", ".join(f"{k}: {v}" for k, v in meal.items())

    # ── Analyse button ───────────────────────────────────────────────────────
    with st.container():
        if st.button("Check nutrition information"):
            with st.spinner("🔄 Please wait, analyzing your food..."):
                time.sleep(5)
                lines_food = food_recommendation_prompt_process(
                    ss.data_kesehatan, sarapan_string, siang_string, malam_string
                )
                if not lines_food:
                    st.error("AI analysis is currently unavailable. Please try again later.")
                else:
                    _parse_food_response(lines_food)

    # ── Show analysis results ────────────────────────────────────────────────
    if ss.reasons_markdown_conclusion_food:
        _show_food_results()

    # ── Navigation ───────────────────────────────────────────────────────────
    st.markdown("<h3>Choose page:</h3>", unsafe_allow_html=True)
    st.markdown(streamlit_css.choose_page_option, unsafe_allow_html=True)
    st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to health page"):
            switch_page("Prediction")
    with col2:
        if st.button("Analyze product photo"):
            switch_page("Upload File")


# ── Private helpers ──────────────────────────────────────────────────────────


def _show_health_cards(ss):
    """Render the calorie / glucose summary cards."""

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


def _parse_meal(output_text: str, meal_label: str):
    """Parse a single meal block into a dict and DataFrame."""
    lines = output_text.split("\n")
    name = lines[0].split(f"{meal_label} menu: ")[1]
    pola = lines[1].split("- Eating pattern: ")[1]
    gula = lines[2].split("- Sugar content: ")[1]
    kalori = lines[3].split("- Daily calories: ")[1]
    nutrisi = lines[4].split("- Nutrition: ")[1]

    df = pd.DataFrame({
        "Nutrition Info": ["Eating pattern", "Sugar content", "Daily calories", "Nutrition"],
        "Description": [pola, gula, kalori, nutrisi],
    })
    return name, df, df.to_html(index=False, classes="dataframe", border=0)


def _parse_food_response(lines_food):
    """Parse the AI food-recommendation response into session_state."""

    ss = st.session_state

    sarapan_out = lines_food[0].strip()
    lunch_out = lines_food[1].strip()
    dinner_out = lines_food[2].strip()
    conclusion_out = lines_food[3].strip()

    ss.makanan_sarapan, ss.df_sarapan, ss.df_sarapan_upload = _parse_meal(sarapan_out, "Breakfast")
    ss.makanan_lunch, ss.df_lunch, ss.df_lunch_upload = _parse_meal(lunch_out, "Lunch")
    ss.makanan_dinner, ss.df_dinner, ss.df_dinner_upload = _parse_meal(dinner_out, "Dinner")

    # Conclusion
    conclusion_text = conclusion_out.split("Conclusion:")[1]
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
    ss.reasons_markdown_conclusion_food = "\n".join(conclusion_list)


def _show_food_results():
    """Display meal analysis results."""

    ss = st.session_state

    st.subheader("Your meal nutrition information:")
    st.markdown(streamlit_css.responsive_table_styling, unsafe_allow_html=True)
    st.markdown(
        "<style>.streamlit-expanderHeader { font-size: 200px; font-weight: bold; color: #333; }</style>",
        unsafe_allow_html=True,
    )

    _meal_expander("🌅 Breakfast menu", ss.makanan_sarapan, ss.df_sarapan_upload)
    _meal_expander("🌤️ Lunch menu", ss.makanan_lunch, ss.df_lunch_upload)
    _meal_expander("🌙 Dinner menu", ss.makanan_dinner, ss.df_dinner_upload)

    st.markdown("<h3>Your food selection conclusion:</h3>", unsafe_allow_html=True)
    st.markdown(
        f"""<div style="background-color: #3cb371; color: white; padding: 5px;
             border-radius: 5px; text-align: center; margin-bottom: 20px;">
            <h3 style='margin: 0; color: white; font-size: 20px;'>
            {ss.reasons_markdown_conclusion_food}</h3></div>""",
        unsafe_allow_html=True,
    )


def _meal_expander(icon_label: str, meal_name: str, table_html: str):
    """Render a single meal expander with its nutrition table."""
    with st.expander(icon_label, expanded=True):
        st.markdown(
            f"<div style='background-color: #ffffff; padding: 10px; border-radius: 5px; "
            f"margin-bottom: 20px;'><h3 style='margin: 0;'>{icon_label}: {meal_name}</h3></div>",
            unsafe_allow_html=True,
        )
        st.markdown(table_html, unsafe_allow_html=True)
