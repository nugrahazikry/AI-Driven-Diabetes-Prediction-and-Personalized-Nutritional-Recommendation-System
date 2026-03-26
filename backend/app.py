"""
app.py — Healthkaton Flask REST API backend.

Exposes only /api/* endpoints. Static files and the HTML frontend
are served separately by the nginx container.

Run locally :  python app.py
Run via Docker:  gunicorn --bind 0.0.0.0:5000 app:app
"""

import sys
import math
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import pandas as pd

from configuration import constants
from utils.data_cleaning import data_cleaning_food_dataset
from pipelines.diabetes_prediction import (
    calculate_bmr,
    calculate_daily_calories,
    bmi_calculator,
    diabetes_advice_prompt_process,
)
from pipelines.food_recommendation import generate_nutrisi
from pipelines.image_analysis import image_ocr_nutrition, image_analysis_composition

# ── Load heavy resources once at startup ────────────────────────────────────
_ml_model  = constants.model
_food_data = data_cleaning_food_dataset(
    str(ROOT_DIR / "data" / "dataset" / "food_calories_dataset.csv")
)

# ── Flask app (API only — no static / template serving) ─────────────────────
app = Flask(__name__)


# ── Health check ─────────────────────────────────────────────────────────────
@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok"})


# ── API: Diabetes Prediction ─────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json(force=True)

    glucose        = float(body.get("glucose", 0))
    blood_pressure = float(body.get("bloodPressure", 0))
    age            = float(body.get("age", 0))
    weight         = float(body.get("weight", 0))
    height_cm      = float(body.get("height", 0))
    gender         = body.get("gender", "Male")
    activity       = body.get("activity", "Sedentary (No exercise)")

    if height_cm <= 0 or weight <= 0:
        return jsonify({"error": "Height and Weight must be greater than 0."}), 400

    height_m = height_cm / 100.0

    bmi_string, bmi_category, bmi_color = bmi_calculator(weight, height_m)
    bmi_val = round(weight / (height_m ** 2), 2)

    bmr_raw = calculate_bmr(weight, height_m, age, gender)
    daily   = int(calculate_daily_calories(bmr_raw, activity))
    bmr1    = int(0.35 * daily)
    bmr2    = int(0.40 * daily)
    bmr3    = int(0.25 * daily)

    feature_df = pd.DataFrame(
        [[glucose, blood_pressure, bmi_val, age]],
        columns=["Glucose", "BloodPressure", "BMI", "Age"],
    )
    prediction  = int(_ml_model.predict(feature_df)[0])
    is_diabetic = prediction == 1
    kondisi     = "Patient has diabetes" if is_diabetic else "Patient does not have diabetes"

    data_kesehatan = (
        f"glucose level: {glucose} mg/dL\n"
        f"diastolic blood pressure: {blood_pressure} mmHg\n"
        f"BMI: {bmi_val} kg/m²\n"
        f"BMR: {daily} calories per day\n"
        f"age: {int(age)} years old\n"
        f"gender: {gender}\n"
        f"daily activity: {activity}\n"
        f"diabetes status: {kondisi}\n"
    )

    ai_result = diabetes_advice_prompt_process(data_kesehatan)
    health_rows, advice_lines, conclusion = _parse_prediction_advice(ai_result, {
        "glucose": glucose, "blood_pressure": blood_pressure,
        "bmi_val": bmi_val, "daily": daily, "age": age,
        "gender": gender, "activity": activity,
        "is_diabetic": is_diabetic, "bmi_category": bmi_category,
    })

    return jsonify({
        "isDiabetic":  is_diabetic,
        "bmi":         bmi_val,
        "bmiString":   bmi_string,
        "bmiCategory": bmi_category,
        "bmiColor":    bmi_color,
        "daily":       daily,
        "bmr1":        bmr1,
        "bmr2":        bmr2,
        "bmr3":        bmr3,
        "glucose":     glucose,
        "healthRows":  health_rows,
        "adviceLines": advice_lines,
        "conclusion":  conclusion,
        "dataString":  data_kesehatan,
    })


def _parse_prediction_advice(ai_result, fallback):
    """Parse AI advice text into (health_rows, advice_lines, conclusion)."""
    response_text = ai_result[0] if isinstance(ai_result, tuple) else ai_result

    if response_text:
        try:
            sections = response_text.split("\n\n")
            health_section = sections[0].split("Health data information:")[1]
            health_items   = [l.strip().lstrip("- ") for l in health_section.split("\n") if l.strip() and l.strip() != "-"]
            health_rows    = [item.split(": ", 1) for item in health_items if ": " in item]

            advice_section = sections[1].split("Healthy lifestyle guidelines:")[1]
            advice_lines   = [l.strip().lstrip("- ") for l in advice_section.split("\n") if l.strip() and l.strip() != "-"]

            concl_section  = sections[2].split("Conclusion:")[1]
            conclusion     = concl_section.strip()
            return health_rows, advice_lines, conclusion
        except Exception:
            pass

    g      = fallback
    status = "has diabetes risk" if g["is_diabetic"] else "does not have diabetes risk"

    glucose_note = ("elevated — indicates potential insulin resistance" if g["glucose"] > 140
                    else "borderline — requires monitoring"              if g["glucose"] > 100
                    else "within normal range")
    bp_note      = ("elevated — increases cardiovascular risk" if g["blood_pressure"] > 90
                    else "borderline high"                      if g["blood_pressure"] > 80
                    else "within normal diastolic range")
    bmi_note     = (f"{g['bmi_category']} — significantly increases diabetes risk" if g["bmi_val"] >= 30
                    else f"{g['bmi_category']} — moderately increases diabetes risk" if g["bmi_val"] >= 25
                    else f"{g['bmi_category']} — healthy body weight")

    health_rows = [
        ["Glucose Level",            f"{g['glucose']} mg/dL — {glucose_note}"],
        ["Diastolic Blood Pressure", f"{g['blood_pressure']} mmHg — {bp_note}"],
        ["BMI",                      f"{g['bmi_val']} kg/m² — {bmi_note}"],
        ["Daily Calories (BMR)",     f"{g['daily']} kcal/day based on {g['activity']}"],
        ["Age",                      f"{int(g['age'])} years old"],
        ["Gender",                   g["gender"]],
        ["Diabetes Status",          f"Patient {status}"],
    ]
    advice = []
    if g["glucose"] > 140:       advice.append("Reduce intake of refined sugars and high-glycemic foods.")
    if g["glucose"] > 100:       advice.append("Monitor blood sugar regularly and consider a low-glycemic diet.")
    if g["blood_pressure"] > 90: advice.append("Reduce sodium intake and practice stress-management techniques.")
    if g["bmi_val"] >= 30:       advice.append("Aim for gradual 5–10% body weight reduction through diet and exercise.")
    elif g["bmi_val"] >= 25:     advice.append("Include 30 minutes of moderate aerobic activity at least 5 days per week.")
    if g["age"] > 45:            advice.append("Schedule regular health screenings every 6–12 months.")
    advice.append("Stay well-hydrated — aim for 8 glasses (2 litres) of water per day.")
    advice.append("Prioritize 7–9 hours of quality sleep each night.")

    conclusion = (
        "Based on your health data, you show indicators associated with diabetes risk. "
        "Please consult a healthcare professional for a comprehensive clinical assessment."
        if g["is_diabetic"] else
        "Based on your health data, your current indicators do not suggest active diabetes. "
        "Continue maintaining a healthy lifestyle to preserve this status."
    )
    return health_rows, advice, conclusion


# ── API: Food Recommendation ─────────────────────────────────────────────────
@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    body  = request.get_json(force=True)
    daily = float(body.get("daily", 2000))
    recommendations = {"Breakfast": 0.35, "Lunch": 0.40, "Dinner": 0.25}

    try:
        results = generate_nutrisi(_food_data, recommendations, daily)

        def _clean(records):
            return [
                {k: (None if isinstance(v, float) and math.isnan(v) else v)
                 for k, v in r.items()}
                for r in records
            ]

        return jsonify({
            "breakfast": _clean(results[0]) if results and len(results) > 0 else [],
            "lunch":     _clean(results[1]) if results and len(results) > 1 else [],
            "dinner":    _clean(results[2]) if results and len(results) > 2 else [],
        })
    except Exception as e:
        return jsonify({"error": str(e), "breakfast": [], "lunch": [], "dinner": []}), 500


# ── API: Image Analysis ──────────────────────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    image_file  = request.files["image"]
    health_info = request.form.get("healthData", "")

    try:
        image, composition = image_ocr_nutrition(image_file)
        analysis = image_analysis_composition(composition, health_info)
        return jsonify({
            "composition": composition,
            "analysis":    analysis or "Analysis unavailable.",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Food Analysis ───────────────────────────────────────────────────────
@app.route("/api/food-analysis", methods=["POST"])
def api_food_analysis():
    body        = request.get_json(force=True)
    health_data = body.get("healthData", "")

    def food_to_str(food):
        if not food:
            return "N/A"
        return (
            f"Makanan: {food.get('name', 'Unknown')}, "
            f"Kalori: {food.get('calories', 0)} kcal, "
            f"Lemak: {food.get('fat', 0)}g, "
            f"Lemak Jenuh: {food.get('satFat', 0)}g, "
            f"Kolesterol: {food.get('chol', 0)}mg, "
            f"Sodium: {food.get('sodium', 0)}mg, "
            f"Karbohidrat: {food.get('carbs', 0)}g, "
            f"Serat: {food.get('fiber', 0)}g, "
            f"Gula: {food.get('sugar', 0)}g, "
            f"Protein: {food.get('protein', 0)}g"
        )

    breakfast_str = food_to_str(body.get("breakfast"))
    lunch_str     = food_to_str(body.get("lunch"))
    dinner_str    = food_to_str(body.get("dinner"))

    try:
        from pipelines.food_recommendation import food_recommendation_prompt_process
        sections = food_recommendation_prompt_process(
            health_data, breakfast_str, lunch_str, dinner_str
        )
        if sections:
            return jsonify({"sections": sections, "ok": True})
        return jsonify({"sections": [], "ok": False})
    except Exception as e:
        return jsonify({"error": str(e), "sections": [], "ok": False}), 500


# ── API: Sample Image ─────────────────────────────────────────────────────────
@app.route("/api/sample-image")
def api_sample_image():
    sample_dir  = ROOT_DIR / "data" / "analysis_input"
    sample_file = "Food picture composition for analysis.png"
    return send_from_directory(str(sample_dir), sample_file, mimetype="image/png")


# ── Run (development only) ────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
