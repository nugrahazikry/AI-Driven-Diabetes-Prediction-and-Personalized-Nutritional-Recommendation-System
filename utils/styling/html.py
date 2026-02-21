"""
HTML Templates Module
=====================
HTML template functions for Streamlit UI components.
"""

# Initial page header HTML
initial_page = """
<h2 style='margin: 0; color: black; text-align: center; border: 2px solid red;'>
    <strong>🏥💊 Predict your diabetes level & nutritional health!</strong>
</h2>
<div style="text-align: center; margin-bottom: 20px;">
    <br><strong>Created by:</strong>
    <a href="https://www.linkedin.com/in/nugrahazikry" target="_blank" style="text-decoration: none; color: blue;">Zikry Adjie Nugraha</a>, 
    <a href="https://www.linkedin.com/in/diki-rustian/" target="_blank" style="text-decoration: none; color: blue;">Diki Rustian</a>, 
    <a href="https://www.linkedin.com/in/mff/" target="_blank" style="text-decoration: none; color: blue;">Muhammad Fikri Fadillah</a>
</div>
"""


def patient_diabetes_condition_html(box_color, patient_condition):
    """Generate HTML for patient diabetes condition display."""
    return f"""
    <h3>Do you have diabetes?</h3>
    <div style="background-color: {box_color}; color: white; padding: 5px; border-radius: 5px; text-align: center;">
        <h3 style='margin: 0; color: white;'>{patient_condition}</h3>
    </div>
    """


def weight_result_html(weight_param, weight_info):
    """Generate HTML for weight/BMI result display."""
    return f"""
    <div style='background-color: #cccccc; padding: 10px 10px 0px; border-radius: 5px; margin-bottom: 20px; text-align: center;'>
        <h6 style='margin: 0px; font-weight: normal;'>{weight_info}</h6>
        <h2 style='margin: -30px 0 0 0;'>{weight_param}</h2>
    </div>
    """


def calories_daily_html(calories_params, calories_time):
    """Generate HTML for daily calories display."""
    return f"""
    <div style='background-color: #cccccc; padding: 10px 10px 0px; border-radius: 5px; text-align: center;'>
        <h6 style='margin: 0px; font-weight: normal;'>{calories_time}</h6>
        <h2 style='margin: -30px 0 0 0;'>{calories_params} Calories</h2>
    </div>
    """


def bmi_info_html(color, category):
    """Generate HTML for BMI category information."""
    return f"""
    <div style="background-color: {color}; padding: 0px 0px 0px; margin-top: 20px; text-align: center; border-radius: 5px;">
        <h3 style='margin: 0; color: white;'>{category}</h3>
        <h5 style='margin: -15px 0 0 0; font-weight: normal; text-align: center; color: white;'>
            A healthy BMI is generally between: 18.5 kg/m² - 25 kg/m²
        </h5>
    </div>
    """
