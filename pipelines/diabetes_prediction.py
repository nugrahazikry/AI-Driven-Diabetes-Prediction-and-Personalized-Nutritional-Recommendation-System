import configuration.constants as constants
import re
import traceback

# Calculates Basal Metabolic Rate (BMR) based on the Harris-Benedict formula for men and women.
def calculate_bmr(weight, height, age, sex):
    if sex == "Male":
        bmr = 66.5 + (13.75 * weight) + (5.003 * height*100) - (6.75 * age)
    else:
        bmr = 655.1 + (9.563 * weight) + (1.850 * height*100) - (4.676 * age)
    return bmr
    
# Estimates daily calorie needs based on BMR and activity level.
def calculate_daily_calories(bmr, activity_level):
    if activity_level == "Sedentary (No exercise)":
        calories = bmr * 1
    elif activity_level == "Light (exercise 1-2 times per week)":
        calories = bmr * 1.2
    elif activity_level == "Moderate (exercise 3-4 times per week)":
        calories = bmr * 1.375    
    elif activity_level == "Active (exercise 3-5 times per week)":
        calories = bmr * 1.55    
    elif activity_level == "Very Active (exercise 6-7 times per week)":
        calories = bmr * 1.725
    else:
        calories = bmr * 1.9
    return calories

# Calculates BMI and returns the value with its category and corresponding color code.
def bmi_calculator(weight,height):
    bmi=round(weight/((height)**2),2)
    bmi_string=f'{bmi} kg/m²'
    if bmi<18.5:
        category='Underweight'
        color='#ff2b47'
    elif 18.5<=bmi<25:
        category='Normal'
        color='#3cb371'
    elif 25<=bmi<30:
        category='Overweight'
        color='#ffa500'
    else:
        category='Obese'    
        color='#ff2b47'
    return bmi_string,category,color   



def diabetes_advice_prompt_process(health_data):
    
    # Prompt to give insight based on health_data parameter
    diabetes_advice_prompt = f"""
    As an AI, you can provide general information and guidance. Please answer my question.
    You are a nutritionist specializing in diabetes management with over 10 years of experience.
    You can provide advice, guidance, and general information regarding health issues the patient is experiencing.
    You can explain in detail the diagnosis of a person's health condition based on the data input by the patient.

    {health_data}

    Please provide me with health guidance based on my health condition data above.
    Format your output text as follows. No other output is allowed.

    Health data information:
    - glucose level: your glucose level is <glucose level> mg/dL, indicating <cause and effect on patient's diabetes condition>
    - diastolic blood pressure: your diastolic blood pressure is <diastolic blood pressure> mmHg, indicating <cause and effect on patient's diabetes condition>
    - BMI: your BMI is <BMI> kg/m², indicating <cause and effect on patient's diabetes condition>
    - BMR: your BMR is <BMR> calories per day, indicating <cause and effect on patient's diabetes condition>
    - age: your age is <age> years old, indicating <cause and effect on patient's diabetes condition>
    - diabetes status: Based on the data you provided, you are <affected/not affected> by diabetes, <explanation of the most significant contributing factor above>

    Healthy lifestyle guidelines:
    - To <stabilize/reduce> your diabetes status, it is recommended to <activities or lifestyle habits that can maintain your health>
    - To <stabilize/reduce> your glucose level, <activities or lifestyle habits that can maintain your health>
    - To <stabilize/reduce> your blood pressure, <activities or lifestyle habits that can maintain your health>
    - To <stabilize/reduce> your BMI, <activities or lifestyle habits that can maintain your health>
    - To <stabilize> your BMR, <activities or lifestyle habits that can maintain your health>
    - At your age of <your age>, it would be good for you to <activities or lifestyle habits that can maintain your health>
    - and so on...

    Conclusion:
    Based on the health information and healthy lifestyle guidelines, you should <tips to stabilize or reduce patient's diabetes> (conclusion must be less than 40 words)
    """

    # # Using Gemini
    try:
        response_advice = constants.model_generative.generate_content(diabetes_advice_prompt)
        response_advice_text = getattr(response_advice, "text", None)
    except Exception as e:
        # If the generative API fails (quota, credentials, etc.), return None and
        # the formatted traceback so callers can show diagnostic information.
        tb = traceback.format_exc()
        return None, tb

    # Using GPT 4o mini (fallback example)
    # prompt = ChatPromptTemplate.from_template(diabetes_advice_prompt)
    # chain = prompt | gpt_llm
    # response = chain.invoke({})
    # response_advice_text = response.content

    # Remove all asterisks (*) from the string
    if response_advice_text is not None:
        response_advice_text = re.sub(r"\*", "", response_advice_text)

    return response_advice_text
