## Import all important libraries
import streamlit as st
from PIL import Image, UnidentifiedImageError
import google.generativeai as genai
import mimetypes
import pandas as pd
import math
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from random import uniform as rnd
import re
import plotly.express as px
import numpy as np
import time
from pydantic import BaseModel, Discriminator
import traceback
import json
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv("../.env")

# OPEN_AI_API_KEY = os.getenv('OPEN_API_KEY')
GEN_AI_API_KEY = os.getenv('GEN_AI_API_KEY')

## Define AI configuration, dataset, and ML model
# Configure the GPT 4o mini
# gpt_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, openai_api_key=OPEN_AI_API_KEY)

# Configure the Generative AI model
genai.configure(api_key=GEN_AI_API_KEY)
model_generative = genai.GenerativeModel(model_name='gemini-1.5-flash')

# Optimize the page layout for mobile
st.set_page_config(page_title="Analisa diabetes dan kesehatan", layout="centered")

# Load Diabetes ML Model
model = pd.read_pickle('model_list/best_model.pkl')

# Load diabetes dataset
df = pd.read_csv('dataset/diabetes.csv')

# Load the food dataset
data = pd.read_csv('dataset/food_calories_dataset.csv',sep=';')
data['makanan'] = data['makanan']+' '+ data['porsi']

# Data cleaning on food dataset
col_prep=data.columns[6:15].tolist()
for col in col_prep:
    data[col]=data[col].fillna(0).astype(str)
    data[col]=np.where(data[col].str.contains('(mg)'),(data[col].str.replace(r'[a-zA-Z]','',regex=True).astype(float)/1000).astype(float),data[col].str.replace(r'[a-zA-Z]','',regex=True).astype(float))


## List of important function
# Define a function to switch pages by updating the option menu selection
def switch_page(page_name):
    st.session_state.page = page_name
    st.rerun()  # Rerun the app to reflect the change

# Scales the selected columns (6th to 14th) of the dataframe using StandardScaler.
def scaling(dataframe):
    scaler=StandardScaler()
    prep_data=scaler.fit_transform(dataframe.iloc[:,6:15].to_numpy())
    return prep_data,scaler

# Initializes and fits a Nearest Neighbors model using cosine distance on the preprocessed data.
def nn_predictor(prep_data):
    neigh = NearestNeighbors(metric='cosine',algorithm='brute')
    neigh.fit(prep_data)
    return neigh

# Creates a pipeline that scales data and applies Nearest Neighbors for prediction.
def build_pipeline(neigh,scaler,params):
    transformer = FunctionTransformer(neigh.kneighbors,kw_args=params)
    pipeline=Pipeline([('std_scaler',scaler),('NN',transformer)])
    return pipeline

# Extracts data by filtering rows containing specific ingredients from the 'komponenUtama' column.
def extract_data(dataframe,ingredients):
    extracted_data=dataframe.copy()
    extracted_data=extract_ingredient_filtered_data(extracted_data,ingredients)
    return extracted_data
    
 # Filters rows that contain all the specified ingredients (case-insensitive matching).
def extract_ingredient_filtered_data(dataframe,ingredients):
    extracted_data=dataframe.copy()
    regex_string=''.join(map(lambda x:f'(?=.*{x})',ingredients))
    extracted_data=extracted_data[extracted_data['komponenUtama'].str.contains(regex_string,regex=True,flags=re.IGNORECASE)]
    return extracted_data

# Transforms the input through the pipeline and returns the corresponding rows from the extracted data.
def apply_pipeline(pipeline,_input,extracted_data):
    _input=np.array(_input).reshape(1,-1)
    return extracted_data.iloc[pipeline.transform(_input)[0]]

# Recommends food options based on ingredient filtering and nearest neighbor matching.
def recommend(dataframe,_input,ingredients=[],params={'n_neighbors':5,'return_distance':False}):
        extracted_data=extract_data(dataframe,ingredients)
        if extracted_data.shape[0]>=params['n_neighbors']:
            prep_data,scaler=scaling(extracted_data)
            neigh=nn_predictor(prep_data)
            pipeline=build_pipeline(neigh,scaler,params)
            return apply_pipeline(pipeline,_input,extracted_data)
        else:
            return None

# Calculates Basal Metabolic Rate (BMR) based on the Harris-Benedict formula for men and women.
def calculate_bmr(weight, height, age, sex):
    if sex == "Laki-laki":
        bmr = 66.5 + (13.75 * weight) + (5.003 * height*100) - (6.75 * age)
    else:
        bmr = 655.1 + (9.563 * weight) + (1.850 * height*100) - (4.676 * age)
    return bmr
    
# Estimates daily calorie needs based on BMR and activity level.
def calculate_daily_calories(bmr, activity_level):
    if activity_level == "Sangat Ringan (Tidak olahraga)":
        calories = bmr * 1
    elif activity_level == "Ringan (olahraga 1-2 kali per minggu)":
        calories = bmr * 1.2
    elif activity_level == "Sedang (olahraga 3-4 kali hari per minggu)":
        calories = bmr * 1.375    
    elif activity_level == "Aktif (olahraga 3-5 kali per minggu)":
        calories = bmr * 1.55    
    elif activity_level == "Sangat Aktif (olahraga 6-7 kali per minggu)":
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
        category='Obesitas'    
        color='#ff2b47'
    return bmi_string,category,color   

# Generates nutrition recommendations based on calorie needs for each meal.
def generate_nutrisi():
    recommended_nutrition=[]
    makanan_list = []
    data_part_copy = data_part.copy()
    for meal,kalori_butuh in st.session_state.recommendations.items():
        
        meal_calories = kalori_butuh*st.session_state.bmr
        
        if meal=='Sarapan':        
            recommended_nutrition_part = [meal_calories,rnd(10,25),rnd(0,5),rnd(0,0.1),rnd(0,0.8),rnd(20,60),rnd(4,10),rnd(0,3),rnd(5,30)]
            data_part_copy = data_part_copy[data_part_copy['Kalori'] <= meal_calories]
        elif meal=='Siang':
            recommended_nutrition_part = [meal_calories,rnd(20,40),rnd(0,5),rnd(0,0.2),rnd(0,1.200),rnd(40,75),rnd(4,20),rnd(0,3),rnd(20,47)]
            data_part_copy = data_part_copy[data_part_copy['Kalori'] <= meal_calories]
        elif meal=='Malam':
            recommended_nutrition_part = [meal_calories,rnd(20,30),rnd(0,5),rnd(0,0.2),rnd(0,1.200),rnd(40,75),rnd(4,20),rnd(0,3),rnd(20,47)] 
            data_part_copy = data_part_copy[data_part_copy['Kalori'] <= meal_calories]
        
        generator = recommend(data_part_copy,recommended_nutrition_part,ingredients=[],params={'n_neighbors':5,'return_distance':False})

        for item in generator.to_dict('records'):
            makanan_value = item.get('makanan', 'No value found')
            makanan_list.append(makanan_value)  # Add to list

        data_part_copy = data_part_copy[~data_part_copy['makanan'].isin(makanan_list)]
        
        recommended_nutrition.append(generator.to_dict('records'))
    return recommended_nutrition

# Function to handle changes in radio selection
def update_kelamin(selected_kelamin):
    if st.session_state.jenis_kelamin != selected_kelamin:
        st.session_state.jenis_kelamin = selected_kelamin


## Define the initial session_state parameters
# Initialize the state for 'page' if not already set
if 'page' not in st.session_state:
    st.session_state.page = "Prediction"  # Default starting page
    st.session_state.bmr = None

# Initialize the state for 'uploaded_file' if not already set
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
    st.session_state.reasons_markdown_conclusion = None
    st.session_state.reasons_markdown_conclusion_food = None
    st.session_state.reasons_markdown_conclusion_image_food = None

# Initialize the state for 'image' if not already set
if 'image' not in st.session_state:
    st.session_state.image = None

# Initialize the state for 'jenis_kelamin' if not already set
if 'jenis_kelamin' not in st.session_state:
    st.session_state.jenis_kelamin = None


## Page 1: Diabetes Prediction 
if st.session_state.page == "Prediction":
    
    st.markdown("<h2 style='margin: 0; color: black; text-align: center; border: 2px solid red;'><strong>🏥💊 Prediksi level diabetes & kesehatan gizimu!</strong></h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <br><strong>Created by:</strong>
        <a href="https://www.linkedin.com/in/nugrahazikry" target="_blank" style="text-decoration: none; color: blue;">Zikry Adjie Nugraha</a>, 
        <a href="https://www.linkedin.com/in/diki-rustian/" target="_blank" style="text-decoration: none; color: blue;">Diki Rustian</a>, 
        <a href="https://www.linkedin.com/in/mff/" target="_blank" style="text-decoration: none; color: blue;">Muhammad Fikri Fadillah</a>
    </div>
    """, unsafe_allow_html=True)
    
    
    st.subheader('Masukkan data kesehatanmu')
    st.markdown(
        """
        <style>
        /* Grey background for the number input container */
        div[data-baseweb="input"] {
            background-color: #d3d3d3;  /* Light grey background */
            border-radius: 8px;         /* Rounded corners */
            padding: 5px;               /* Padding for spacing */
        }

        /* Grey background directly on the input field */
        input {
            background-color: #d3d3d3 !important;
            color: black !important;    /* Ensure text color is black */
            border: none !important;
        }

        /* Green background for the "+" and "-" buttons */
        button[aria-label="decrement"], 
        button[aria-label="increment"] {
            background-color: #32CD32 !important;  /* Lime green */
            color: white !important;              /* White text for contrast */
            border-radius: 5px;                   /* Optional: Rounded buttons */
            border: none !important;              /* Remove border */
        }

        /* Optional: Adjust button size */
        button[aria-label="decrement"], 
        button[aria-label="increment"] {
            width: 30px;
            height: 30px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Initialize session state variables if not already present
    for key, default_value in {
        'Pregnancies': 0, 'Glucose': 0, 'BloodPressure': 0,
        'SkinThickness': 0, 'Insulin': 0, 'Weight': 0.0,
        'Height': 0.0, 'DiabetesPedigreeFunction': 0, 'Age': 0,
        'BMI': None, 'prediction': None
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # Input form in two rows
    with st.container():
        row1_1, row1_2 = st.columns(2)

        with row1_1:
            st.session_state.Glucose = st.number_input(
                'Kadar Gula (mg/dL)', min_value=0, max_value=200, value=st.session_state.Glucose)
            st.session_state.BloodPressure = st.number_input(
                'Tekanan Darah Diastolik (mmHg)', min_value=0, max_value=200, value=st.session_state.BloodPressure)
            st.session_state.Age = st.number_input(
                'Umur', min_value=0, max_value=100, value=st.session_state.Age)

        with row1_2:
            st.session_state.Weight = st.number_input(
                'Berat Badan (kg)', min_value=0.0, max_value=200.0, value=st.session_state.Weight)
            st.session_state.Height = st.number_input(
                'Tinggi Badan (cm)', min_value=0.0, max_value=250.0, value=st.session_state.Height)

            # Custom CSS for the selectbox
            st.markdown(
                """
                <style>
                /* Adjust the dropdown styling */
                div[data-baseweb="select"] {
                    background-color: #D3D3D3 !important;  /* Light grey */
                    border-radius: 5px !important;
                    padding: 5px !important;
                    margin: 0px;  /* Adjust margin as needed */
                }
                </style>
                """,
                unsafe_allow_html=True)
            st.session_state.Jenis_Kelamin = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])

        st.markdown("<h4>Rencana aktivitasmu:</h4>", unsafe_allow_html=True)
        st.session_state.aktivitas = st.select_slider('', options=["Sangat Ringan (Tidak olahraga)", "Ringan (olahraga 1-2 kali per minggu)", "Sedang (olahraga 3-4 kali hari per minggu)", 'Aktif (olahraga 3-5 kali per minggu)', 'Sangat Aktif (olahraga 6-7 kali per minggu)', 'Intens'], label_visibility="collapsed")
            
    # Define the select button to predict Diabetes
    with st.container():
        if st.button("Prediksi data kesehatanmu"):
            with st.spinner("🔄 Mohon tunggu, data kesehatanmu sedang kami proses..."):
                time.sleep(3)
                if st.session_state.Height > 0:
                    # Calculate BMI
                    st.session_state.BMI = round(
                        st.session_state.Weight / (st.session_state.Height / 100) ** 2, 2)

                    # Prepare data for prediction
                    data = np.array([[
                        st.session_state.Glucose,
                        st.session_state.BloodPressure, 
                        st.session_state.BMI,
                        st.session_state.Age]])
                    
                    # Calculate bmr and bmi parameters
                    st.session_state.bmr = int(calculate_daily_calories(calculate_bmr(st.session_state.Weight, st.session_state.Height / 100, st.session_state.Age, st.session_state.Jenis_Kelamin),st.session_state.aktivitas))
                    st.session_state.bmr_light = int(calculate_daily_calories(calculate_bmr(st.session_state.Weight, st.session_state.Height / 100, st.session_state.Age, st.session_state.Jenis_Kelamin),'Sangat Ringan'))
                    st.session_state.bmi_string,st.session_state.category,st.session_state.color=bmi_calculator(st.session_state.Weight,st.session_state.Height / 100)
                    st.session_state.bmr_1=int(0.35*st.session_state.bmr)
                    st.session_state.bmr_2=int(0.4*st.session_state.bmr)
                    st.session_state.bmr_3=int(0.25*st.session_state.bmr)

                    # Make prediction
                    st.session_state.prediction = model.predict(data)
                    pasien_tidak_diabetes = 'Pasien tidak terkena diabetes'
                    pasien_iya_diabetes = 'Pasien terkena diabetes'
                    if st.session_state.prediction == 1:
                        st.session_state.kondisi_pasien = pasien_iya_diabetes
                        st.session_state.warna_kotak = '#ff2b47'
                    elif st.session_state.prediction == 0:
                        st.session_state.kondisi_pasien = pasien_tidak_diabetes
                        st.session_state.warna_kotak = '#3cb371'

                    # Define the list of data_kesehatan parameter
                    st.session_state.data_kesehatan =f"""
                    kadar glukosa: {st.session_state.Glucose} mg/DL\n
                    tekanan darah diastolik: {st.session_state.BloodPressure} mmHg\n
                    BMI: {st.session_state.BMI} kg/m²\n
                    BMR: {st.session_state.bmr} kalori per hari\n
                    umur: {st.session_state.Age} tahun\n
                    gender: {st.session_state.Jenis_Kelamin}\n
                    aktivitas harian: {st.session_state.aktivitas}\n
                    status diabetes: {st.session_state.kondisi_pasien}\n
                    """

                    # Prompt to give insight based on data_kesehatan parameter
                    diabetes_advice_prompt = f"""
                    Sebagai AI anda dapat memberikan informasi umum berupa panduan umum. Tolong jawab pertanyaan saya.
                    Anda merupakan seorang ahli gizi yang berfokus dalam penanganan diabetes dengan pengalaman kerja lebih dari 10 tahun.
                    Anda dapat memberikan saran, arahan dan informasi umum terkait masalah kesehatan yang sedang pasien alami.
                    Anda dapat menjelaskan secara rinci diagnosa keadaan data kesehatan seseorang berdasarkan dari data yang diinput oleh pasien.

                    {st.session_state.data_kesehatan}

                    Tolong berikan saya panduan kesehatan berdasarkan dari data kondisi kesehatan saya seperti di atas.
                    Buat output textnya sebagai berikut. Tidak boleh ada output lainnya.

                    Informasi tentang data kesehatan:
                    - kadar glukosa: kadar glukosa anda <kadar glukosa> mg/DL, mengindikasikan <sebab akibat pada keadaan diabetes pasien>
                    - tekanan darah diastolik: tekanan darah diastolik anda <tekanan darah diastolik> mmHg, mengindikasikan <sebab akibat pada keadaan diabetes pasien>
                    - BMI: BMI anda <BMI> kg/m², mengindikasikan <sebab akibat pada keadaan diabetes pasien>
                    - BMR: BMR anda <BMR> kalori per hari, mengindikasikan <sebab akibat pada keadaan diabetes pasien>
                    - umur: umur anda <umur> tahun, mengindikasikan <sebab akibat pada keadaan diabetes pasien>
                    - status diabetes: Berdasarkan dari data yang anda berikan, anda termasuk <terkena/tidak terkena> diabetes, <penjelasan pengaruh paling besar faktor di atas>

                    Panduan hidup sehat:
                    - Untuk <menstabilkan/mengurangi> status diabetes anda, diutamakan untuk melakukan <aktivitas atau pola hidup yang dapat menjaga kesehatan anda>
                    - Untuk <menstabilkan/mengurangi> kadar glukosa anda, <aktivitas atau pola hidup yang dapat menjaga kesehatan anda>
                    - Untuk <menstabilkan/mengurangi> tekanan darah anda, <aktivitas atau pola hidup yang dapat menjaga kesehatan anda>
                    - Untuk <menstabilkan/mengurangi> BMI anda, <aktivitas atau pola hidup yang dapat menjaga kesehatan anda>
                    - Untuk <menstabilkan> BMR anda, <aktivitas atau pola hidup yang dapat menjaga kesehatan anda>
                    - Dalam usia anda yang menginjak <umur anda>, ada baiknya anda <aktivitas atau pola hidup yang dapat menjaga kesehatan anda>
                    - dan seterusnya...

                    Kesimpulan:
                    Berdasarkan dari informasi kesehatan dan arahan panduan hidup sehat, anda harus melakukan <tips agar dapat menstabilkan atau mengurangi diabetes pasien> (kesimpulan harus kurang dari 40 kata)
                    """

                    # # Using Gemini
                    st.session_state.response_advice = model_generative.generate_content(diabetes_advice_prompt)
                    st.session_state.response_advice_text = st.session_state.response_advice.text

                    # Using GPT 4o mini
                    # prompt = ChatPromptTemplate.from_template(diabetes_advice_prompt)
                    # chain = prompt | gpt_llm
                    # response = chain.invoke({})
                    # st.session_state.response_advice_text = response.content
                    
                    # Remove all asterisks (*) from the string
                    st.session_state.response_advice_text = re.sub(r"\*", "", st.session_state.response_advice_text)

                    # Split the response into sections
                    lines = st.session_state.response_advice_text.split("\n\n")
                    health_info_output = lines[0].strip()  # "Apakah produk aman?: Tidak"
                    advices_output = lines[1].strip()  # Alasan detail
                    conclusion_output = lines[2].strip()  # Detail penjelasan komposisi gizi

                    # Define information on health_info parameter
                    health_info = health_info_output.split("Informasi tentang data kesehatan:")[1]  # Get everything after the heading
                    health_info_list = health_info.split("\n            - ")
                    health_info_list = [
                    item.strip() for item in health_info_list 
                    if item.strip() and not (item.strip().lower() == 'nan' or math.isnan(float(item.strip())) if item.strip().replace('.', '', 1).isdigit() else False)]
                    reasons_markdown_health = "\n".join([f"{item}" for item in health_info_list])

                    # Process the health_info into a list without leading hyphens
                    reasons_list = [
                        item.strip()[2:] if item.startswith("- ") else item.strip()
                        for item in reasons_markdown_health.split("\n")
                        if item.strip() and not (item.strip().lower() == 'nan' or
                        math.isnan(float(item.split(":")[1].strip())) if item.split(":")[1].strip().replace('.', '', 1).isdigit() else False)]

                    # Split into "Data Kesehatan" and "Deskripsi"
                    data = [reason.split(": ", 1) for reason in reasons_list]

                    # Convert to DataFrame and then html table
                    df = pd.DataFrame(data, columns=["Data Kesehatan", "Deskripsi"])
                    st.session_state.table_html = df.to_html(index=False, classes='dataframe', border=0)

                    # Define information on advices_output parameter
                    advices = advices_output.split("Panduan hidup sehat:")[1]  # Get everything after the heading
                    advice_list = advices.split("\n            - ")
                    advice_list = [
                    item.strip() for item in advice_list 
                    if item.strip() and not (item.strip().lower() == 'nan' or math.isnan(float(item.strip())) if item.strip().replace('.', '', 1).isdigit() else False)]
                    st.session_state.reasons_markdown_advice = "\n".join([f"{item}" for item in advice_list])

                    # Define information on conclusion parameter
                    conclusion = conclusion_output.split("Kesimpulan:")[1]  # Get everything after the heading
                    conclusion_list = conclusion.split("\n            - ")
                    conclusion_list = [
                    item.strip() for item in conclusion_list 
                    if item.strip() and not (item.strip().lower() == 'nan' or math.isnan(float(item.strip())) if item.strip().replace('.', '', 1).isdigit() else False)]
                    st.session_state.reasons_markdown_conclusion = "\n".join([f"{item}" for item in conclusion_list])

    # Showcase the result of prompt and ML prediction result
    if st.session_state.reasons_markdown_conclusion:
        st.markdown("<h3>Apakah anda terkena diabetes?</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background-color: {st.session_state.warna_kotak}; color: white; padding: 5px; border-radius: 5px; text-align: center;">
            <h3 style='margin: 0; color: white;'>{st.session_state.kondisi_pasien}</h3>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.markdown("<h3>Keterangan kesehatan anda:</h3>", unsafe_allow_html=True)

        # Showcase the result of BMI, and daily BMR
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
            f"<div style='background-color: #cccccc; padding: 10px 10px 0px; border-radius: 5px; margin-bottom: 20px; text-align: center;'>"
            f"<h6 style='margin: 0px;  font-weight: normal;'>Indeks Massa Tubuh (BMI)</h6>"
            f"<h2 style='margin: -30px 0 0 0;'>{st.session_state.bmi_string}</h2>"
            f"</div>", unsafe_allow_html=True)

        with col2:
            st.markdown(
            f"<div style='background-color: #cccccc; padding: 10px 3px 0px; border-radius: 5px; margin-bottom: 20px;  text-align: center;'>"
            f"<h6 style='margin: 0px;  font-weight: normal;'>Kalori Harian (BMR)</h6>"
            f"<h2 style='margin: -30px 0 0 0;'>{st.session_state.bmr} Kalori/hari</h2>"
            f"</div>", unsafe_allow_html=True)    
            
        col1c, col2c, col3c = st.columns(3)
        with col1c:
            st.markdown(
            f"<div style='background-color: #cccccc; padding: 10px 10px 0px; border-radius: 5px;  text-align: center;'>"
            f"<h6 style='margin: 0px;  font-weight: normal;'>Makan Pagi (Kalori)</h6>"
            f"<h2 style='margin: -30px 0 0 0;'>{st.session_state.bmr_1} Kalori</h2>"
            f"</div>", unsafe_allow_html=True)     
                
        with col2c:
            st.markdown(
                f"<div style='background-color: #cccccc; padding: 10px 10px 0px; border-radius: 5px;  text-align: center;'>"
                f"<h6 style='margin: 0px;  font-weight: normal;'>Makan Siang (Kalori)</h6>"
                f"<h2 style='margin: -30px 0 0 0;'>{st.session_state.bmr_2} Kalori</h2>"
                f"</div>", unsafe_allow_html=True)

        with col3c:
            st.markdown(
                f"<div style='background-color: #cccccc; padding: 10px 10px 0px; border-radius: 5px; text-align: center;'>"
                f"<h6 style='margin: 0px;  font-weight: normal;'>Makan Malam (Kalori)</h6>"
                f"<h2 style='margin: -30px 0 0 0;'>{st.session_state.bmr_3} Kalori</h2>"
                f"</div>", unsafe_allow_html=True) 

        st.markdown(f"""
            <div style="background-color: {st.session_state.color}; padding: 0px 0px 0px; margin-top: 20px; text-align: center; border-radius: 5px;">
                <h3 style='margin: 0; color: white;'>{st.session_state.category}</h3>
                <h5 style='margin: -15px 0 0 0;  font-weight: normal; text-align: center; color: white;'>Umumnya BMI yang sehat ada di angka: 18.5 kg/m² - 25 kg/m²</h5>
            </div>""", unsafe_allow_html=True)

        st.write("")
        st.markdown("<h3>Informasi kesehatan anda:</h3>", unsafe_allow_html=True)

        # Add CSS for responsive table styling
        st.markdown(
            """
            <style>
            table {
                width: 100%;  /* Make the table take up the full width */
                border-collapse: collapse;
            }
            th {
                text-align: center;  /* Center-align only the headers */
                padding: 8px;
                border: 1px solid #ddd;
                background-color: #ff9900;
                color: white;
            }
            td {
                text-align: left;  /* Left-align the row values */
                padding: 8px;
                border: 1px solid #ddd;

            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Display the table in Streamlit using markdown
        st.markdown(st.session_state.table_html, unsafe_allow_html=True)

        st.markdown("<h3>Saran pola hidup sehat:</h3>", unsafe_allow_html=True)
        st.markdown(st.session_state.reasons_markdown_advice, unsafe_allow_html=True)

        st.markdown("<h3>Kesimpulan kesehatan anda:</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background-color: {st.session_state.warna_kotak}; color: white; padding: 5px; border-radius: 5px; text-align: center; margin-bottom: 20px;">
            <h3 style='margin: 0; color: white; font-size: 20px;'>{st.session_state.reasons_markdown_conclusion}</h3>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.kesehatan_info_list = st.session_state.data_kesehatan

        # Button to move from one page to another
        st.markdown("<h3>Pilih halaman:</h3>", unsafe_allow_html=True)
        st.markdown(
            """
            <style>
            .element-container:has(style){
                display: none;
            }
            #button-after {
                display: none;
            }
            .element-container:has(#button-after) {
                display: none;
            }
            .element-container:has(#button-after) + div button {
                background-color: white;
                width: 300px;  /* Change this value for button width */
                height: 50px;  /* Change this value for button height */
                font-size: 500px;  /* Adjust font size */
                border-radius: 8px; /* Optional: to round the corners */
                color: black;
                border: 2px solid red; /* Added red outline */
            }
            .button-container {
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 20px; /* Space between buttons */
            }
            </style>
            """,unsafe_allow_html=True)

        st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        # Add buttons inside the columns
        with col1:
            if st.button("Rekomendasi makanan"):
                switch_page("Food recommendation")
        with col2:
            if st.button("Analisa foto produk konsumsi"):
                switch_page("Upload File")

## Page 2: Food recommendation
elif st.session_state.page == "Food recommendation":
    st.markdown(f"""
    <div style="background-color: #ffffff; padding: 0px 0px 0px; margin-top: 20px; text-align: center; border-radius: 5px;">
        <h1 style='margin: 0; color: black; text-align: center; border: 2px solid red;'><strong>🔍🍔 Rekomendasi Makanan untukmu!</strong></h1>
    </div>"""
    , unsafe_allow_html=True)

    st.write("")
    st.markdown("<h3>Data kesehatan anda:</h3>", unsafe_allow_html=True)
    
    # Showcase the result of Glucose, and daily BMR
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
        f"<div style='background-color: #cccccc; padding: 10px 10px 0px; border-radius: 5px; margin-bottom: 20px; text-align: center;'>"
        f"<h6 style='margin: 0px;  font-weight: normal;'>Kadar gula darah</h6>"
        f"<h2 style='margin: -30px 0 0 0;'>{st.session_state.Glucose} mg/dL</h2>"
        f"</div>", 
        unsafe_allow_html=True
    )          

    with col2:
        st.markdown(
        f"<div style='background-color: #cccccc; padding: 10px 10px 0px; border-radius: 5px; margin-bottom: 20px;  text-align: center;'>"
        f"<h6 style='margin: 0px;  font-weight: normal;'>Kalori Harian (BMR)</h6>"
        f"<h2 style='margin: -30px 0 0 0;'>{st.session_state.bmr} Kalori/hari</h2>"
        f"</div>", 
        unsafe_allow_html=True
    )     

    col1c, col2c,col3c = st.columns(3)
    with col1c:
        st.markdown(
        f"<div style='background-color: #cccccc; padding: 10px 10px 0px; border-radius: 5px;  text-align: center;'>"
        f"<h6 style='margin: 0px;  font-weight: normal;'>Makan Pagi (Kalori)</h6>"
        f"<h2 style='margin: -30px 0 0 0;'>{st.session_state.bmr_1} Kalori</h2>"
        f"</div>", 
        unsafe_allow_html=True
    )     
                
    with col2c:
        st.markdown(
            f"<div style='background-color: #cccccc; padding: 10px 10px 0px; border-radius: 5px;  text-align: center;'>"
            f"<h6 style='margin: 0px;  font-weight: normal;'>Makan Siang (Kalori)</h6>"
            f"<h2 style='margin: -30px 0 0 0;'>{st.session_state.bmr_2} Kalori</h2>"
            f"</div>", 
            unsafe_allow_html=True
        )

    with col3c:
        st.markdown(
            f"<div style='background-color: #cccccc; padding: 10px 10px 0px; border-radius: 5px; text-align: center;'>"
            f"<h6 style='margin: 0px;  font-weight: normal;'>Makan Malam (Kalori)</h6>"
            f"<h2 style='margin: -30px 0 0 0;'>{st.session_state.bmr_3} Kalori</h2>"
            f"</div>", 
            unsafe_allow_html=True
        )

    # Initialize session state with the percentage distribution of calorie needs for each meal.
    st.session_state.recommendations = {'Sarapan':0.35,'Siang':0.4,'Malam':0.25}
    
    # Store the predicted diabetes status from session state.
    diabet_status = st.session_state.prediction
    data_part = data.copy()

    # If the user has diabetes (prediction = 1), filter the data to include only diabetic-friendly foods.
    if diabet_status == 1:
        data_part = data_part[data_part.diabetic_friendly==1]

    # Check if 'recommended_nutrition' exists in session state.
    # If not, generate nutrition recommendations and store the user's gender for comparison.
    if 'recommended_nutrition' not in st.session_state:
        st.session_state.recommended_nutrition = generate_nutrisi()
        st.session_state.gender=st.session_state.Jenis_Kelamin

    # If the user's gender changes, regenerate nutrition recommendations.
    elif (st.session_state.gender!=st.session_state.Jenis_Kelamin):
        st.session_state.recommended_nutrition = generate_nutrisi()
        st.session_state.gender=st.session_state.Jenis_Kelamin

    st.write("")
    st.subheader('Daftar makanan hasil rekomendasi:')  
            
    col1d, col2d,col3d = st.columns(3)
    nutritions_values=['Kalori', 'lemak', 'lemakJenuh', 'kolesterol', 'sodium', 'karbohidrat', 'serat', 'gula', 'protein']
    
    # Logic to fill up the list of rekomendasi makanan
    for cold, pecahin, part_nutri in zip([col1d, col2d,col3d],st.session_state.recommendations, st.session_state.recommended_nutrition):
        cnt=0
        with cold:
            # Logic for processing rekomendasi makanan on each column of choose
            st.markdown(f'#### Menu {pecahin}')
            for recipe in part_nutri:
                recipe_name = recipe['makanan']
                expander = st.expander(recipe_name)
                expander.markdown(f'<h5 style="text-align: center;font-family:sans-serif;">Nutritional Values (g):</h5>', unsafe_allow_html=True)   
                nutritions_df = pd.DataFrame({value:[recipe[value]] for value in nutritions_values}).T
                nutritions_df = nutritions_df.rename(columns={0:'Komposisi'})
                expander.dataframe(nutritions_df)

    st.subheader('Pilih menu makanmu:')

    # Define a column on each meal time
    breakfast_column, launch_column, dinner_column=st.columns(3)
    
    with breakfast_column:
        breakfast_choice=st.selectbox(f'Choose your breakfast:',[recipe['makanan'] for recipe in st.session_state.recommended_nutrition[0]])    
    with launch_column:
        launch_choice=st.selectbox(f'Choose your launch:',[recipe['makanan'] for recipe in st.session_state.recommended_nutrition[1]])
    with dinner_column:
        dinner_choice=st.selectbox(f'Choose your dinner:',[recipe['makanan'] for recipe in st.session_state.recommended_nutrition[2]])  

    # food choices from user
    choices=[breakfast_choice,launch_choice,dinner_choice]
    
    # Calculating the sum of nutritional values of the choosen recipes
    total_nutrition_values={nutrition_value:0 for nutrition_value in nutritions_values}
    categorized_meals = {'makan sarapan': [], 'makan siang': [], 'makan malam': []}
    for choice, meals_ in zip(choices,st.session_state.recommended_nutrition):
        for meal in meals_:
            if meal['makanan'] == choice:
                
                for nutrition_value in nutritions_values:
                    total_nutrition_values[nutrition_value]+=meal[nutrition_value]

                if choice == breakfast_choice:
                    meal['waktu makan'] = 'makan sarapan'
                    categorized_meals['makan sarapan'].append(meal)
                    meal_sarapan = categorized_meals['makan sarapan'][0]
                    sarapan_string = ', '.join([f"{key}: {value}" for key, value in meal_sarapan.items()])

                elif choice == launch_choice:
                    meal['waktu makan'] = 'makan siang'
                    categorized_meals['makan siang'].append(meal)
                    meal_siang = categorized_meals['makan siang'][0]
                    siang_string = ', '.join([f"{key}: {value}" for key, value in meal_siang.items()])

                elif choice == dinner_choice:
                    meal['waktu makan'] = 'makan malam'
                    categorized_meals['makan malam'].append(meal)
                    meal_malam = categorized_meals['makan malam'][0]
                    malam_string = ', '.join([f"{key}: {value}" for key, value in meal_malam.items()])

    total_calories_chose = total_nutrition_values['Kalori']
    loss_calories_chose=round(st.session_state.bmr)

    # Define the select button to analyze the nutritional value of each selected recipe
    with st.container():
        if st.button("Cek informasi gizi"):
            with st.spinner("🔄 Mohon tunggu, analisa makananmu sedang diproses..."):
                time.sleep(5)
                food_recommendation_prompt = f"""
                Sebagai AI anda dapat memberikan informasi umum berupa panduan umum. Tolong jawab pertanyaan saya.
                Anda merupakan seorang ahli gizi yang berfokus dalam penanganan diabetes dengan pengalaman kerja lebih dari 10 tahun.
                Anda dapat memberikan saran, arahan dan informasi umum terkait masalah kesehatan yang sedang pasien alami.
                Anda dapat menjelaskan secara rinci diagnosa keadaan data kesehatan seseorang berdasarkan dari data yang diinput oleh pasien.

                (###)
                data kesehatan.

                {st.session_state.data_kesehatan}
                (###)

                (***)
                waktu makan.

                makan sarapan: {sarapan_string}

                makan siang: {siang_string}

                makan malam: {malam_string}
                (***)

                Deskripsi tugas:
                - Tolong berikan saya pandangan akan rencana konsumsi harian saya berdasarkan dari waktu makan yang sudah saya siapkan yang ditandai dengan (***).
                - Korelasikan kandungan gizi yang terdapat pada setiap makanan yang tersedia dengan data kesehatan pasien yang ditandai dengan (###).

                Buat output textnya seperti dibawah beserta poin-poin nya. Tidak boleh ada output lainnya.

                Menu makan sarapan: <menu makan sarapan yang dipilih>
                - Pola makan: Berdasarkan kandungan gula yang sebesar <gula makan sarapan> gram, Anda boleh memakan makanan ini dalam jangka waktu <sekali/dua/tiga/empat kali> dalam satu minggu, <sebab akibat jika mengonsumsi pada kesehatan level diabetes>
                - Kandungan gula: Makanan ini memiliki karbohidrat sebesar <karbohidrat makan sarapan> gram dan gula <gula makan sarapan> gram,  mengindikasikan <sebab akibat pengaruh kepada level diabetes>
                - Kalori harian: Total kalori harian pada makanan ini <kalori makan sarapan> kalori, mengindikasikan <sebab akibat pengaruh kepada total BMR dan level diabetes>
                - Nutrisi: Makanan ini kaya akan <kandungan gizi lain yang menonjol beserta angka per gram nya>, mengindikasikan <sebab akibat pengaruh terhadap level diabetes>
                - Saran penyajian: Dengan kandungan karbohidrat sebesar <karbohidrat makan sarapan> gram dan gula <gula makan sarapan> gram, alangkah baik jika makanan ini disajikan dengan <metode atau bahan tambahan atau pengganti yang dapat mengurangi kadar gula darah>

                Menu makan siang: <menu makan sarapan yang dipilih>
                - Pola makan: Berdasarkan kandungan gula yang sebesar <gula makan sarapan> gram, Anda boleh memakan makanan ini dalam jangka waktu <sekali/dua/tiga/empat kali> dalam satu minggu, <sebab akibat jika mengonsumsi pada kesehatan level diabetes>
                - Kandungan gula: Makanan ini memiliki karbohidrat sebesar <karbohidrat makan siang> gram dan gula <gula makan siang> gram,  mengindikasikan <sebab akibat pengaruh kepada level diabetes>
                - Kalori harian: Total kalori harian pada makanan ini <kalori makan siang> kalori, mengindikasikan <sebab akibat pengaruh kepada total BMR dan level diabetes>
                - Nutrisi: Makanan ini kaya akan <kandungan gizi yang paling menonjol beserta angka per gram nya>, mengindikasikan <sebab akibat pengaruh terhadap level diabetes>
                - Saran penyajian: Dengan kandungan karbohidrat sebesar <karbohidrat makan sarapan> gram dan gula <gula makan sarapan> gram, alangkah baik jika makanan ini disajikan dengan <metode atau bahan tambahan atau pengganti yang dapat mengurangi kadar gula darah>

                Menu makan malam: <menu makan sarapan yang dipilih>
                - Pola makan: Berdasarkan kandungan gula yang sebesar <gula makan sarapan> gram, Anda boleh memakan makanan ini dalam jangka waktu <sekali/dua/tiga/empat kali> dalam satu minggu, <sebab akibat jika mengonsumsi pada kesehatan level diabetes>
                - Kandungan gula: Makanan ini memiliki karbohidrat sebesar <karbohidrat makan malam> gram dan gula <gula makan malam> gram,  mengindikasikan <sebab akibat pengaruh kepada level diabetes>
                - Kalori harian: Total kalori harian pada makanan ini <kalori makan malam> kalori, mengindikasikan <sebab akibat pengaruh kepada total BMR dan level diabetes>
                - Nutrisi: Makanan ini kaya akan <kandungan gizi yang paling menonjol beserta angka per gram nya>, mengindikasikan <sebab akibat pengaruh terhadap level diabetes>
                - Saran penyajian: Dengan kandungan karbohidrat sebesar <karbohidrat makan sarapan> gram dan gula <gula makan sarapan> gram, alangkah baik jika makanan ini disajikan dengan <metode atau bahan tambahan atau pengganti yang dapat mengurangi kadar gula darah>

                Kesimpulan:
                Berdasarkan dari informasi kesehatan dan menu makanan yang anda pilih, anda harus memilih <tips yang fokus hanya dalam memilih makanan sehat yang dapat menstabilkan atau mengurangi level diabetes pasien> (kesimpulan harus kurang dari 30 kata)

                """
                # # Using Gemini AI
                st.session_state. response_food_recommend = model_generative.generate_content(food_recommendation_prompt)
                st.session_state.response_food_recommend_text = st.session_state.response_food_recommend.text

                # Using GPT 4o mini
                # prompt_food_recommend = ChatPromptTemplate.from_template(food_recommendation_prompt)
                # chain_food_recommend = prompt_food_recommend | gpt_llm
                # response_food_recommend = chain_food_recommend.invoke({})
                # st.session_state.response_food_recommend_text = response_food_recommend.content

                # Remove all asterisks (*) from the string
                st.session_state.response_food_recommend_text_recommend = re.sub(r"\*", "", st.session_state.response_food_recommend_text)

                # Split the response into sections
                lines_food_recommend = st.session_state.response_food_recommend_text_recommend.split("\n\n")
                
                # Splitting the data            
                sarapan_output = lines_food_recommend[0].strip()
                lunch_output = lines_food_recommend[1].strip()
                dinner_output = lines_food_recommend[2].strip()
                conclusion_food_output = lines_food_recommend[3].strip()

                # Focus on sarapan
                sarapan_output_detail = sarapan_output.split("\n")
                st.session_state.makanan_sarapan = sarapan_output_detail[0].split("Menu makan sarapan: ")[1]
                st.session_state.pola_sarapan = sarapan_output_detail[1].split("Pola makan: ")[1]
                st.session_state.gula_sarapan = sarapan_output_detail[2].split("Kandungan gula: ")[1]
                st.session_state.kalori_sarapan = sarapan_output_detail[3].split("Kalori harian: ")[1]
                st.session_state.nutrisi_sarapan = sarapan_output_detail[4].split("Nutrisi: ")[1]

                # Create a DataFrame for breakfast details
                st.session_state.breakfast_data = {
                    "Informasi gizi": ["Pola makan", "Kandungan gula", "Kalori harian", "Nutrisi"],
                    "Deskripsi gizi": [
                        st.session_state.pola_sarapan,
                        st.session_state.gula_sarapan,
                        st.session_state.kalori_sarapan,
                        st.session_state.nutrisi_sarapan]}

                # Convert to DataFrame and html table
                st.session_state.df_sarapan = pd.DataFrame(st.session_state.breakfast_data)
                st.session_state.df_sarapan_upload = st.session_state.df_sarapan.to_html(index=False, classes='dataframe', border=0)

                # Focus on makan siang
                lunch_output_detail = lunch_output.split("\n")
                st.session_state.makanan_lunch = lunch_output_detail[0].split("Menu makan siang: ")[1]
                st.session_state.pola_lunch = lunch_output_detail[1].split("Pola makan: ")[1]
                st.session_state.gula_lunch = lunch_output_detail[2].split("Kandungan gula: ")[1]
                st.session_state.kalori_lunch = lunch_output_detail[3].split("Kalori harian: ")[1]
                st.session_state.nutrisi_lunch = lunch_output_detail[4].split("Nutrisi: ")[1]

                # Create a DataFrame for lunch details
                st.session_state.lunch_data = {
                    "Informasi gizi": ["Pola makan", "Kandungan gula", "Kalori harian", "Nutrisi"],
                    "Deskripsi gizi": [
                        st.session_state.pola_lunch,
                        st.session_state.gula_lunch,
                        st.session_state.kalori_lunch,
                        st.session_state.nutrisi_lunch]}

                # Convert to DataFrame and html table
                st.session_state.df_lunch = pd.DataFrame(st.session_state.lunch_data)
                st.session_state.df_lunch_upload = st.session_state.df_lunch.to_html(index=False, classes='dataframe', border=0)

                # Focus on makan malam
                dinner_output_detail = dinner_output.split("\n")
                st.session_state.makanan_dinner = dinner_output_detail[0].split("Menu makan malam: ")[1]
                st.session_state.pola_dinner = dinner_output_detail[1].split("Pola makan: ")[1]
                st.session_state.gula_dinner = dinner_output_detail[2].split("Kandungan gula: ")[1]
                st.session_state.kalori_dinner = dinner_output_detail[3].split("Kalori harian: ")[1]
                st.session_state.nutrisi_dinner = dinner_output_detail[4].split("Nutrisi: ")[1]

                # Create a DataFrame for dinner details
                st.session_state.dinner_data = {
                    "Informasi gizi": ["Pola makan", "Kandungan gula", "Kalori harian", "Nutrisi"],
                    "Deskripsi gizi": [
                        st.session_state.pola_dinner,
                        st.session_state.gula_dinner,
                        st.session_state.kalori_dinner,
                        st.session_state.nutrisi_dinner]}

                # Convert to DataFrame and html table
                st.session_state.df_dinner = pd.DataFrame(st.session_state.dinner_data)
                st.session_state.df_dinner_upload = st.session_state.df_dinner.to_html(index=False, classes='dataframe', border=0)

                # Focus on the conclusion
                st.session_state.conclusion_food_output = conclusion_food_output.split("Kesimpulan:")[1]
                conclusion_list_food = st.session_state.conclusion_food_output.split("\n            - ")
                conclusion_list_food = [
                item.strip() for item in conclusion_list_food 
                if item.strip() and not (item.strip().lower() == 'nan' or math.isnan(float(item.strip())) if item.strip().replace('.', '', 1).isdigit() else False)]
                st.session_state.reasons_markdown_conclusion_food = "\n".join([f"{item}" for item in conclusion_list_food])

    # Showcase the result of prompt for food recommendation nutritional analysis result
    if st.session_state.reasons_markdown_conclusion_food:
        st.subheader('Berikut informasi gizi makananmu:')
        # Add CSS for responsive table styling
        st.markdown(
            """
            <style>
            table {
                width: 100%;  /* Make the table take up the full width */
                border-collapse: collapse;
            }
            th {
                text-align: center;  /* Center-align only the headers */
                padding: 8px;
                border: 1px solid #ddd;
                background-color: #ff9900;
                color: white;
                
            }
            td {
                text-align: left;  /* Left-align the row values */
                padding: 8px;
                border: 1px solid #ddd;

            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Add custom CSS to increase the font size of expander labels
        st.markdown(
            """
            <style>
            .streamlit-expanderHeader {
                font-size: 200px;  /* Change this to your preferred font size */
                font-weight: bold;
                color: #333;  /* Change text color if needed */
            }
            </style>
            """, unsafe_allow_html=True)

        # Showcase each nutritional value of each meal time and chosen recipe name
        with st.expander("🌅 Menu sarapan", expanded=True):
            st.markdown(
            f"<div style='background-color: #ffffff; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>"
            f"<h3 style='margin: 0;'>🌅 Menu sarapan: {st.session_state.makanan_sarapan}</h3>"
            f"</div>", unsafe_allow_html=True)

            # Display the breakfast DataFrame as a table
            st.markdown(st.session_state.df_sarapan_upload, unsafe_allow_html=True)

        with st.expander("🌤️ Menu makan siang", expanded=True):
            st.markdown(
            f"<div style='background-color: #ffffff; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>"
            f"<h3 style='margin: 0;'>🌤️ Menu makan siang: {st.session_state.makanan_lunch}</h3>"
            f"</div>", unsafe_allow_html=True)

            # Display the lunch DataFrame as a table
            st.markdown(st.session_state.df_lunch_upload, unsafe_allow_html=True)

        with st.expander("🌙 Menu makan malam", expanded=True):
            st.markdown(
            f"<div style='background-color: #ffffff; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>"
            f"<h3 style='margin: 0;'>🌙 Menu makan malam: {st.session_state.makanan_dinner}</h3>"
            f"</div>", unsafe_allow_html=True)

            # Display the dinner DataFrame as a table
            st.markdown(st.session_state.df_dinner_upload, unsafe_allow_html=True)

        st.markdown("<h3>Kesimpulan makanan pilihan anda:</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background-color: #3cb371; color: white; padding: 5px; border-radius: 5px; text-align: center; margin-bottom: 20px;">
            <h3 style='margin: 0; color: white; font-size: 20px;'>{st.session_state.reasons_markdown_conclusion_food}</h3>
        </div>
        """, unsafe_allow_html=True)

    # Button to move from one page to another
    st.markdown("<h3>Pilih halaman:</h3>", unsafe_allow_html=True)
    st.markdown(
    """
    <style>
    .element-container:has(style){
        display: none;
    }
    #button-after {
        display: none;
    }
    .element-container:has(#button-after) {
        display: none;
    }
    .element-container:has(#button-after) + div button {
        background-color: white;
        width: 300px;  /* Change this value for button width */
        height: 50px;  /* Change this value for button height */
        font-size: 500px;  /* Adjust font size */
        border-radius: 8px; /* Optional: to round the corners */
        color: black;
        border: 2px solid red; /* Added red outline */
    }
    .button-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px; /* Space between buttons */
    }
    </style>
    """,
    unsafe_allow_html=True)

    st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Kembali ke halaman kesehatan"):
            switch_page("Prediction")
    with col2:
        if st.button("Analisa foto produk konsumsi"):
            switch_page("Upload File")


## Page 3: Food composition and nutritional analysis
elif st.session_state.page == "Upload File":
    st.markdown("<h2 style='text-align: center; color: black; border: 2px solid red;'><strong>📸🍕 Analisa Nutrisi Foto Komposisi Makanan</strong></h2>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<h3>Data kesehatan anda:</h3>", unsafe_allow_html=True)

    # Showcase the result of Glucose, and daily BMR
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
        f"<div style='background-color: #cccccc; padding: 10px 10px 0px; border-radius: 5px; margin-bottom: 20px; text-align: center;'>"
        f"<h6 style='margin: 0px;  font-weight: normal;'>Kadar gula darah</h6>"
        f"<h2 style='margin: -30px 0 0 0;'>{st.session_state.Glucose} mg/dL</h2>"
        f"</div>", unsafe_allow_html=True)          

    with col2:
        st.markdown(
        f"<div style='background-color: #cccccc; padding: 10px 10px 0px; border-radius: 5px; margin-bottom: 20px;  text-align: center;'>"
        f"<h6 style='margin: 0px;  font-weight: normal;'>Kalori Harian (BMR)</h6>"
        f"<h2 style='margin: -30px 0 0 0;'>{st.session_state.bmr} Kalori/hari</h2>"
        f"</div>", unsafe_allow_html=True)     

    col1c, col2c,col3c = st.columns(3)
    with col1c:
        st.markdown(
        f"<div style='background-color: #cccccc; padding: 10px 10px 0px; border-radius: 5px;  text-align: center;'>"
        f"<h6 style='margin: 0px;  font-weight: normal;'>Makan Pagi (Kalori)</h6>"
        f"<h2 style='margin: -30px 0 0 0;'>{st.session_state.bmr_1} Kalori</h2>"
        f"</div>", unsafe_allow_html=True)     
                
    with col2c:
        st.markdown(
            f"<div style='background-color: #cccccc; padding: 10px 10px 0px; border-radius: 5px;  text-align: center;'>"
            f"<h6 style='margin: 0px;  font-weight: normal;'>Makan Siang (Kalori)</h6>"
            f"<h2 style='margin: -30px 0 0 0;'>{st.session_state.bmr_2} Kalori</h2>"
            f"</div>", unsafe_allow_html=True)

    with col3c:
        st.markdown(
            f"<div style='background-color: #cccccc; padding: 10px 10px 0px; border-radius: 5px; text-align: center;'>"
            f"<h6 style='margin: 0px;  font-weight: normal;'>Makan Malam (Kalori)</h6>"
            f"<h2 style='margin: -30px 0 0 0;'>{st.session_state.bmr_3} Kalori</h2>"
            f"</div>", unsafe_allow_html=True)

    st.write("")
    st.subheader("Upload Foto Komposisi Produk Konsumsi")

    # File uploader for image upload
    st.session_state.uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    # Define the select button to analyze the nutritional value on uploaded food composition image
    with st.container():
        if st.button("Analisa gambar produk"):
            try:
                with st.spinner("🔄 Mohon tunggu, foto dan konten sedang diproses..."):
                    # Attempt to open the image
                    st.session_state.image = Image.open(st.session_state.uploaded_file)  # Open and store the image

                    # Determine the MIME type based on the file extension
                    mime_type, _ = mimetypes.guess_type(st.session_state.uploaded_file.name)
                
                    # Upload the file to the generative AI model with the mime_type
                    myfile = genai.upload_file(st.session_state.uploaded_file, mime_type=mime_type)

                    # Define the OCR prompt
                    prompt_ocr = """
                    Berikan saya informasi komposisi gizi dan nutrisi dalam kemasan produk ini.
                    Apabila satuannya dalam g, maka anda harus mengubah nya menjadi gram.

                    Buat output textnya sebagai berikut:
                    Jenis produk: <jenis_produk>
                    1. Energi kalori: <jumlah> <kalori/kkal>
                    2. gula: <jumlah> <gram> (jika tidak tersedia bisa diskip)
                    3. karbohidrat: <jumlah> <gram> (jika tidak tersedia bisa diskip)
                    4. <lemak>: <jumlah> <gram> (jika tidak tersedia bisa diskip)
                    5. ... (seterusnya)
                                    """
                
                    # Generate content using the model
                    st.session_state.result = model_generative.generate_content([myfile, "\n\n", prompt_ocr])
                    st.session_state.komposisi = st.session_state.result.text

                    # Validate if nutritional info was found
                    if not st.session_state.komposisi.strip():
                        raise ValueError("The uploaded image did not contain relevant nutritional information.")

                    # Extract lines and clean them
                    st.session_state.lines = [line.strip() for line in st.session_state.komposisi.splitlines() if line.strip()]

                    # Extract product type separately
                    st.session_state.product_type = st.session_state.lines[0].split(":")[1].strip()

                    # Parse the remaining lines into key-value pairs
                    st.session_state.data_upload = []
                    for st.session_state.line in st.session_state.lines[1:]:
                        if ':' in st.session_state.line:
                            st.session_state.key, st.session_state.value = st.session_state.line.split(":", 1)
                            st.session_state.data_upload.append({"Kandungan Gizi": st.session_state.key.split(". ", 1)[-1].strip(), "Satuan": st.session_state.value.strip()})

                    # Convert to DataFrame
                    st.session_state.df_upload = pd.DataFrame(st.session_state.data_upload)

                    # Convert DataFrame to HTML without index
                    st.session_state.table_html_upload = st.session_state.df_upload.to_html(index=False, classes='dataframe', border=0)

                    # Define the analysis prompt based on the composition
                    prompt_analisa = f"""
                    Sebagai AI anda dapat memberikan informasi umum berupa panduan umum. Tolong jawab pertanyaan saya.
                    Anda merupakan seorang ahli gizi yang berfokus dalam penanganan diabetes dengan pengalaman kerja lebih dari 10 tahun.
                    Anda dapat memberikan saran, arahan dan informasi umum terkait masalah kesehatan yang sedang pasien alami.
                    Anda dapat menjelaskan secara rinci diagnosa keadaan data kesehatan seseorang berdasarkan dari data yang diinput oleh pasien.

                    komposisi produk:
                    {st.session_state.komposisi}

                    kondisi kesehatan:
                    {st.session_state.kesehatan_info_list}

                    Deskripsi:
                    1. Apakah dari komposisi produk di atas saya dan kondisi kesehatan saya yang difokuskan ke bagian diabetes saja, saya dapat mengonsumsi produk tersebut?
                    2. Tolong hanya fokus terhadap resiko penyakit diabetes. Jangan memberikan solusi untuk penyakit lain. 
                    3. Berikan saya jawaban yang absolut antara ya atau tidak, jangan berikan jawaban mungkin.

                    Buat output textnya sebagai berikut. Tidak boleh ada output lainnya.
                    Apakah produk direkomendasikan?: <ya atau tidak>

                    Alasan detail: 
                    - Kandungan karbohidrat sebesar <karbohidrat komposisi> gram dan gula <gula komposisi> gram, mengindikasikan <sebab akibat pengaruh kepada level diabetes> (cantumkan jika tersedia, jika tidak cantumkan gizi lain)
                    - Kandungan <komposisi gizi lainnya seperti lemak> <gram>, berpengaruh terhadap <sebab akibat pengaruh kepada level diabetes> (cantumkan jika tersedia, jika tidak cantumkan gizi lain)
                    - Kandungan <komposisi gizi lainnya seperti kalori> <gram>, berpengaruh terhadap <sebab akibat pengaruh kepada level diabetes> (cantumkan jika tersedia, jika tidak cantumkan gizi lain)
                    - dan seterusnya...

                    Informasi gizi:
                    - Pola konsumsi: Berdasarkan kandungan <gizi yang berpengaruh pada level diabetes> yang sebesar <jumlah> gram, Anda <boleh/tidak boleh mengonsumsi produk ini> (jika boleh dalam jangka waktu <sekali/dua/tiga/empat kali)> dalam satu minggu, <sebab akibat jika mengonsumsi pada kesehatan level diabetes>
                    - Kalori harian: Total kalori harian pada <makanan/minuman> ini <kalori produk> kalori, mengindikasikan <sebab akibat pengaruh kepada total BMR dan level diabetes>
                    - Nutrisi: Produk ini kaya akan <kandungan gizi lain yang menonjol beserta angka per gram nya>, mengindikasikan <sebab akibat pengaruh terhadap level diabetes>
                    - Saran penyajian: Dengan kandungan karbohidrat sebesar <karbohidrat produk> gram dan gula <gula produk> gram, alangkah baik jika produk ini <boleh atau tidak boleh disajikan untuk dapat menstabilkan gula darah anda>

                    Kesimpulan:
                    Berdasarkan dari informasi kesehatan dan kandungan gizi produk yang anda pilih, anda <direkomendasikan / tidak direkomendasikan untuk mengonsumsi produk ini> (kesimpulan harus kurang dari 30 kata)
                    """

                    # # Using Gemini AI
                    st.session_state.response = model_generative.generate_content(prompt_analisa)
                    st.session_state.response_text = st.session_state.response.text

                    # Using GPT 4o mini
                    # prompt = ChatPromptTemplate.from_template(prompt_analisa)
                    # chain = prompt | gpt_llm
                    # response = chain.invoke({})
                    # st.session_state.response_text = response.content

                    st.session_state.response_text = re.sub(r"\*", "", st.session_state.response_text)

                    # Split the response into sections
                    st.session_state.lines_picture = st.session_state.response_text.split("\n\n")
                    st.session_state.safe_question = st.session_state.lines_picture[0].strip()  # "Apakah produk aman?: Tidak"
                    st.session_state.reasons = st.session_state.lines_picture[1].strip()  # Alasan detail
                    st.session_state.nutrition_details = st.session_state.lines_picture[2].strip()  # Detail penjelasan komposisi gizi
                    st.session_state.conclusion_food_image = st.session_state.lines_picture[3].strip()  # Rekomendasi konsumsi harian

                    # Alasan detail split
                    st.session_state.reasons = st.session_state.reasons.split("Alasan detail:")[1]  # Get everything after the heading
                    st.session_state.reasons_list = st.session_state.reasons.split("\n            - ")
                    st.session_state.reasons_list = [
                    item.strip() for item in st.session_state.reasons_list 
                    if item.strip() and not (item.strip().lower() == 'nan' or math.isnan(float(item.strip())) if item.strip().replace('.', '', 1).isdigit() else False)]
                    st.session_state.reasons_markdown = "\n".join([f"{item}" for item in st.session_state.reasons_list])

                    # Focus on food nutrition
                    food_nutrition_upload_detail = st.session_state.nutrition_details.split("\n")
                    st.session_state.pola_food_upload = food_nutrition_upload_detail[1].split("Pola konsumsi: ")[1]
                    st.session_state.kalori_food_upload = food_nutrition_upload_detail[2].split("Kalori harian: ")[1]
                    st.session_state.nutrisi_food_upload = food_nutrition_upload_detail[3].split("Nutrisi: ")[1]
                    st.session_state.saran_penyajian_food_upload = food_nutrition_upload_detail[4].split("Saran penyajian: ")[1]

                    # Create a DataFrame for image details
                    st.session_state.food_upload_data = {
                        "Informasi gizi": ["Pola makan", "Kalori harian", "Nutrisi", "Saran penyajian"],
                        "Deskripsi gizi": [
                            st.session_state.pola_food_upload,
                            st.session_state.kalori_food_upload,
                            st.session_state.nutrisi_food_upload,
                            st.session_state.saran_penyajian_food_upload]}

                    # Convert to DataFrame
                    st.session_state.df_food_upload_output = pd.DataFrame(st.session_state.food_upload_data)

                    # Displaying recommendations with bullet points
                    st.session_state.conclusion_food_image_output = st.session_state.conclusion_food_image.split("Kesimpulan:")[1]
                    conclusion_list_image_food = st.session_state.conclusion_food_image_output.split("\n            - ")
                    conclusion_list_image_food = [
                    item.strip() for item in conclusion_list_image_food 
                    if item.strip() and not (item.strip().lower() == 'nan' or math.isnan(float(item.strip())) if item.strip().replace('.', '', 1).isdigit() else False)]
                    st.session_state.reasons_markdown_conclusion_image_food = "\n".join([f"{item}" for item in conclusion_list_image_food])

            except UnidentifiedImageError:
                st.error("The uploaded file is not a valid image. Please upload a valid JPG, JPEG, or PNG file. Pastikan gambar yang ada memang memuat komposisi kandungan gizi suatu makanan/minuman")
            except Exception as e:
                st.error(f"Pastikan gambar yang ada memang memuat komposisi kandungan gizi suatu makanan/minuman yang tertera jelas")
                error_message = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                st.error(f"⚠️ failed:\n{error_message}")

    # Display the image if it exists in session state
    if st.session_state.reasons_markdown_conclusion_image_food is not None:
        st.image(st.session_state.image, caption="Uploaded Image", use_column_width=True)
        try:
            # Streamlit code to display the response
            st.markdown(f"<h2 style='border: 2px solid #ff9900; text-align: center;'>🍔 Produk konsumsi: {st.session_state.product_type}</h2>", unsafe_allow_html=True)
            st.subheader("Kandungan Gizi:")

            # Add CSS for responsive table styling
            st.markdown(
                """
                <style>
                table {
                    width: 100%;  /* Make the table take up the full width */
                    border-collapse: collapse;
                }
                th {
                    text-align: center;  /* Center-align only the headers */
                    padding: 8px;
                    border: 1px solid #ddd;
                    background-color: #ff9900;
                    color: white;
                }
                td {
                    text-align: left;  /* Left-align the row values */
                    padding: 8px;
                    border: 1px solid #ddd;

                tr:nth-child(even) {
                    background-color: #f2f2f2;
                }
                </style>
                """,unsafe_allow_html=True)

            # Display your table within HTML
            st.markdown(st.session_state.table_html_upload, unsafe_allow_html=True)

            # Display the analysis response in a new text box
            st.subheader("Apakah produk direkomendasikan?")

            # Determine if the product is safe and display the corresponding colored box
            st.session_state.product_safe = st.session_state.safe_question.split(": ")[1]  # Extracting the safety status
            if st.session_state.product_safe.lower() == "ya":
                warna_kotak = '#3cb371'
                st.markdown(f"""
                <div style="background-color: {warna_kotak}; color: white; padding: 5px; border-radius: 5px; text-align: center;">
                    <h3 style='margin: 0; color: white;'>Produk direkomendasikan</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                warna_kotak = '#ff2b47'
                st.markdown(f"""
                <div style="background-color: {warna_kotak}; color: white; padding: 5px; border-radius: 5px; text-align: center;">
                    <h3 style='margin: 0; color: white;'>Produk tidak direkomendasikan</h3>
                </div>
                """, unsafe_allow_html=True)

            # Displaying reasons with bullet points
            st.markdown("<h3>Alasan detail:</h3>", unsafe_allow_html=True)
            st.markdown(st.session_state.reasons_markdown, unsafe_allow_html=True)

            # Display the breakfast DataFrame as a table
            st.markdown("<h3>Informasi gizi:</h3>", unsafe_allow_html=True)
            st.session_state.df_food_upload_output_table = st.session_state.df_food_upload_output.to_html(index=False, classes='dataframe', border=0)
            st.markdown(st.session_state.df_food_upload_output_table, unsafe_allow_html=True)

            # # Displaying nutrition details with bullet points
            st.markdown("<h3>Kesimpulan keterangan produk:</h3>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background-color: {warna_kotak}; color: white; padding: 5px; border-radius: 5px; text-align: center; margin-bottom: 20px;">
                <h3 style='margin: 0; color: white; font-size: 20px;'>{st.session_state.reasons_markdown_conclusion_image_food}</h3>
            </div>
            """, unsafe_allow_html=True)

        except UnidentifiedImageError:
            st.error("The uploaded file is not a valid image. Please upload a valid JPG, JPEG, or PNG file. Pastikan gambar yang ada memang memuat komposisi kandungan gizi suatu makanan/minuman")
        except Exception as e:
            st.error(f"Pastikan gambar yang ada memang memuat komposisi kandungan gizi suatu makanan/minuman yang tertera jelas")
            error_message = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            st.error(f"⚠️ failed:\n{error_message}")

    else:
        # Display a placeholder message if no image is uploaded
        st.info("Upload an image to analyze nutritional composition.", icon="ℹ️")

    # Button to move from one page to another
    st.markdown("<h3>Pilih halaman:</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        .element-container:has(style){
            display: none;
        }
        #button-after {
            display: none;
        }
        .element-container:has(#button-after) {
            display: none;
        }
        .element-container:has(#button-after) + div button {
            background-color: white;
            width: 300px;  /* Change this value for button width */
            height: 50px;  /* Change this value for button height */
            font-size: 500px;  /* Adjust font size */
            border-radius: 8px; /* Optional: to round the corners */
            color: black;
            border: 2px solid red; /* Added red outline */
        }
        .button-container {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px; /* Space between buttons */
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Kembali ke halaman kesehatan"):
            switch_page("Prediction")
    with col2:
        if st.button("Rekomendasi makanan"):
            switch_page("Food recommendation")
