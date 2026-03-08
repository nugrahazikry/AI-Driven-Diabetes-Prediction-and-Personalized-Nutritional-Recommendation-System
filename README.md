# AI Driven Diabetes Prediction and Personalized Nutritional Recommendation System
This project integrates machine learning and AI to support personalized diabetes management and health improvement through data-driven insights and food recommendations. By predicting diabetes status from patient health data, the system offers tailored health advice. Using these predictions, it suggests traditional Indonesian foods appropriate for various stages of diabetes and provides health insights based on selected foods. Additionally, a computer vision component with OCR detects nutritional information from food labels, allowing AI to offer real-time health guidance based on nutritional content. Together, these components provide a comprehensive, personalized approach to diabetes care and dietary management.

# Application Demo
You can try the application yourself here:

https://ai-driven-diabetes-prediction-and-personalized-nutrition-insig.streamlit.app/

# Project Structure
```
project-repo/
├── app.py                          # Flask web server & REST API endpoints
├── requirements.txt                # Python dependencies
├── .env                            # API keys and environment variables (not committed)
├── .gitignore                      # Files and folders to ignore in Git (e.g., .env, __pycache__/)
├── README.md                       # Project overview and setup instructions
│
├── templates/                      # (Renamed from 'template') Main HTML frontend
│   └── index.html                  
│
├── static/                         # Static assets (Served directly to the client)
│   ├── css/
│   │   └── styles.css              # Stylesheet
│   └── js/
│       └── app.js                  # Frontend JavaScript
│
├── configuration/                  # App configurations
│   ├── __init__.py                 
│   └── constants.py                # App-wide constants & ML model loader
│
├── pipelines/                      # Core business logic and ML integration
│   ├── __init__.py                 
│   ├── diabetes_prediction.py      # BMI/BMR calculation & AI health advice
│   ├── food_recommendation.py      # Food recommendation logic & AI insights
│   └── image_analysis.py           # OCR + AI nutritional analysis
│
├── utils/                          # Helper functions
│   ├── __init__.py                 
│   └── data_cleaning.py            # Food dataset cleaning utilities
│
├── data/                           # Local data storage (Often ignored in version control)
│   ├── dataset/                    # CSV datasets
│   └── analysis_input/             # Sample food label images
│
├── model/                          # Trained ML model files (.pkl, .h5, etc.)
│
└── notebooks/                      # Exploratory Data Analysis & training notebooks
    └── exploratory_analysis.ipynb
```

# Description
## 1. **Diabetes Prediction**
Apply machine learning to predict a patient's diabetes status based on their health data (glucose level, blood pressure, BMI, age, gender, and activity level) and use AI to generate tailored health insights and recommendations accordingly.

## 2. **Food Recommendation**
Use machine learning to provide food recommendations based on diabetes prediction results and health data, focusing on traditional Indonesian foods suitable for individuals at various stages of diabetes. Additionally, use AI to generate health insights based on the selected foods.

## 3. **Computer Vision AI Food Recommendation**
Utilize computer vision with OCR technology to detect nutritional information from food labels, enabling AI to generate health insights and personalized advice based on the nutritional composition of the food.

# Getting Started
## Dependencies
To set up the project environment, make sure all necessary libraries and dependencies are installed. Use the following command to install everything from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

# API Keys Setup
To use the AI functionalities in the project, create a `.env` file in the project's root directory. This file should store your API keys in the following format:
```
OPENAI_API_KEY=your_openai_api_key
GEMINI_AI_API_KEY=your_gemini_ai_api_key
```
Replace `your_openai_api_key` and `your_gemini_ai_api_key` with your actual OpenAI and Gemini API keys. Having this `.env` file is essential, as it allows secure access to the AI services within the project.

# Executing the Program
Once dependencies and API keys are set up, follow these steps to run and interact with the project:

## 1. **Launch the Application**
Run the Flask web server by executing:
```bash
python app.py
```
Then open your browser and navigate to:
```
http://localhost:5000
```

## 2. **Access the Prediction Page**
On the first page, input your health data  including glucose levels, blood pressure, weight, height, age, gender, and activity level  for diabetes assessment.

## 3. **Generate a Diabetes Prediction and AI Health Insights**
After entering the health data, click the **Predict your health data** button. This will provide a diabetes prediction along with AI-generated health insights, including BMI, daily calorie targets, lifestyle advice, and a health conclusion.

## 4. **Navigate to the Food Recommendation Page**
Move to the **Food Recommendations for You** page. Personalised meal options for breakfast, lunch, and dinner are listed based on your health profile. Select your preferred items from the dropdowns and click **Check nutrition information** to view AI-generated nutritional insights per meal.

## 5. **Navigate to the Food Composition Photo Analysis Page**
Go to the **Food Composition Photo Analysis** page. You can either upload your own product label image or click **Sample input** to load a pre-provided sample. Click **Analyze product image** to receive OCR-based nutritional analysis and AI health guidance tailored to your health profile.

# Contributors
Contributors names and contact info:
1. **[Zikry Adjie Nugraha](https://github.com/nugrahazikry)**: Developed the Flask web interface, implemented the Computer Vision AI Food Recommendation feature, and integrated all features with AI to gather health insights.
2. **[Diki Rustian](https://github.com/dikirust)**: Built the Diabetes Prediction feature using machine learning.
3. **[Muhammad Fikri Fadillah](https://github.com/boxside)**: Created the personalized food with Indonesian local cuisine recommendation feature.
