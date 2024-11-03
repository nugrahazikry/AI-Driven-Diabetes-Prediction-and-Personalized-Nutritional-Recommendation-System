# AI Driven Diabetes Prediction and Personalized Nutritional Recommendation System
This project integrates machine learning and AI to support personalized diabetes management and health improvement through data-driven insights and food recommendations. By predicting diabetes status from patient health data, the system offers tailored health advice. Using these predictions, it suggests traditional foods appropriate for various stages of diabetes and provides health insights based on selected foods. Additionally, a computer vision component with OCR detects nutritional information from food labels, allowing AI to offer real-time health guidance based on nutritional content. Together, these components provide a comprehensive, personalized approach to diabetes care and dietary management.

# Application Demo
You can try the application yourself in here:

https://ai-driven-diabetes-prediction-and-personalized-nutrition-insig.streamlit.app/

# Description
## 1. **Diabetes Prediction**  
   Apply machine learning to predict a patient’s diabetes status based on their health data and use AI to generate tailored health insights and recommendations accordingly.

## 2. **Food Recommendation**  
   Use machine learning to provide food recommendations based on diabetes prediction results and health data, focusing on traditional foods suitable for individuals at various stages of diabetes. Additionally, use AI to generate health insights based on the selected foods.

## 3. **Computer Vision AI Food Recommendation**  
   Utilize computer vision with OCR technology to detect nutritional information from food labels, enabling AI to generate health insights and personalized advice based on the nutritional composition of the food.

# Getting Started
## Dependencies
To set up the project environment, make sure all necessary libraries and dependencies are installed. Use the following command to install everything from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

# Installing API Keys Setup
## 1. **API Keys Setup**
  To use the AI functionalities in the project, create a .env file in the project’s root directory. This file should store your API keys in the following format:
  ```
  OPENAI_API_KEY=your_openai_api_key
  GEMINI_AI_API_KEY=your_gemini_ai_api_key
  ```
  Replace your_openai_api_key and your_gemini_ai_api_key with your actual OpenAI and Gemini API keys.

## 2. **Environment Configuration**
  Having this .env file is essential, as it allows secure access to the AI services within the project.

# Executing program
Once dependencies and API keys are set up, follow these steps to run and interact with the project:

## 1. **Launch the Application**
  Run the main Streamlit application file by executing:
```
streamlit run streamlit_code.py
```
## 2. **Access the Prediction Page**
  On the first page, input the relevant health data, including glucose levels, blood pressure, weight, height, age, and gender for diabetes assessment.

  ![Prediciton page fill data](https://github.com/nugrahazikry/healthkathon-diabetes-prediction-cyber-warriors/blob/main/dataset/Diabetes%20health%20data.png)

## 3. **Generate a Diabetes Prediction and AI health insights**
  After entering the health data, click the Predict button. This will provide a diabetes prediction for user and AI-generated health insights based on the provided data.
  
  ![Diabetes health insights](https://github.com/nugrahazikry/healthkathon-diabetes-prediction-cyber-warriors/blob/main/dataset/Prediction%20health%20data%20insights.png)

## 4. **Navigate to the Food Recommendation Page**
  Move to the Food Recommendation page, where you can select your food choices for breakfast, lunch, and dinner.
  
  ![Food recommendation page fill data](https://github.com/nugrahazikry/healthkathon-diabetes-prediction-cyber-warriors/blob/main/dataset/Food%20recommendation%20fill%20data.png)

## 5. **Get Food Recommendations and AI food insights**
  Click the Get Recommendation button to receive AI-driven healthy food recommendations. The AI will assess each food choice and provide insights on whether it is recommended for consumption based on your health profile.
  
  ![Food recommendation insights](https://github.com/nugrahazikry/healthkathon-diabetes-prediction-cyber-warriors/blob/main/dataset/food%20recommendation%20insights.png)

## 6. **Navigate to the Food composition image Nutritional Analysis page**
  Go to the Food Composition Image Nutritional Analysis section, where you can upload an image containing the nutritional details of a product for AI-based analysis and insights.
  
  ![Food image composition fill data](https://github.com/nugrahazikry/healthkathon-diabetes-prediction-cyber-warriors/blob/main/dataset/Food%20composition%20picture%20analysis%20fill%20data.png)

## 7. **Generate Health Insights from the Food composition image and recommendation**
  After uploading the image, click the Analyze Product Image button. The AI will use OCR to interpret the nutritional content and provide health insights, recommending whether the product is suitable for consumption.
  
  ![Food image example](https://github.com/nugrahazikry/healthkathon-diabetes-prediction-cyber-warriors/blob/main/dataset/Food%20picture%20composition%20for%20analysis.png)
  ![Food image insights](https://github.com/nugrahazikry/healthkathon-diabetes-prediction-cyber-warriors/blob/main/dataset/Food%20composition%20picture%20analysis%20insights.png)

# Contributors
Contributors names and contact info: 
1. **[Zikry Adjie Nugraha](https://github.com/nugrahazikry)**: Developed the Streamlit interface, implemented the Computer Vision AI Food Recommendation feature, and integrated all features with AI to gather health insights.
2. **[Diki Rustian](https://github.com/dikirust)**: Built the Diabetes Prediction feature using machine learning.
3. **[Muhammad Fikri Fadillah](https://github.com/boxside)**: Created the personalized food with Indonesian local cuisine recommendation feature.
