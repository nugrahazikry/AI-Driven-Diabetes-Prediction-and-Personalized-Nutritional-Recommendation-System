from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import re
from random import uniform as rnd
import configuration.constants as constants

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

 # Filters rows that contain all the specified ingredients (case-insensitive matching).
def extract_ingredient_filtered_data(dataframe,ingredients):
    extracted_data=dataframe.copy()
    regex_string=''.join(map(lambda x:f'(?=.*{x})',ingredients))
    extracted_data=extracted_data[extracted_data['komponenUtama'].str.contains(regex_string,
                                                                               regex=True,
                                                                               flags=re.IGNORECASE)]
    return extracted_data

def extract_data(dataframe,ingredients):
    extracted_data=dataframe.copy()
    extracted_data=extract_ingredient_filtered_data(extracted_data,ingredients)
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
        

# Generates nutrition recommendations based on calorie needs for each meal.
def generate_nutrisi(data_part_input, recommendations, bmr):
    recommended_nutrition=[]
    makanan_list = []
    data_part_copy = data_part_input.copy()
    # for meal,kalori_butuh in st.session_state.recommendations.items():
    for meal,kalori_butuh in recommendations.items():
        
        # meal_calories = kalori_butuh*st.session_state.bmr
        meal_calories = kalori_butuh * bmr
        
        if meal=='Breakfast':        
            recommended_nutrition_part = [meal_calories,rnd(10,25),rnd(0,5),rnd(0,0.1),rnd(0,0.8),rnd(20,60),rnd(4,10),rnd(0,3),rnd(5,30)]
            data_part_copy = data_part_copy[data_part_copy['Kalori'] <= meal_calories]
        elif meal=='Lunch':
            recommended_nutrition_part = [meal_calories,rnd(20,40),rnd(0,5),rnd(0,0.2),rnd(0,1.200),rnd(40,75),rnd(4,20),rnd(0,3),rnd(20,47)]
            data_part_copy = data_part_copy[data_part_copy['Kalori'] <= meal_calories]
        elif meal=='Dinner':
            recommended_nutrition_part = [meal_calories,rnd(20,30),rnd(0,5),rnd(0,0.2),rnd(0,1.200),rnd(40,75),rnd(4,20),rnd(0,3),rnd(20,47)] 
            data_part_copy = data_part_copy[data_part_copy['Kalori'] <= meal_calories]
        
        generator = recommend(data_part_copy,recommended_nutrition_part,ingredients=[],params={'n_neighbors':5,'return_distance':False})

        for item in generator.to_dict('records'):
            makanan_value = item.get('makanan', 'No value found')
            makanan_list.append(makanan_value)  # Add to list

        data_part_copy = data_part_copy[~data_part_copy['makanan'].isin(makanan_list)]
        
        recommended_nutrition.append(generator.to_dict('records'))

    return recommended_nutrition




def food_recommendation_prompt_process(health_data, breakfast_string,
                                       lunch_string, dinner_string):
    
    food_recommendation_prompt = f"""As an AI, you can provide general information and guidance. Please answer my question.
    You are a nutritionist specializing in diabetes management with over 10 years of experience.
    You can provide advice, guidance, and general information regarding health issues the patient is experiencing.
    You can explain in detail the diagnosis of a person's health condition based on the data input by the patient.

    (###)
    Health data.

    {health_data}
    (###)

    (***)
    Meal times.

    breakfast: {breakfast_string}

    lunch: {lunch_string}

    dinner: {dinner_string}
    (***)

    Task description:
    - Please provide me with insights on my daily consumption plan based on the meal times I have prepared, marked with (***).
    - Correlate the nutritional content in each available food with the patient's health data marked with (###).

    Format your output text as follows with bullet points. No other output is allowed.

    Breakfast menu: <selected breakfast menu>
    - Eating pattern: Based on the sugar content of <breakfast sugar> grams, you may eat this food <once/twice/three/four times> per week, <cause and effect if consumed on diabetes level>
    - Sugar content: This food has carbohydrates of <breakfast carbs> grams and sugar <breakfast sugar> grams, indicating <cause and effect on diabetes level>
    - Daily calories: Total daily calories from this food is <breakfast calories> calories, indicating <cause and effect on total BMR and diabetes level>
    - Nutrition: This food is rich in <most prominent nutritional content with amount per gram>, indicating <cause and effect on diabetes level>
    - Serving suggestion: With carbohydrate content of <breakfast carbs> grams and sugar <breakfast sugar> grams, it would be good if this food is served with <method or additional ingredients or substitutes that can reduce blood sugar levels>

    Lunch menu: <selected lunch menu>
    - Eating pattern: Based on the sugar content of <lunch sugar> grams, you may eat this food <once/twice/three/four times> per week, <cause and effect if consumed on diabetes level>
    - Sugar content: This food has carbohydrates of <lunch carbs> grams and sugar <lunch sugar> grams, indicating <cause and effect on diabetes level>
    - Daily calories: Total daily calories from this food is <lunch calories> calories, indicating <cause and effect on total BMR and diabetes level>
    - Nutrition: This food is rich in <most prominent nutritional content with amount per gram>, indicating <cause and effect on diabetes level>
    - Serving suggestion: With carbohydrate content of <lunch carbs> grams and sugar <lunch sugar> grams, it would be good if this food is served with <method or additional ingredients or substitutes that can reduce blood sugar levels>

    Dinner menu: <selected dinner menu>
    - Eating pattern: Based on the sugar content of <dinner sugar> grams, you may eat this food <once/twice/three/four times> per week, <cause and effect if consumed on diabetes level>
    - Sugar content: This food has carbohydrates of <dinner carbs> grams and sugar <dinner sugar> grams, indicating <cause and effect on diabetes level>
    - Daily calories: Total daily calories from this food is <dinner calories> calories, indicating <cause and effect on total BMR and diabetes level>
    - Nutrition: This food is rich in <most prominent nutritional content with amount per gram>, indicating <cause and effect on diabetes level>
    - Serving suggestion: With carbohydrate content of <dinner carbs> grams and sugar <dinner sugar> grams, it would be good if this food is served with <method or additional ingredients or substitutes that can reduce blood sugar levels>

    Conclusion:
    Based on the health information and food menu you selected, you should choose <tips focusing only on selecting healthy foods that can stabilize or reduce patient's diabetes level> (conclusion must be less than 30 words)

    """
    # # Using Gemini AI
    try:
        response_food_recommend = constants.model_generative.generate_content(food_recommendation_prompt)
        response_food_recommend_text = getattr(response_food_recommend, "text", None)
    except Exception:
        return None

    # Using GPT 4o mini
    # prompt_food_recommend = ChatPromptTemplate.from_template(food_recommendation_prompt)
    # chain_food_recommend = prompt_food_recommend | gpt_llm
    # response_food_recommend = chain_food_recommend.invoke({})
    # response_food_recommend_text = response_food_recommend.content

    if response_food_recommend_text is None:
        return None

    # Remove all asterisks (*) from the string
    response_food_recommend_text_recommend = re.sub(r"\*", "", response_food_recommend_text)

    # Split the response into sections
    lines_food_recommend = response_food_recommend_text_recommend.split("\n\n")

    return lines_food_recommend
