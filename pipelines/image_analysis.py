import configuration.constants as constants
from PIL import Image, UnidentifiedImageError
import mimetypes
import google.generativeai as genai
import re
import tempfile
import os

def image_ocr_nutrition(uploaded_file):
    # Read bytes so we can use the same data for both PIL and genai
    # uploaded_file may be a Flask FileStorage or a file-like object
    file_bytes = uploaded_file.read()

    # Rewind stream so PIL can also read it
    import io
    image = Image.open(io.BytesIO(file_bytes))

    # Determine the MIME type from the original filename
    filename = getattr(uploaded_file, 'filename', None) or getattr(uploaded_file, 'name', 'image.jpg')
    mime_type, _ = mimetypes.guess_type(filename)
    if not mime_type:
        mime_type = 'image/jpeg'

    # Write to a temp file so genai.upload_file gets a real path
    suffix = os.path.splitext(filename)[-1] or '.jpg'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        myfile = genai.upload_file(tmp_path, mime_type=mime_type)
    finally:
        os.unlink(tmp_path)

    # Define the OCR prompt
    prompt_ocr = """
    Please provide me with the nutritional composition information from this product packaging.
    If the unit is in g, you must convert it to grams.

    Format your output text as follows:
    Product type: <product_type>
    1. Energy calories: <amount> <calories/kcal>
    2. sugar: <amount> <grams> (skip if not available)
    3. carbohydrates: <amount> <grams> (skip if not available)
    4. <fat>: <amount> <grams> (skip if not available)
    5. ... (and so on)
                    """

    # Generate content using the model
    result = constants.model_generative.generate_content([myfile, "\n\n", prompt_ocr])
    komposisi = result.text

    # Validate if nutritional info was found
    if not komposisi.strip():
        raise ValueError("The uploaded image did not contain relevant nutritional information.")
    
    return image, komposisi


def image_analysis_composition(composition, health_info):
    prompt_analisa = f"""
    As an AI, you can provide general information and guidance. Please answer my question.
    You are a nutritionist specializing in diabetes management with over 10 years of experience.
    You can provide advice, guidance, and general information regarding health issues the patient is experiencing.
    You can explain in detail the diagnosis of a person's health condition based on the data input by the patient.

    Product composition:
    {composition}

    Health condition:
    {health_info}

    Description:
    1. Based on the product composition above and my health condition focusing only on diabetes, can I consume this product?
    2. Please focus only on diabetes risk. Do not provide solutions for other diseases.
    3. Give me an absolute answer of yes or no, do not give a maybe answer.

    Format your output text as follows. No other output is allowed.
    Is the product recommended?: <yes or no>

    Detailed reasons: 
    - Carbohydrate content of <composition carbs> grams and sugar <composition sugar> grams, indicating <cause and effect on diabetes level> (include if available, otherwise include other nutrients)
    - Content of <other nutritional composition like fat> <grams>, affects <cause and effect on diabetes level> (include if available, otherwise include other nutrients)
    - Content of <other nutritional composition like calories> <grams>, affects <cause and effect on diabetes level> (include if available, otherwise include other nutrients)
    - and so on...

    Nutrition information:
    - Consumption pattern: Based on <nutrient affecting diabetes level> content of <amount> grams, you <may/may not consume this product> (if allowed within <once/twice/three/four times)> per week, <cause and effect if consumed on diabetes level>
    - Daily calories: Total daily calories of this <food/beverage> is <product calories> calories, indicating <cause and effect on total BMR and diabetes level>
    - Nutrition: This product is rich in <other prominent nutritional content with amount per gram>, indicating <cause and effect on diabetes level>
    - Serving suggestion: With carbohydrate content of <product carbs> grams and sugar <product sugar> grams, it would be good if this product <should or should not be served to stabilize your blood sugar>

    Conclusion:
    Based on the health information and nutritional content of the product you selected, you are <recommended / not recommended to consume this product> (conclusion must be less than 30 words)
    """

    # # Using Gemini AI
    try:
        response = constants.model_generative.generate_content(prompt_analisa)
        response_text = getattr(response, "text", None)
    except Exception:
        return None

    # Using GPT 4o mini
    # prompt = ChatPromptTemplate.from_template(prompt_analisa)
    # chain = prompt | gpt_llm
    # response = chain.invoke({})
    # response_text = response.content

    if response_text is not None:
        response_text = re.sub(r"\*", "", response_text)

    return response_text