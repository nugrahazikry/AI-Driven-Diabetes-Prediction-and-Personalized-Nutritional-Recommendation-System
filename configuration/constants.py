from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import google.generativeai as genai
import os
from pathlib import Path
import pandas as pd
from google.api_core.exceptions import ResourceExhausted

# Load environment variables from .env file (project root)
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# ------------------------------------------
# LOAD UP AI CREDENTIALS
# ------------------------------------------

# # OPENAI
# OPEN_AI_API_KEY = os.getenv('OPEN_API_KEY')
# gpt_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, openai_api_key=OPEN_AI_API_KEY)

# GEMINI
GEN_AI_API_KEY = os.getenv('GEN_AI_API_KEY')
genai.configure(api_key=GEN_AI_API_KEY)

# Model fallback list (in priority order)
_MODEL_FALLBACK_LIST = [
    'gemini-2.5-flash-lite',
    'gemini-2.5-flash',
    'gemini-2.0-flash',
]


def _get_available_model():
    """
    Try each model in the fallback list and return the first one that works.
    If quota is exhausted (429), move to the next model.
    """
    
    for model_name in _MODEL_FALLBACK_LIST:
        try:
            test_model = genai.GenerativeModel(model_name=model_name)
            # Quick test to check if model is available
            test_model.generate_content("test", generation_config={"max_output_tokens": 1})
            print(f"[Gemini] Using model: {model_name}")
            return test_model
        except ResourceExhausted:
            print(f"[Gemini] Quota exhausted for {model_name}, trying next...")
            continue
        except Exception as e:
            print(f"[Gemini] Error with {model_name}: {e}, trying next...")
            continue
    
    # If all models fail, return the first one anyway (error will be handled at call time)
    print(f"[Gemini] All models unavailable, defaulting to {_MODEL_FALLBACK_LIST[0]}")
    return genai.GenerativeModel(model_name=_MODEL_FALLBACK_LIST[0])


model_generative = _get_available_model()


# ------------------------------------------
# LOAD ML MODEL
# ------------------------------------------
model = pd.read_pickle(ROOT_DIR / 'model/diabetes_prediction_best_model.pkl')

