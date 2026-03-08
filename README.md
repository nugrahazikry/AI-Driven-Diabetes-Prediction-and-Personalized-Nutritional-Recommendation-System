# AI-Driven Diabetes Prediction and Personalized Nutritional Recommendation System

This project integrates machine learning and generative AI to support personalized diabetes management through data-driven insights and food recommendations. It predicts a patient's diabetes status from health data and offers tailored health advice, suggests traditional Indonesian foods appropriate for various stages of diabetes, and uses computer vision with OCR to analyze nutritional information from food labels for real-time AI-powered guidance.

The application is containerized with Docker and served via an nginx frontend proxy communicating with a Flask REST API backend.

---

## Table of Contents

1. [Application Demo](#application-demo)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Environment Variables](#environment-variables)
   - [Run with Docker (Recommended)](#run-with-docker-recommended)
   - [Run Locally (Without Docker)](#run-locally-without-docker)
6. [API Endpoints](#api-endpoints)
7. [Makefile Commands](#makefile-commands)
8. [Usage Guide](#usage-guide)

---

## Application Demo

You can try the application yourself here:

https://ai-driven-diabetes-prediction-and-personalized-nutrition-insig.streamlit.app/

---

## Features

- **Diabetes Prediction** — ML model predicts diabetes status (normal / pre-diabetes / diabetes) from patient health data.
- **AI Health Insights** — Generative AI produces personalized BMI analysis, daily calorie targets, and lifestyle advice.
- **Food Recommendation** — Suggests traditional Indonesian meal options for breakfast, lunch, and dinner tailored to the user's health profile.
- **Computer Vision Food Label Analysis** — OCR extracts nutritional data from food label images, and AI generates health guidance based on the nutritional content.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, Flask, Gunicorn |
| ML | scikit-learn, NumPy, pandas |
| Generative AI | Google Gemini (`google-generativeai`), LangChain OpenAI |
| Computer Vision | Pillow (OCR pipeline) |
| Frontend | HTML, CSS, JavaScript (served by nginx) |
| Containerization | Docker, Docker Compose |

---

## Project Structure

```
v3_2/
├── docker-compose.yml              # Orchestrates backend + frontend services
├── Makefile                        # Convenience commands for Docker workflows
├── README.md                       # Project overview and setup instructions
│
├── backend/                        # Flask REST API
│   ├── app.py                      # API entry point & route definitions
│   ├── Dockerfile                  # Backend Docker image (python:3.11-slim + gunicorn)
│   ├── requirements.txt            # Python dependencies
│   ├── .env                        # API keys (not committed — see Environment Variables)
│   │
│   ├── configuration/
│   │   ├── __init__.py
│   │   └── constants.py            # App-wide constants & ML model loader
│   │
│   ├── pipelines/                  # Core business logic
│   │   ├── diabetes_prediction.py  # BMI/BMR calculation & AI health advice
│   │   ├── food_recommendation.py  # Food recommendation logic & AI insights
│   │   └── image_analysis.py       # OCR + AI nutritional label analysis
│   │
│   ├── utils/
│   │   └── data_cleaning.py        # Food dataset cleaning utilities
│   │
│   ├── data/
│   │   ├── dataset/
│   │   │   ├── diabetes.csv        # Diabetes training dataset
│   │   │   └── food_calories_dataset.csv  # Indonesian food nutrition dataset
│   │   └── analysis_input/         # Sample food label images for OCR
│   │
│   ├── model/                      # Trained ML model files (.pkl / .h5)
│   │
│   └── notebooks/
│       └── ml_model_diabetes_prediction.ipynb  # Model training & EDA notebook
│
└── frontend/                       # nginx static server + API proxy
    ├── Dockerfile                  # Frontend Docker image (nginx:alpine)
    ├── nginx.conf                  # Reverse-proxy config (routes /api/* → backend)
    └── public/
        ├── index.html              # Main production HTML
        ├── index_dev.html          # Development HTML
        └── static/
            ├── css/
            │   └── styles.css      # Application stylesheet
            └── js/
                └── app.js          # Frontend JavaScript
```

---

## Getting Started

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (recommended)
- Or Python 3.11+ for running locally without Docker
- A Google Gemini API key and/or an OpenAI API key

### Environment Variables

Create a `.env` file inside the `backend/` directory:

```
GEN_AI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
```

> This file is loaded automatically by Docker Compose via `env_file: ./backend/.env` and by `python-dotenv` when running locally.

---

### Run with Docker (Recommended)

**Build and start both services (foreground):**
```bash
docker compose up --build
```

**Or use the Makefile shorthand:**
```bash
make start      # build & start detached (background)
```

The application will be available at:
```
http://localhost
```

The backend API is exposed internally on port `5000` and is reachable only through the nginx proxy — it is not directly accessible from the host.

---

### Run Locally (Without Docker)

1. Install Python dependencies:
```bash
pip install -r backend/requirements.txt
```

2. Start the Flask development server:
```bash
cd backend
python app.py
```

3. Open your browser at:
```
http://localhost:5000
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check — returns `{"status": "ok"}` |
| `POST` | `/api/predict` | Diabetes prediction + AI health insights |
| `POST` | `/api/food` | Food recommendations based on health profile |
| `POST` | `/api/image` | OCR nutritional analysis of a food label image |

---

## Makefile Commands

> Requires `make` (available via Git Bash, WSL, or `choco install make` on Windows).

| Command | Description |
|---|---|
| `make build` | Build both Docker images |
| `make up` | Build & start in foreground |
| `make start` | Build & start detached (background) |
| `make stop` | Stop containers (keep them) |
| `make down` | Stop & remove containers |
| `make restart` | `down` then `start` |
| `make clean` | Remove containers, images, and build cache |
| `make logs` | Tail logs for all services |
| `make logs-be` | Tail backend logs only |
| `make logs-fe` | Tail frontend logs only |
| `make status` | Show running container status |
| `make shell-be` | Open bash shell in backend container |
| `make shell-fe` | Open sh shell in frontend container |

---

## Usage Guide

### 1. Diabetes Prediction
On the home page, enter your health data — glucose level, blood pressure, weight, height, age, gender, and activity level — then click **Predict your health data**. The system returns a diabetes prediction along with AI-generated health insights including BMI, daily calorie targets, and lifestyle recommendations.

### 2. Food Recommendations
Navigate to the **Food Recommendations for You** page. Personalized meal options for breakfast, lunch, and dinner are displayed based on your health profile. Select preferred items from the dropdowns and click **Check nutrition information** to view AI-generated nutritional insights for each meal.

### 3. Food Label Photo Analysis
Go to the **Food Composition Photo Analysis** page. Upload a product label image or click **Sample input** to load a pre-provided example, then click **Analyze product image**. The system uses OCR to extract nutritional data from the label and generates AI health guidance tailored to your health profile.

# Contributors
Contributors names and contact info:
1. **[Zikry Adjie Nugraha](https://github.com/nugrahazikry)**: Developed the Flask web interface, implemented the Computer Vision AI Food Recommendation feature, and integrated all features with AI to gather health insights.
2. **[Diki Rustian](https://github.com/dikirust)**: Built the Diabetes Prediction feature using machine learning.
3. **[Muhammad Fikri Fadillah](https://github.com/boxside)**: Created the personalized food with Indonesian local cuisine recommendation feature.
