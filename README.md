# Sentiment Analysis using FastAPI and NLP



## Project Overview

The **Sentiment Analysis using FastAPI and Natural Language Processing (NLP)** project is a machine learning web application that analyzes text and identifies its sentiment as **Positive, Negative, or Neutral**.

The application uses TF-IDF (Term Frequency–Inverse Document Frequency) for text feature extraction and Logistic Regression for sentiment classification. FastAPI provides REST API endpoints, while an interactive dark-themed dashboard allows users to enter text and view prediction results.
 
## Key Features

- **Sentiment Classification:** Predicts positive, negative, or neutral sentiment.
- **NLP Text Processing:** Converts text into numerical features using TF-IDF.
- **Machine Learning:** Uses a Logistic Regression classification model.
- **FastAPI REST API:** Provides endpoints for individual and batch predictions.
- **Interactive Dashboard:** Modern dark-themed user interface.
- **Prediction Confidence:** Displays model probability scores where available.
- **Batch Analysis:** Supports analyzing multiple text inputs.
- **Prediction History:** Displays recent predictions during the current application session.
- **Model Persistence:** Saves the trained model using Joblib.
- **API Documentation:** Interactive API testing through Swagger UI.

## Technologies Used

- **Programming Language:** Python
- **Backend Framework:** FastAPI
- **Machine Learning:** Scikit-learn
- **Natural Language Processing:** TF-IDF Vectorization
- **Algorithm:** Logistic Regression
- **Data Processing:** Pandas
- **Model Serialization:** Joblib
- **Frontend:** HTML, CSS, JavaScript
- **API Server:** Uvicorn

## Project Structure

```text
SENTIMENT_ANALYSIS_FASTAPI/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── sentiment.py
├── ml/
│   ├── train_model.py
│   ├── reviews.csv
│   └── sentiment_model.joblib
├── static/
│   ├── style.css
│   └── dashboard.js
├── templates/
│   └── index.html
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/SENTIMENT_ANALYSIS_FASTAPI.git
cd SENTIMENT_ANALYSIS_FASTAPI
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Train the machine learning model

```bash
python ml/train_model.py
```

### 4. Start the FastAPI server

```bash
python -m uvicorn app.main:app --reload
```

### 5. Open the application

- **Dashboard:** http://127.0.0.1:8000/
- **Swagger API Documentation:** http://127.0.0.1:8000/docs

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Opens the dashboard |
| GET | `/api/health` | Checks application health |
| POST | `/api/predict` | Predicts sentiment for one text |
| POST | `/api/batch` | Predicts sentiment for multiple texts |
| GET | `/api/stats` | Retrieves prediction statistics |
| GET | `/api/history` | Retrieves recent prediction history |
| GET | `/api/export` | Exports prediction history |

## Example

**Input:**
```text
I really love this product. The quality is excellent!
```

**Expected sentiment:** Positive

Another example:

**Input:**
```text
The service was terrible and disappointing.
```

**Expected sentiment:** Negative

Predictions depend on the training dataset and model performance.

## Learning Outcomes

This project demonstrates practical skills in:

- Natural Language Processing and text classification
- Feature extraction using TF-IDF
- Training and saving machine learning pipelines
- Building REST APIs with FastAPI
- Integrating a frontend with a machine learning backend
- Handling JSON requests and prediction responses
- Organizing and deploying a Python application

 ##Future Improvements

- Train the model on a larger, real-world labeled dataset.
- Add advanced text preprocessing and lemmatization.
- Experiment with Naive Bayes, Support Vector Machines, and transformer models.
- Add model evaluation metrics, including precision, recall, F1-score, and a confusion matrix.
- Deploy the application to a cloud hosting platform.
- Add persistent storage for prediction history.

## Disclaimer

This project is intended for educational and portfolio purposes. Its current sample training dataset is small, so predictions should not be treated as production-grade sentiment analysis.

