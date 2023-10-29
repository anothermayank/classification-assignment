import pickle
import string
import numpy as np
import pandas as pd
from fastapi import FastAPI, Body, Query
from sentence_transformers import SentenceTransformer

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = FastAPI()

# Text preprocessing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = ''.join([char for char in text if char not in string.punctuation and not char.isdigit()])
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Load the sentence_transformer model and XGBoost classifier
model = SentenceTransformer("Tuned_all-MiniLM-L6-v2")
with open("xgboost_clf_model_with_transformer_tuning.pkl", "rb") as f:
    classifier = pickle.load(f)

# Define a FastAPI endpoint to predict the class of a sentence using POST method
@app.post("/predict")
async def predict(sentence: str = Body(...)):

    # Process the input sentence
    processed_sentence = preprocess_text(sentence)

    # Encode the sentence
    encoding = model.encode(processed_sentence)

    # Apply avg pooling to flatten the embeddings
    flat_encoding = pd.Series(np.mean(encoding))

    # Predict the class of the sentence
    prediction = classifier.predict(flat_encoding)

    # Return the prediction (+1 to adjust for label)
    return {"Predicted class": prediction + 1}

# Define a FastAPI endpoint to predict the class of a sentence using GET method
@app.get("/predict")
async def predict_get(sentence: str = Query(..., title="Sentence to Predict")):
    # Encode the sentence
    encoding = model.encode(sentence)

    # Apply avg pooling to flatten the embeddings
    flat_encoding = pd.Series(np.mean(encoding))

    # Predict the class of the sentence
    prediction = classifier.predict(flat_encoding)

    # Return the prediction (+1 to adjust for label)
    return {"Predicted class": prediction + 1}
