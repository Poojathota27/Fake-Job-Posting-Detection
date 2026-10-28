from fastapi import FastAPI
from pydantic import BaseModel

import tensorflow as tf
import pickle
import numpy as np
import re
import string
import nltk

from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    GRU,
    Dropout,
    Dense
)

# Download stopwords (only first time)
nltk.download('stopwords')

# ===============================
# FastAPI
# ===============================

app = FastAPI(
    title="Fake Job Posting Detection API",
    description="Hybrid TextCNN + BiGRU Framework",
    version="1.0"
)

# ===============================
# Parameters
# ===============================

VOCAB_SIZE = 20000
EMBEDDING_DIM = 128
MAX_LENGTH = 300

# ===============================
# Load Tokenizer
# ===============================

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ===============================
# Stopwords
# ===============================

stop_words = set(stopwords.words("english"))

# ===============================
# Build Model
# ===============================

model = Sequential()

model.add(
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_LENGTH
    )
)

model.add(
    Conv1D(
        filters=128,
        kernel_size=5,
        activation="relu"
    )
)

model.add(
    MaxPooling1D(
        pool_size=2
    )
)

model.add(
    Bidirectional(
        GRU(64)
    )
)

model.add(
    Dropout(0.5)
)

model.add(
    Dense(
        64,
        activation="relu"
    )
)

model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)

# Build model before loading weights
model.build(input_shape=(None, MAX_LENGTH))

# ===============================
# Load Trained Weights
# ===============================

model.load_weights("TextCNN_BiGRU.weights.h5")

# ===============================
# Text Cleaning
# ===============================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"<.*?>", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\d+", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# ===============================
# Input Schema
# ===============================

class JobRequest(BaseModel):
    text: str

# ===============================
# Home API
# ===============================

@app.get("/")
def home():

    return {
        "Project": "Hybrid TextCNN + BiGRU Fake Job Detection",
        "Status": "Running Successfully"
    }

# ===============================
# Prediction API
# ===============================

@app.post("/predict")
def predict(job: JobRequest):

    text = clean_text(job.text)

    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(
        padded,
        verbose=0
    )

    probability = float(prediction[0][0])

    if probability >= 0.5:
        label = "Fake Job Posting"
    else:
        label = "Genuine Job Posting"

    return {
        "Prediction": label,
        "Probability": round(probability,4),
        "Confidence": f"{round(probability*100,2)}%"
    }