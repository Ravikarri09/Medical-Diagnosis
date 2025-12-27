from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.preprocessing import load_data, preprocess_data

# Load model & artifacts
model = load_model("models/medical_model.h5")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("models/disease_encoder.pkl", "rb") as f:
    le_disease = pickle.load(f)

with open("models/prescription_encoder.pkl", "rb") as f:
    le_prescription = pickle.load(f)

# Get max sequence length
data = load_data("data/raw/medical_data.csv")
_, _, _, _, _, _, max_length = preprocess_data(data)

app = Flask(__name__)


def predict_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length, padding="post")

    prediction = model.predict(padded)

    disease_idx = np.argmax(prediction[0], axis=1)[0]
    pres_idx = np.argmax(prediction[1], axis=1)[0]

    disease = le_disease.inverse_transform([disease_idx])[0]
    prescription = le_prescription.inverse_transform([pres_idx])[0]

    return disease, prescription


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["symptoms"]
    disease, prescription = predict_text(text)

    return render_template(
        "index.html",
        symptoms=text,
        disease=disease,
        prescription=prescription
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json
    disease, prescription = predict_text(data["symptoms"])
    return jsonify({
        "predicted_disease": disease,
        "suggested_prescription": prescription
    })


if __name__ == "__main__":
    app.run(debug=True)
