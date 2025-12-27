import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_artifacts():
    model=load_model('models/medical_model.h5')


    with open('models/tekenizer.pkl','rb') as f:
        tokenizer=pickle.load(f)

    with open('models/disease_encoder.pkl','rb') as f:
        le_disease=pickle.load(f)
    with open('models/prescription_encoder.pkl','rb') as f:
        le_prescription=pickle.load(f)
    
    return model,tokenizer,le_disease,le_prescription

def make_prediction(patient_problem, max_length):
    model,tokenizer,le_disease,le_prescription = load_artifacts()

    sequence = tokenizer.texts_to_sequences([patient_problem])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

    prediction=model.predict(padded_sequence)
    disease_index=np.argmax(prediction[0],axis=1)[0]
    prescription_index=np.argmax(prediction[1],axis=1)[0]

    disease = le_disease.inverse_transform([disease_index])[0]
    prescription = le_prescription.inverse_transform([prescription_index])[0] 

    return disease, prescription