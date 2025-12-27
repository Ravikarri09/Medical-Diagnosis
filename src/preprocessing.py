import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
def load_data(path):
    data=pd.read_csv('D:/Medical_diagnosis/Data/medical_data.csv')
    return data
def preprocess_data(data):
    tokenizer=Tokenizer(num_words=5000,oov_token='<OOV>')
    tokenizer.fit_on_texts(data['Patient_Problem'])
    sequences=tokenizer.texts_to_sequences(data['Patient_Problem'])
    max_length=max(len(x) for x in sequences)
    padded_sequences=pad_sequences(sequences,maxlen=max_length,padding='post')
    label_encoder_disease=LabelEncoder()
    label_encoder_prescription=LabelEncoder()
    disease_labels=label_encoder_disease.fit_transform(data['Disease'])
    prescription_labels=label_encoder_prescription.fit_transform(data['Prescription'])

    disease_labels_categorical=to_categorical(disease_labels)
    prescription_labels_categorical=to_categorical(prescription_labels)

    return(
        padded_sequences,
        disease_labels_categorical,
        prescription_labels_categorical,
        tokenizer,
        label_encoder_disease,
        label_encoder_prescription,
        max_length
    )
