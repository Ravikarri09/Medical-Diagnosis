from tensorflow.keras.models import load_model
import numpy as np

def evaluate(model_path,X, y):
    model=load_model(model_path)
    results=model.evaluate(X, y)
    print("Evaluation Results:",results)