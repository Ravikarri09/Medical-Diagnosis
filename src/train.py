from src.preprocessing import load_data,preprocess_data
from src.model import build_model

def train():
    #load the data
    data=load_data('data/medical_records.csv')
    (X,y_disease,y_prescription,
     tokenizer, le_disease, le_prescription,
     max_length)= preprocess_data(data)
    
    model= build_model(
        max_length,
        y_disease.shape[1],
        y_prescription.shape[1]
    )

    model.fit(
        X,
        {'disease_output':y_disease,
         'prescription_output':y_prescription},
         epochs=100,
         batch_size=32
    )
    model.save('models/medical_model.h5')
    import pickle
    with open('models/tokenizer.pkl','wb') as f:
        pickle.dump(tokenizer,f)
    with open('models/disease_encoder.pkl','wb') as f:
        pickle.dump(le_disease,f)
    with open('models/prescription_encoder.pkl','wb') as f:
        pickle.dump(le_prescription,f)
if __name__=='__main__':
    train()
