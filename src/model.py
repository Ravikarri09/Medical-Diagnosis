from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Embedding,LSTM,Dense

def build_model(max_length, num_diseases, num_prescriptions):
    input_layer=Input(shape=(max_length,))

    embedding=Embedding(input_dim=5000, output_dim=64)(input_layer)
    lstm_layer=LSTM(64)(embedding)

    disease_output=Dense(num_diseases, activation='softmax', name='disease_output')(lstm_layer)
    prescription_output=Dense(num_prescriptions, activation='softmax', name='prescription_output')(lstm_layer)
    model=Model(inputs=input_layer, outputs=[disease_output, prescription_output])

    model.compile(
        loss={
            'disease_output':'categorical_crossentropy',
            'prescription_output':'categorical_crossentropy'
        },
        optimizer='adam',
        metrics={
            'disease_output':['accuracy'],
            'prescription_output':['accuracy']
        }
    )

    return model