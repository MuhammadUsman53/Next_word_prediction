from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
import pickle
import os

app = FastAPI()

# Use relative paths for deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "../model/next_word_model.onnx")
TOKENIZER_FILE = os.path.join(BASE_DIR, "../model/tokenizer.pickle")

SEQUENCE_LENGTH = 5 # Must match training

model = None
tokenizer = None

def pad_sequences(sequences, maxlen, padding='pre'):
    padded = np.zeros((len(sequences), maxlen), dtype=np.float32)
    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue
        if len(seq) > maxlen:
            if padding == 'pre':
                padded[i] = seq[-maxlen:]
            else:
                padded[i] = seq[:maxlen]
        else:
            if padding == 'pre':
                padded[i, -len(seq):] = seq
            else:
                padded[i, :len(seq)] = seq
    return padded

def load_resources():
    global model, tokenizer
    try:
        print("Loading model...")
        model = ort.InferenceSession(MODEL_FILE)
        print("Model loaded.")
        
        print("Loading tokenizer...")
        with open(TOKENIZER_FILE, 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("Tokenizer loaded.")
    except Exception as e:
        print(f"Error loading resources: {e}")

@app.on_event("startup")
async def startup_event():
    load_resources()

class PredictionRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict_next_word(request: PredictionRequest):
    global model, tokenizer
    if model is None or tokenizer is None:
        load_resources()
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

    text = request.text.lower()
    
    # Preprocess
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=SEQUENCE_LENGTH, padding='pre')
    
    # Predict
    input_name = model.get_inputs()[0].name
    predicted = model.run(None, {input_name: token_list})[0]
    predicted_index = np.argmax(predicted, axis=-1)[0]
    
    predicted_word = tokenizer.index_word.get(predicted_index, "")
            
    return {"prediction": predicted_word}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

