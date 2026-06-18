import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import pickle
import requests
import os

# Configuration
DATA_URL = "https://www.gutenberg.org/files/1661/1661-0.txt"
# DATA_FILE = "a:/Assignment/data/pak_rupee_data.txt"
# MODEL_FILE = "a:/Assignment/model/next_word_model.h5"
# TOKENIZER_FILE = "a:/Assignment/model/tokenizer.pickle"

# Use relative paths for deployment
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "../data/pak_rupee_data.txt")
MODEL_FILE = os.path.join(BASE_DIR, "next_word_model.h5") # Save in same dir as script or ensure model dir exists
TOKENIZER_FILE = os.path.join(BASE_DIR, "tokenizer.pickle")


SEQUENCE_LENGTH = 5  # Reduced sequence length
MAX_WORDS = 2500     # Increased vocab size
EPOCHS = 50            # Increased epochs for small data
BATCH_SIZE = 32        # Decreased batch size for small data





def download_data():
    if not os.path.exists(DATA_FILE):
        print(f"Downloading data from {DATA_URL}...")
        try:
            response = requests.get(DATA_URL)
            response.raise_for_status()
            os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
            with open(DATA_FILE, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading data: {e}")
            return False
    return True

def load_and_preprocess_data():
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    
    # Basic cleaning (remove some unwanted characters if necessary)
    text = text.replace('\n', ' ').replace('\r', '')
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1
    print(f"Total words: {total_words}")
    
    # Create input sequences
    input_sequences = []
    # Work with a subset of text to speed up training for assignment purposes
    corpus = text.split() 

    
    for i in range(1, len(corpus)):
        n_gram_sequence = corpus[i-SEQUENCE_LENGTH:i+1] if i >= SEQUENCE_LENGTH else corpus[:i+1]
        encoded = tokenizer.texts_to_sequences([' '.join(n_gram_sequence)])[0]
        input_sequences.append(encoded)
    
    # Pad sequences
    max_sequence_len = SEQUENCE_LENGTH + 1 # Input + Label
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    # Predictors and Label
    X, y = input_sequences[:,:-1], input_sequences[:,-1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)
    
    return X, y, total_words, tokenizer, max_sequence_len

def build_model(total_words, max_sequence_len):
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
    model.add(LSTM(150))
    model.add(Dense(total_words, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def main():
    # if not download_data():
    #     return


    print("Preprocessing data...")
    try:
        X, y, total_words, tokenizer, max_sequence_len = load_and_preprocess_data()
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return

    print("Building model...")
    model = build_model(total_words, max_sequence_len)
    
    print("Training model...")
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    
    print("Saving model and tokenizer...")
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    model.save(MODEL_FILE)
    
    with open(TOKENIZER_FILE, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Training complete and artifacts saved.")

if __name__ == "__main__":
    main()
