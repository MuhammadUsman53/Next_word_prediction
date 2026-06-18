import streamlit as st
import requests

st.title("Next Word Prediction System")
st.write("Enter a sequence of words, and the model will predict the next word.")

input_text = st.text_input("Enter text:", "The quick brown fox")

if st.button("Predict Next Word"):
    if input_text:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"text": input_text}
            )
            
            if response.status_code == 200:
                prediction = response.json().get("prediction", "")
                st.success(f"Predicted next word: **{prediction}**")
            else:
                st.error(f"Error: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Error: Could not connect to backend server. Make sure it's running.")
    else:
        st.warning("Please enter some text first.")
