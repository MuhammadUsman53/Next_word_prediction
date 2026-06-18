import streamlit as st
import subprocess
import sys
import time
import os
import requests
import socket

# --- Helper Functions ---

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

# --- Backend Launcher ---

@st.cache_resource
def start_backend():
    """Starts the FastAPI backend using uvicorn in a subprocess."""
    if is_port_in_use(8000):
        print("Port 8000 is already in use. Assuming backend is running.")
        return None

    print("Starting FastAPI backend on port 8000...")
    cwd = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"]
    
    # Run process
    process = subprocess.Popen(cmd, cwd=cwd)
    
    # Wait for startup with progress bar
    progress_text = "Starting backend server..."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.05)
        my_bar.progress(percent_complete + 1, text=progress_text)
    
    my_bar.empty()
    return process

# Launch backend
backend_process = start_backend()

# --- Frontend UI ---

st.title("Next Word Prediction System")
st.write("Enter a sequence of words, and the model will predict the next word.")

# Check connection status
status_placeholder = st.empty()
try:
    response = requests.get("http://127.0.0.1:8000/docs", timeout=2)
    if response.status_code == 200:
        status_placeholder.success("Backend is running and connected.")
    else:
        status_placeholder.warning(f"Backend responded with status: {response.status_code}")
except requests.exceptions.ConnectionError:
    status_placeholder.error("Backend connection failed. Server might be starting or down.")

input_text = st.text_input("Enter text:", "The Pakistani rupee")

if st.button("Predict Next Word"):
    if input_text:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"text": input_text},
                timeout=10
            )
            
            if response.status_code == 200:
                prediction = response.json().get("prediction", "")
                st.success(f"Predicted next word: **{prediction}**")
            else:
                st.error(f"Error: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Error: Could not connect to backend server.")
        except requests.exceptions.Timeout:
            st.error("Error: Prediction request timed out.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please enter some text first.")
