import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load KMeans model and scaler
with open("kmeans_model.pkl", "rb") as model_file:
    kmeans = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "welcome"

def go_to_index():
    st.session_state.page = "index"

# Apply background image using CSS
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .title-text {{
            text-align: center;
            color: white;
            font-size: 40px;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }}
        .sub-text {{
            text-align: center;
            color: white;
            font-size: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image (Use local path or online image URL)
background_image_url = "https://camo.githubusercontent.com/80130bce143a5d70f56703b66f18fc5468677ec16329cde9c5fdb83a02220a9e/68747470733a2f2f65787465726e616c2d636f6e74656e742e6475636b6475636b676f2e636f6d2f69752f3f753d6874747073253341253246253246737461746963312e73717561726573706163652e636f6d253246737461746963253246353635333039393965346230393931616233316236376231253246742532463537343835653337386136356532326438376535613135352532463134363433363035323634333025324626663d31266e6f66623d31266970743d343361383062616461653630333834306539376531663535303138376138653366313333333366623761336462313135666633633063333536323333613832362669706f3d696d61676573"  # Online image
set_background(background_image_url)

# Welcome Page
if st.session_state.page == "welcome":
    st.markdown('<p class="title-text">Welcome to EEG Seizure Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-text">This application helps predict seizures using EEG feature values.</p>', unsafe_allow_html=True)
    
    if st.button("Get Started"):
        go_to_index()

# Index Page (Main App)
elif st.session_state.page == "index":
    st.title("EEG Seizure Detection")
    st.write("Upload a CSV file with EEG feature values to predict seizure or non-seizure.")

    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:", df.head())

        if st.button("Predict"):
            input_values = df.iloc[-1].values
            input_array = np.array(input_values).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            cluster = kmeans.predict(input_scaled)[0]

            result = "Seizure" if cluster == 1 else "Non-Seizure"
            st.write(f"Predicted Cluster: {cluster} ({result})")
