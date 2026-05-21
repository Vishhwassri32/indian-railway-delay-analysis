import streamlit as st
import pandas as pd
st.title("🚂 Indian Railways Delay Predictor")
st.write("This model was built **completely from scratch** using basic math (Linear Regression)!")
@st.cache_data
def train_scratch_model():
    df = pd.read_csv('indian_railway_delay_data_.csv')
    def clean_to_minutes(time_text):
        if pd.isna(time_text): return 0
        time_text = str(time_text).strip().split()[-1]
        try:
            h, m, s = map(int, time_text.split(':'))
            return h * 60 + m
        except: return 0 
    df['Delay_minutes'] = df['Dealy_min'].apply(clean_to_minutes)
    X = df['Distance(Km)'].tolist()
    y = df['Delay_minutes'].tolist() 
    mean_x = sum(X) / len(X)
    mean_y = sum(y) / len(y)
    num, den = 0, 0
    for i in range(len(X)):
        num += (X[i] - mean_x) * (y[i] - mean_y)
        den += (X[i] - mean_x) ** 2  
    m = num / den
    c = mean_y - (m * mean_x)
    return m, c
# Get our trained model weights
m, c = train_scratch_model()
# --- USER INPUT CONTROLS ---
st.header("Enter New Train Details:")
distance_input = st.number_input("Train Route Distance (in Km):", min_value=10, max_value=5000, value=1200)
# --- PREDICTION BUTTON ---
if st.button("Predict Delay Time 🎯"):
    # y = mx + c
    predicted_delay = (m * distance_input) + c
    
    st.success(f"### ⏱️ Predicted Delay: {predicted_delay:.2f} minutes")
    
    if predicted_delay < 20:
        st.info("🟢 Wow, this train is expected to run mostly on-time!")
    elif predicted_delay < 40:
        st.warning("🟡 Expect a minor delay. Pack a few extra snacks!")
    else:
        st.error("🔴 Heavy delay expected on this route!")
