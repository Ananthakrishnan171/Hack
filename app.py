import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================
# 🔁 Load the trained model
# =========================
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# =========================
# 🎨 Streamlit App Layout
# =========================
st.set_page_config(page_title="Lifestyle Expense Predictor", layout="centered")
st.title("💰 Monthly Expense Predictor")
st.write("Estimate your monthly expenses based on your lifestyle habits.")

# =========================
# 📝 User Inputs
# =========================
outings = st.slider("Number of outings per month", 0, 30, 5)
shopping = st.slider("Monthly shopping frequency", 0, 30, 4)
recharge = st.number_input("Mobile/data recharge cost (₹)", 0, 5000, 300)
food = st.slider("Food ordering frequency per month", 0, 50, 6)
rent = st.number_input("Rent or hostel fee (₹)", 0, 30000, 5000)

# =========================
# 📌 Predict Button
# =========================
if st.button("Predict Monthly Expense"):
    input_data = np.array([[outings, shopping, recharge, food, rent]])
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Monthly Expense: ₹{prediction:.2f}")
