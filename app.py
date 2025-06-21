import streamlit as st
import numpy as np
import pickle

# Load trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit App Layout
st.set_page_config(page_title="Lifestyle Expense Predictor", layout="centered")
st.title("💸 Monthly Lifestyle Expense Estimator")

st.markdown("Estimate your monthly expenses based on your lifestyle habits.")

# Input fields
outings = st.slider("Number of outings per month", 0, 30, 5)
shopping_freq = st.selectbox("Shopping frequency", ['Rarely', 'Sometimes', 'Often', 'Very Often'])
recharge_cost = st.selectbox("Monthly data recharge cost", ['<100', '100-300', '300-500', '>500'])
food_orders = st.slider("Food ordering frequency per month", 0, 30, 5)
rent = st.number_input("Rent or Hostel fee (₹)", min_value=0, value=5000)

# Label Encoding Manual Map (should match training encoding)
shop_map = {'Rarely': 0, 'Sometimes': 1, 'Often': 2, 'Very Often': 3}
recharge_map = {'<100': 0, '100-300': 1, '300-500': 2, '>500': 3}

if st.button("Predict Expense"):
    input_data = np.array([[outings, shop_map[shopping_freq], recharge_map[recharge_cost], food_orders, rent]])
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Monthly Expense: ₹{prediction:.2f}")
