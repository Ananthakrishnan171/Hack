import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ======================
# 📦 Load the trained model
# ======================
with open('expense_predictor.pkl', 'rb') as f:
    model = pickle.load(f)

# ======================
# 🎨 Page Config
# ======================
st.set_page_config(page_title="Lifestyle Expense Predictor", layout="centered")

st.title("💸 Monthly Lifestyle Expense Estimator")
st.markdown("Use the sidebar to input your lifestyle habits and estimate your monthly expenses.")

# ======================
# 🧾 Sidebar Inputs
# ======================
st.sidebar.header("📝 Lifestyle Inputs")

outings = st.sidebar.slider("Number of outings per month", 0, 30, 5)
shopping_freq = st.sidebar.selectbox("Shopping frequency", ['Rarely', 'Sometimes', 'Often', 'Very Often'])
recharge_cost = st.sidebar.selectbox("Monthly data recharge cost", ['<100', '100-300', '300-500', '>500'])
food_orders = st.sidebar.slider("Food ordering frequency per month", 0, 30, 5)
rent = st.sidebar.number_input("Rent or Hostel fee (₹)", min_value=0, value=5000)

# ======================
# 🔤 Manual Encoding (based on training)
# ======================
shop_map = {'Rarely': 0, 'Sometimes': 1, 'Often': 2, 'Very Often': 3}
recharge_map = {'<100': 0, '100-300': 1, '300-500': 2, '>500': 3}

# ======================
# 📌 Predict Button
# ======================
if st.sidebar.button("🔮 Predict Expense"):
    input_data = np.array([[outings, shop_map[shopping_freq], recharge_map[recharge_cost], food_orders, rent]])
    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Monthly Expense: ₹{prediction:.2f}")

    # ======================
    # 📊 Breakdown Pie Chart (Based on weights)
    # ======================
    # For demo purposes, assign percentage weights
    weights = {
        'Recharge': 0.10,
        'Shopping': 0.15,
        'Food Delivery': 0.20,
        'Outings': 0.10,
        'Rent': 0.45
    }

    values = [
        prediction * weights['Recharge'],
        prediction * weights['Shopping'],
        prediction * weights['Food Delivery'],
        prediction * weights['Outings'],
        prediction * weights['Rent']
    ]

    labels = list(weights.keys())

    # Show breakdown
    st.markdown("### 📊 Spending Breakdown")
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig)
