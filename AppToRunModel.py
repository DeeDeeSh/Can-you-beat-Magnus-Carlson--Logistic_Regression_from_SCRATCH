import streamlit as st
import math
import pandas as pd

# Example placeholder mean and std values
mean_B = 2676.285  # Elo Rating
std_B = 100.158
mean_C = 352.66   # GMs beaten
std_C = 170.546
mean_D = 2615.4  # Avg opponent rating
std_D = 83.287

# Example weights from training
w1 = 4.622946379781242  # No. of GMs Beaten
w2 = 1.2654035428938344  # Elo
w3 = 1.1113814781565543  # Avg Opponent Rating
b = -5.589256715675954

st.title("Can You Beat Magnus Carlsen? 🤯♟️")
st.caption("Model trained using synthetic data for demonstration. It CAN be imbalanced. Savdhani se istemal kare :) ")

# User inputs
c1 = st.number_input("Enter your Elo Rating:", min_value=600, max_value=3200, value=1500)
c2 = st.number_input("No. of GMs you have beaten:", min_value=0, max_value=2500, value=20)
c3 = st.number_input("Avg opponent rating (last 50 games):", min_value=600, max_value=3200, value=1500)

# Normalize features
X1 = (c2 - mean_C) / std_C
X2 = (c1 - mean_B) / std_B
X3 = (c3 - mean_D) / std_D

# Logistic Regression prediction
z = w1*X1 + w2*X2 + w3*X3 + b
pred = 1 / (1 + math.exp(-z))

# Display prediction
st.write(f"**Predicted chance: {pred:.2f}**")

# Interpretation
if pred >= 0.7:
    st.success("You will beat Magnus Carlsen most times ['o']!")
elif pred <0.7 and pred >= 0.5:
    st.info("You are at level of Magnus Carlsen!")
elif pred <0.5 and pred >0.3:
    st.info("Doing good — keep practicing!")
else:
    st.warning("Not at that level yet...Improvement needed")
# Bonus bar chart
st.subheader("Your Score vs Threshold")
st.bar_chart({"Score": [pred], "Magnus's level (0.5) ": [0.5]})
