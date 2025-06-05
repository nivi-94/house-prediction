import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("House Price Prediction")

data = pd.read_csv("data.csv")
st.write(data)

x = data[['Area_in_Sqft']]
y = data['Price_in_Thousands']

model = LinearRegression()
model.fit(x, y)

area = st.number_input("Enter Area in Sqft:", 500.0, 5000.0, step=100.0)
prediction = model.predict([[area]])
st.write("Predicted Price (in Thousands):", prediction[0])
