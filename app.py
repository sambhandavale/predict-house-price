# import streamlit as st
# import pickle
# import numpy as np

# # Load the trained model and list of top features
# with open('rf_top5_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# with open('top5_features.pkl', 'rb') as f:
#     top5_features = pickle.load(f)

# st.title("House Price Prediction")

# st.write("Enter the values for the following features:")

# # Create a dictionary to store user inputs
# user_input = {}

# # For each feature, create an input field.
# # You may want to adjust the type and default values based on your data.
# for feature in top5_features:
#     # Here we assume numeric input. For categorical features, change to st.selectbox etc.
#     user_input[feature] = st.number_input(f"{feature}", value=0.0)

# # When the user clicks the Predict button, make a prediction:
# if st.button("Predict Price"):
#     # Convert user input dictionary to numpy array in the same order as top5_features
#     input_array = np.array([user_input[feature] for feature in top5_features]).reshape(1, -1)
#     prediction = model.predict(input_array)[0]
#     st.success(f"Predicted House Price: ₹{prediction:,.2f}")

import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.title("House Price Prediction with Input Limits")

st.write("Enter the values for the features:")
#Assume top5_features are like:
#['living area', 'grade of the house', 'Postal Code', 'Built Year', 'living_area_renov']
#Adjust the feature list order as needed.
#Use input limits for specific features:

living_area = st.number_input("living area", min_value=100.0, max_value=20000.0, value=1000.0)
grade = st.number_input("grade of the house", min_value=0.0, max_value=10.0, value=5.0)
postal_code = st.number_input("Postal Code", min_value=100000, max_value=400000, value=150000)
built_year = st.number_input("Built Year", min_value=1800, max_value=2025, value=1990)
living_area_renov = st.number_input("living_area_renov", min_value=100.0, max_value=20000.0, value=1000.0)
#Create a dictionary in the same order as your training set features

user_input = {
'living area': living_area,
'grade of the house': grade,
'Postal Code': postal_code,
'Built Year': built_year,
'living_area_renov': living_area_renov
}

if st.button("Predict Price"):
    # Convert the dictionary into a DataFrame to ensure the feature names match
    input_df = pd.DataFrame([user_input])

    # Load your trained model and feature list
    with open('rf_top5_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Prediction
    prediction = model.predict(input_df)[0]  # Make sure input_df columns precisely match what the model was trained on
    st.success(f"Predicted House Price: ₹{prediction:,.2f}")