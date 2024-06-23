import streamlit as st
import pickle
import os
import pandas as pd
import numpy as np

st.title('Kwamena Duker Fifa Prediction')

# Specify the path to the model pickle file
model_path = r"/Users/kwamenaduker/AI/XGBmodel_gs_model.pkl"

# Check if the file exists
if not os.path.isfile(model_path):
    st.error(f"The model file at {model_path} does not exist. Please check the path.")
else:
    # Load the model
    with open(model_path, 'rb') as file:
        md = pickle.load(file)

    # Input fields
    movement_reactions = st.number_input('movement_reactions')
    mentality_composure = st.number_input('mentality_composure')
    potential = st.number_input('potential')
    wage_eur = st.number_input('wage_eur')
    release_clause_eur = st.number_input('release_clause_eur')
    value_eur = st.number_input('value_eur')
    passing = st.number_input('passing')
    attacking_short_passing = st.number_input('attacking_short_passing')
    mentality_vision = st.number_input('mentality_vision')
    international_reputation = st.number_input('international_reputation')
    skill_long_passing = st.number_input('skill_long_passing')
    power_shot_power = st.number_input('power_shot_power')
    physic = st.number_input('physic')
    age = st.number_input('age')
    skill_ball_control = st.number_input('skill_ball_control')

    # Predict button
    if st.button('Predict'):
        prediction = md.predict([[potential, value_eur, wage_eur, age, international_reputation, release_clause_eur, passing, physic,
                                  attacking_short_passing, skill_long_passing, skill_ball_control, movement_reactions, power_shot_power,
                                  mentality_vision, mentality_composure]])
        st.write("The predicted overall for your player is ", prediction[0])
