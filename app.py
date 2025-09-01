import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load files
model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), 'rb'))
label_encoders = pickle.load(open(os.path.join(BASE_DIR, "label_encoders.pkl"), 'rb'))
destinations_df = pd.read_csv(os.path.join(BASE_DIR, "Expanded_Destinations.csv"))
userhistory_df = pd.read_csv(os.path.join(BASE_DIR, "Final_Updated_Expanded_UserHistory.csv"))
df = pd.read_csv(os.path.join(BASE_DIR, "final_df.csv"))

# Collaborative filtering setup
user_item_matrix = userhistory_df.pivot(index='UserID', columns='DestinationID', values='ExperienceRating').fillna(0)
user_similarity = cosine_similarity(user_item_matrix)

# Streamlit UI
st.title("Travel Recommendation System")

user_id = st.number_input("Enter your User ID", min_value=1)
name = st.text_input("Name")
state = st.selectbox("State", df['State'].unique())
type_ = st.selectbox("Type", df['Type'].unique())
best_time = st.text_input("Best Time to Visit")
preferences = st.text_input("Preferences")
gender = st.selectbox("Gender", ['Male', 'Female'])
adults = st.number_input("Number of Adults", min_value=0)
children = st.number_input("Number of Children", min_value=0)

if st.button("Get Recommendations"):
    user_input = {
        'Name_x': name,
        'Type': type_,
        'State': state,
        'BestTimeToVisit': best_time,
        'Preferences': preferences,
        'Gender': gender,
        'NumberOfAdults': adults,
        'NumberOfChildren': children,
    }

    # Encode and predict
    encoded_input = {}
    for feature in user_input:
        if feature in label_encoders:
            encoded_input[feature] = label_encoders[feature].transform([user_input[feature]])[0]
        else:
            encoded_input[feature] = user_input[feature]
    input_df = pd.DataFrame([encoded_input])
    predicted_popularity = model.predict(input_df)[0]

    # Collaborative filtering
    similar_users = user_similarity[user_id - 1]
    similar_users_idx = np.argsort(similar_users)[::-1][1:6]
    similar_user_ratings = user_item_matrix.iloc[similar_users_idx].mean(axis=0)
    recommended_ids = similar_user_ratings.sort_values(ascending=False).head(5).index
    recommendations = destinations_df[destinations_df['DestinationID'].isin(recommended_ids)]

    st.subheader("Predicted Popularity")
    st.write(predicted_popularity)

    st.subheader("Recommended Destinations")
    st.dataframe(recommendations[['Name', 'State', 'Type', 'BestTimeToVisit']])
