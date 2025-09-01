import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

# 📁 Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 📦 Load model and encoders
try:
    model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), 'rb'))
    label_encoders = pickle.load(open(os.path.join(BASE_DIR, "label_encoders.pkl"), 'rb'))
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# 📊 Load datasets
try:
    destinations_df = pd.read_csv(os.path.join(BASE_DIR, "Expanded_Destinations.csv"))
    userhistory_df = pd.read_csv(os.path.join(BASE_DIR, "Final_Updated_Expanded_UserHistory.csv"))
    df = pd.read_csv(os.path.join(BASE_DIR, "final_df.csv"))
except Exception as e:
    st.error(f"Dataset loading error: {e}")
    st.stop()

# 🤝 Collaborative filtering setup
user_item_matrix = userhistory_df.pivot(index='UserID', columns='DestinationID', values='ExperienceRating').fillna(0)
user_similarity = cosine_similarity(user_item_matrix)

# 🔍 Recommendation functions
def collaborative_recommend(user_id):
    try:
        similar_users = user_similarity[user_id - 1]
        similar_users_idx = np.argsort(similar_users)[::-1][1:6]
        similar_user_ratings = user_item_matrix.iloc[similar_users_idx].mean(axis=0)
        recommended_ids = similar_user_ratings.sort_values(ascending=False).head(5).index
        recommendations = destinations_df[destinations_df['DestinationID'].isin(recommended_ids)][[
            'Name', 'State', 'Type', 'Popularity', 'BestTimeToVisit'
        ]]
        return recommendations
    except Exception as e:
        st.warning(f"Collaborative filtering failed: {e}")
        return pd.DataFrame()

def predict_popularity(user_input):
    try:
        encoded_input = {}
        for feature in user_input:
            if feature in label_encoders:
                encoded_input[feature] = label_encoders[feature].transform([user_input[feature]])[0]
            else:
                encoded_input[feature] = user_input[feature]
        input_df = pd.DataFrame([encoded_input])
        return model.predict(input_df)[0]
    except Exception as e:
        st.warning(f"Prediction failed: {e}")
        return "Unknown"

# 🌐 Streamlit UI
st.title("🌍 Travel Recommendation System")

with st.form("recommendation_form"):
    user_id = st.number_input("User ID", min_value=1, step=1)
    name = st.text_input("Name")
    state = st.selectbox("State", sorted(df['State'].dropna().unique()))
    type_ = st.selectbox("Type", sorted(df['Type'].dropna().unique()))
    best_time = st.text_input("Best Time to Visit")
    preferences = st.text_input("Preferences")
    gender = st.selectbox("Gender", ['Male', 'Female'])
    adults = st.number_input("Number of Adults", min_value=0, step=1)
    children = st.number_input("Number of Children", min_value=0, step=1)
    submitted = st.form_submit_button("Get Recommendations")

if submitted:
    user_input = {
        'Name_x': name,
        'State': state,
        'Type': type_,
        'BestTimeToVisit': best_time,
        'Preferences': preferences,
        'Gender': gender,
        'NumberOfAdults': adults,
        'NumberOfChildren': children,
    }

    st.subheader("📈 Predicted Popularity")
    popularity = predict_popularity(user_input)
    st.success(f"Predicted Popularity Score: {popularity}")

    st.subheader("🧭 Recommended Destinations")
    recommendations = collaborative_recommend(user_id)
    if not recommendations.empty:
        st.dataframe(recommendations)
    else:
        st.info("No recommendations found for this user.")
