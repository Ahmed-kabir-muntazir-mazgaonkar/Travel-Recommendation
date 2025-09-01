from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Define base directory dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and encoders safely
try:
    model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), 'rb'))
    label_encoders = pickle.load(open(os.path.join(BASE_DIR, "label_encoders.pkl"), 'rb'))
except FileNotFoundError:
    model = None
    label_encoders = {}
    print("Model or encoders not found. Please check file paths.")

# Load datasets
try:
    destinations_df = pd.read_csv(os.path.join(BASE_DIR, "Expanded_Destinations.csv"))
    userhistory_df = pd.read_csv(os.path.join(BASE_DIR, "Final_Updated_Expanded_UserHistory.csv"))
    df = pd.read_csv(os.path.join(BASE_DIR, "final_df.csv"))
except FileNotFoundError as e:
    print(f"Dataset missing: {e}")
    destinations_df = pd.DataFrame()
    userhistory_df = pd.DataFrame()
    df = pd.DataFrame()

# Collaborative Filtering Setup
if not userhistory_df.empty:
    user_item_matrix = userhistory_df.pivot(index='UserID', columns='DestinationID', values='ExperienceRating').fillna(0)
    user_similarity = cosine_similarity(user_item_matrix)
else:
    user_item_matrix = pd.DataFrame()
    user_similarity = np.array([])

# Recommendation Functions
def collaborative_recommend(user_id, user_similarity, user_item_matrix, destinations_df):
    try:
        similar_users = user_similarity[user_id - 1]
        similar_users_idx = np.argsort(similar_users)[::-1][1:6]
        similar_user_ratings = user_item_matrix.iloc[similar_users_idx].mean(axis=0)
        recommended_ids = similar_user_ratings.sort_values(ascending=False).head(5).index
        recommendations = destinations_df[destinations_df['DestinationID'].isin(recommended_ids)][[
            'DestinationID', 'Name', 'State', 'Type', 'Popularity', 'BestTimeToVisit'
        ]]
        return recommendations
    except Exception as e:
        print(f"Collaborative filtering error: {e}")
        return pd.DataFrame()

def recommend_destinations(user_input, model, label_encoders, features, data):
    try:
        encoded_input = {}
        for feature in features:
            if feature in label_encoders:
                encoded_input[feature] = label_encoders[feature].transform([user_input[feature]])[0]
            else:
                encoded_input[feature] = user_input[feature]
        input_df = pd.DataFrame([encoded_input])
        predicted_popularity = model.predict(input_df)[0]
        return predicted_popularity
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Unknown"

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_id = int(request.form['user_id'])
        user_input = {
            'Name_x': request.form['name'],
            'Type': request.form['type'],
            'State': request.form['state'],
            'BestTimeToVisit': request.form['best_time'],
            'Preferences': request.form['preferences'],
            'Gender': request.form['gender'],
            'NumberOfAdults': int(request.form['adults']),
            'NumberOfChildren': int(request.form['children']),
        }

        recommended_destinations = collaborative_recommend(user_id, user_similarity, user_item_matrix, destinations_df)
        predicted_popularity = recommend_destinations(user_input, model, label_encoders, list(user_input.keys()), df)

        return render_template('recommendation.html',
                               recommended_destinations=recommended_destinations,
                               predicted_popularity=predicted_popularity)
    except Exception as e:
        print(f"Form handling error: {e}")
        return "Error processing recommendation", 500

if __name__ == '__main__':
    app.run(debug=True)
