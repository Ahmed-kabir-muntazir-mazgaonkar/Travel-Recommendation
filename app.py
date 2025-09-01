import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Travel Destination Recommender",
    page_icon="✈️",
    layout="wide"
)

# Load and cache data
@st.cache_data
def load_data():
    destinations_df = pd.read_csv("Expanded_Destinations.csv")
    reviews_df = pd.read_csv("Final_Updated_Expanded_Reviews.csv")
    userhistory_df = pd.read_csv("Final_Updated_Expanded_UserHistory.csv")
    users_df = pd.read_csv("Final_Updated_Expanded_Users.csv")
    
    # Merge datasets
    reviews_destinations = pd.merge(reviews_df, destinations_df, on='DestinationID', how='inner')
    reviews_destinations_userhistory = pd.merge(reviews_destinations, userhistory_df, on='UserID', how='inner')
    df = pd.merge(reviews_destinations_userhistory, users_df, on='UserID', how='inner')
    df.drop_duplicates(inplace=True)
    
    return destinations_df, reviews_df, userhistory_df, users_df, df

# Load the data
destinations_df, reviews_df, userhistory_df, users_df, df = load_data()

# Load or train the model
@st.cache_resource
def load_model():
    try:
        # Try to load the pre-trained model
        with open('travel_recommender_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        # If model doesn't exist, train a new one
        # Prepare data for training
        features = ['Popularity', 'ExperienceRating', 'Rating', 'NumberOfAdults', 'NumberOfChildren']
        X = df[features]
        y = df['ExperienceRating']  # Using ExperienceRating as target
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Train a simple model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Save the model
        with open('travel_recommender_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        return model

model = load_model()

# Content-based recommendation function
def content_based_recommendation(destination_id, destinations_df, n_recommendations=5):
    # Create feature matrix
    features = destinations_df[['Type', 'State', 'BestTimeToVisit']]
    features_combined = features['Type'] + ' ' + features['State'] + ' ' + features['BestTimeToVisit']
    
    # Vectorize features
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(features_combined)
    
    # Calculate similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get similarity scores for the destination
    sim_scores = list(enumerate(cosine_sim[destination_id]))
    
    # Sort destinations based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the n most similar destinations
    sim_scores = sim_scores[1:n_recommendations+1]
    
    # Get the destination indices
    destination_indices = [i[0] for i in sim_scores]
    
    # Return the top n most similar destinations
    return destinations_df.iloc[destination_indices]

# Collaborative filtering recommendation function
def collaborative_filtering_recommendation(user_id, n_recommendations=5):
    # Get user preferences
    user_preferences = users_df[users_df['UserID'] == user_id]['Preferences'].values[0]
    
    # Find similar users based on preferences
    users_df['Preferences'] = users_df['Preferences'].fillna('')
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(users_df['Preferences'])
    
    # Calculate similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get similarity scores for the user
    user_index = users_df[users_df['UserID'] == user_id].index[0]
    sim_scores = list(enumerate(cosine_sim[user_index]))
    
    # Sort users based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the n most similar users
    sim_scores = sim_scores[1:n_recommendations+1]
    
    # Get the user indices
    user_indices = [i[0] for i in sim_scores]
    
    # Get destinations liked by similar users
    similar_users_destinations = []
    for idx in user_indices:
        similar_user_id = users_df.iloc[idx]['UserID']
        user_destinations = userhistory_df[userhistory_df['UserID'] == similar_user_id]['DestinationID'].values
        similar_users_destinations.extend(user_destinations)
    
    # Get unique destinations and their counts
    unique_destinations = pd.Series(similar_users_destinations).value_counts().index.tolist()
    
    # Filter out destinations the user has already visited
    visited_destinations = userhistory_df[userhistory_df['UserID'] == user_id]['DestinationID'].values
    recommended_destinations = [d for d in unique_destinations if d not in visited_destinations][:n_recommendations]
    
    # Return recommended destinations
    return destinations_df[destinations_df['DestinationID'].isin(recommended_destinations)]

# Streamlit app
def main():
    st.title("✈️ Travel Destination Recommendation System")
    
    # Sidebar
    st.sidebar.header("Navigation")
    app_mode = st.sidebar.selectbox("Choose a page", 
                                   ["Home", "Data Overview", "Destination Explorer", 
                                    "Recommendations", "User Analysis", "About"])
    
    if app_mode == "Home":
        show_home()
    elif app_mode == "Data Overview":
        show_data_overview()
    elif app_mode == "Destination Explorer":
        show_destination_explorer()
    elif app_mode == "Recommendations":
        show_recommendations()
    elif app_mode == "User Analysis":
        show_user_analysis()
    elif app_mode == "About":
        show_about()

def show_home():
    st.header("Welcome to the Travel Destination Recommendation System")
    
    st.markdown("""
    This application helps you discover amazing travel destinations based on:
    - Your preferences and travel history
    - Popular destinations and ratings
    - Similar users' preferences
    
    ### Features:
    - **Data Overview**: Explore the datasets used in the system
    - **Destination Explorer**: Browse through all available destinations
    - **Recommendations**: Get personalized destination recommendations
    - **User Analysis**: Understand user preferences and behaviors
    
    ### How to use:
    1. Navigate through the different sections using the sidebar
    2. Explore destinations and their details
    3. Get personalized recommendations based on your preferences
    """)
    
    # Show some statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Destinations", len(destinations_df))
    
    with col2:
        st.metric("Total Users", len(users_df))
    
    with col3:
        st.metric("Total Reviews", len(reviews_df))
    
    # Show popular destinations
    st.subheader("Most Popular Destinations")
    popular_destinations = destinations_df.sort_values(by='Popularity', ascending=False).head(5)
    st.dataframe(popular_destinations[['Name', 'Type', 'State', 'Popularity']])

def show_data_overview():
    st.header("Data Overview")
    
    dataset_option = st.selectbox("Select Dataset to View", 
                                ["Destinations", "Reviews", "User History", "Users", "Merged Data"])
    
    if dataset_option == "Destinations":
        st.dataframe(destinations_df)
        st.subheader("Destination Types Distribution")
        fig, ax = plt.subplots()
        destinations_df['Type'].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)
        
    elif dataset_option == "Reviews":
        st.dataframe(reviews_df.head())
        
    elif dataset_option == "User History":
        st.dataframe(userhistory_df.head())
        
    elif dataset_option == "Users":
        st.dataframe(users_df.head())
        
    elif dataset_option == "Merged Data":
        st.dataframe(df.head())

def show_destination_explorer():
    st.header("Destination Explorer")
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        destination_type = st.multiselect("Filter by Type", destinations_df['Type'].unique())
    
    with col2:
        state_filter = st.multiselect("Filter by State", destinations_df['State'].unique())
    
    # Apply filters
    filtered_destinations = destinations_df.copy()
    
    if destination_type:
        filtered_destinations = filtered_destinations[filtered_destinations['Type'].isin(destination_type)]
    
    if state_filter:
        filtered_destinations = filtered_destinations[filtered_destinations['State'].isin(state_filter)]
    
    # Display filtered destinations
    st.dataframe(filtered_destinations)
    
    # Show destination details when selected
    if len(filtered_destinations) > 0:
        selected_destination = st.selectbox("Select a destination to view details", 
                                          filtered_destinations['Name'].values)
        
        if selected_destination:
            destination_details = filtered_destinations[filtered_destinations['Name'] == selected_destination].iloc[0]
            
            st.subheader(f"Details for {selected_destination}")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Type:** {destination_details['Type']}")
                st.write(f"**State:** {destination_details['State']}")
                st.write(f"**Popularity:** {destination_details['Popularity']}")
                
            with col2:
                st.write(f"**Best Time to Visit:** {destination_details['BestTimeToVisit']}")
                st.write(f"**Destination ID:** {destination_details['DestinationID']}")
            
            # Show similar destinations
            st.subheader("Similar Destinations")
            similar_destinations = content_based_recommendation(destination_details['DestinationID'] - 1, destinations_df)
            st.dataframe(similar_destinations[['Name', 'Type', 'State', 'Popularity']])

def show_recommendations():
    st.header("Personalized Recommendations")
    
    recommendation_type = st.radio("Select Recommendation Type", 
                                  ["Content-Based", "Collaborative Filtering", "Hybrid"])
    
    if recommendation_type == "Content-Based":
        st.subheader("Content-Based Recommendations")
        
        # Let user select a destination they like
        base_destination = st.selectbox("Select a destination you like", 
                                      destinations_df['Name'].values)
        
        if base_destination:
            destination_id = destinations_df[destinations_df['Name'] == base_destination]['DestinationID'].values[0] - 1
            recommendations = content_based_recommendation(destination_id, destinations_df)
            
            st.write(f"Destinations similar to {base_destination}:")
            st.dataframe(recommendations[['Name', 'Type', 'State', 'Popularity']])
    
    elif recommendation_type == "Collaborative Filtering":
        st.subheader("Collaborative Filtering Recommendations")
        
        # Let user select their user ID
        user_id = st.selectbox("Select your User ID", users_df['UserID'].values)
        
        if user_id:
            recommendations = collaborative_filtering_recommendation(user_id)
            
            if len(recommendations) > 0:
                st.write(f"Recommended destinations for user {user_id} based on similar users:")
                st.dataframe(recommendations[['Name', 'Type', 'State', 'Popularity']])
            else:
                st.write("No recommendations available. Try exploring more destinations!")
    
    elif recommendation_type == "Hybrid":
        st.subheader("Hybrid Recommendations")
        
        # Let user select their user ID and a destination they like
        user_id = st.selectbox("Select your User ID", users_df['UserID'].values, key="hybrid_user")
        base_destination = st.selectbox("Select a destination you like", 
                                      destinations_df['Name'].values, key="hybrid_dest")
        
        if user_id and base_destination:
            # Get content-based recommendations
            destination_id = destinations_df[destinations_df['Name'] == base_destination]['DestinationID'].values[0] - 1
            content_recs = content_based_recommendation(destination_id, destinations_df, 10)
            
            # Get collaborative filtering recommendations
            collab_recs = collaborative_filtering_recommendation(user_id, 10)
            
            # Combine recommendations (simple hybrid approach)
            hybrid_recs = pd.concat([content_recs, collab_recs]).drop_duplicates().head(5)
            
            st.write("Hybrid recommendations:")
            st.dataframe(hybrid_recs[['Name', 'Type', 'State', 'Popularity']])

def show_user_analysis():
    st.header("User Analysis")
    
    # User preferences analysis
    st.subheader("User Preferences Distribution")
    
    # Extract all preferences
    all_preferences = []
    for prefs in users_df['Preferences'].dropna():
        all_preferences.extend([pref.strip() for pref in prefs.split(',')])
    
    pref_counts = pd.Series(all_preferences).value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    pref_counts.plot(kind='bar', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # User demographics
    st.subheader("User Demographics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Gender Distribution")
        gender_counts = users_df['Gender'].value_counts()
        fig, ax = plt.subplots()
        gender_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)
    
    with col2:
        st.write("Number of Adults Distribution")
        adults_counts = users_df['NumberOfAdults'].value_counts()
        fig, ax = plt.subplots()
        adults_counts.plot(kind='bar', ax=ax)
        st.pyplot(fig)
    
    # Rating analysis
    st.subheader("Rating Analysis")
    
    rating_counts = reviews_df['Rating'].value_counts().sort_index()
    fig, ax = plt.subplots()
    rating_counts.plot(kind='bar', ax=ax)
    st.pyplot(fig)

def show_about():
    st.header("About This Application")
    
    st.markdown("""
    This Travel Destination Recommendation System is built using:
    - Python for data processing and machine learning
    - Streamlit for the web interface
    - Scikit-learn for recommendation algorithms
    
    ### Features:
    - Content-based filtering based on destination attributes
    - Collaborative filtering based on user preferences
    - Hybrid recommendation system
    
    ### Data Sources:
    The system uses four main datasets:
    1. Destinations data with details about various travel destinations
    2. Reviews data with user ratings and comments
    3. User history data with visit information
    4. Users data with demographic information and preferences
    
    ### How it works:
    1. The system analyzes your preferences and travel history
    2. It finds similar users or destinations based on various features
    3. It provides personalized recommendations based on this analysis
    
    For more information or to contribute to this project, please contact us.
    """)

if __name__ == "__main__":
    main()
