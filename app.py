import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import random

# Set page configuration
st.set_page_config(
    page_title="Travel Recommendation System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #FF6347;
        margin-bottom: 1rem;
    }
    .destination-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        background-color: #f8f9fa;
        border-left: 5px solid #1E90FF;
    }
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-size: 1rem;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #FF6347;
    }
</style>
""", unsafe_allow_html=True)

# Simulate model loading with error handling
@st.cache_resource
def load_model():
    try:
        # Try to load your actual model files
        model = pickle.load(open("model.pkl", 'rb'))
        label_encoders = pickle.load(open("label_encoders.pkl", 'rb'))
        return model, label_encoders, True
    except FileNotFoundError:
        # Create demo data if files not found
        st.warning("Model files not found. Using demo mode with sample data.")
        return None, None, False

# Generate sample data for demonstration
def generate_sample_data():
    # Sample destinations
    destinations = [
        {'DestinationID': 1, 'Name': 'Goa Beach', 'State': 'Goa', 'Type': 'Beach', 'Popularity': 4.5, 'BestTimeToVisit': 'November-February'},
        {'DestinationID': 2, 'Name': 'Manali', 'State': 'Himachal Pradesh', 'Type': 'Hill Station', 'Popularity': 4.3, 'BestTimeToVisit': 'March-June'},
        {'DestinationID': 3, 'Name': 'Jaipur', 'State': 'Rajasthan', 'Type': 'Historical', 'Popularity': 4.2, 'BestTimeToVisit': 'October-March'},
        {'DestinationID': 4, 'Name': 'Kerala Backwaters', 'State': 'Kerala', 'Type': 'Nature', 'Popularity': 4.6, 'BestTimeToVisit': 'September-March'},
        {'DestinationID': 5, 'Name': 'Varanasi', 'State': 'Uttar Pradesh', 'Type': 'Spiritual', 'Popularity': 4.1, 'BestTimeToVisit': 'October-March'},
        {'DestinationID': 6, 'Name': 'Darjeeling', 'State': 'West Bengal', 'Type': 'Hill Station', 'Popularity': 4.4, 'BestTimeToVisit': 'March-May'},
        {'DestinationID': 7, 'Name': 'Rishikesh', 'State': 'Uttarakhand', 'Type': 'Adventure', 'Popularity': 4.0, 'BestTimeToVisit': 'September-November'},
        {'DestinationID': 8, 'Name': 'Mysore', 'State': 'Karnataka', 'Type': 'Cultural', 'Popularity': 4.2, 'BestTimeToVisit': 'October-February'},
    ]
    
    # Sample user history
    user_history = []
    for user_id in range(1, 21):
        for dest_id in random.sample(range(1, 9), 5):
            user_history.append({
                'UserID': user_id,
                'DestinationID': dest_id,
                'ExperienceRating': random.uniform(3.5, 5.0)
            })
    
    return pd.DataFrame(destinations), pd.DataFrame(user_history)

# Collaborative filtering function
def collaborative_recommend(user_id, user_similarity, user_item_matrix, destinations_df, n_recommendations=5):
    if user_id > len(user_similarity):
        # If user ID is out of bounds, return popular destinations
        return destinations_df.nlargest(n_recommendations, 'Popularity')
    
    similar_users = user_similarity[user_id - 1]
    similar_users_idx = np.argsort(similar_users)[::-1][1:6]  # top 5 similar users
    
    # If no similar users found, return popular destinations
    if len(similar_users_idx) == 0:
        return destinations_df.nlargest(n_recommendations, 'Popularity')
    
    similar_user_ratings = user_item_matrix.iloc[similar_users_idx].mean(axis=0)
    recommended_destinations_ids = similar_user_ratings.sort_values(ascending=False).head(n_recommendations).index
    
    recommendations = destinations_df[destinations_df['DestinationID'].isin(recommended_destinations_ids)]
    return recommendations

# Content-based recommendation function
def content_based_recommend(user_input, destinations_df, n_recommendations=5):
    # Simple content-based filtering based on user preferences
    filtered_destinations = destinations_df.copy()
    
    # Filter by type if specified
    if user_input['Type'] != 'Any':
        filtered_destinations = filtered_destinations[filtered_destinations['Type'] == user_input['Type']]
    
    # Filter by state if specified
    if user_input['State'] != 'Any':
        filtered_destinations = filtered_destinations[filtered_destinations['State'] == user_input['State']]
    
    # Filter by best time to visit if specified
    if user_input['BestTimeToVisit'] != 'Any':
        current_month = datetime.now().strftime('%B')
        if user_input['BestTimeToVisit'] == current_month:
            filtered_destinations = filtered_destinations[
                filtered_destinations['BestTimeToVisit'].str.contains(current_month, case=False, na=False)
            ]
    
    # If no destinations match filters, return popular ones
    if len(filtered_destinations) == 0:
        return destinations_df.nlargest(n_recommendations, 'Popularity')
    
    # Return top recommendations by popularity
    return filtered_destinations.nlargest(n_recommendations, 'Popularity')

# Main app function
def main():
    st.markdown('<h1 class="main-header">🌍 Travel Recommendation System</h1>', unsafe_allow_html=True)
    
    # Load or generate sample data
    model, label_encoders, model_loaded = load_model()
    destinations_df, user_history_df = generate_sample_data()
    
    # Create user-item matrix for collaborative filtering
    user_item_matrix = user_history_df.pivot_table(
        index='UserID', 
        columns='DestinationID', 
        values='ExperienceRating', 
        fill_value=0
    )
    
    # Calculate user similarity
    user_similarity = cosine_similarity(user_item_matrix)
    
    # Sidebar for user input
    with st.sidebar:
        st.header("Plan Your Trip")
        
        user_id = st.number_input("User ID", min_value=1, max_value=20, value=1, 
                                 help="Enter a user ID between 1-20")
        
        name = st.text_input("Your Name", "John Doe")
        
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        with col2:
            adults = st.number_input("Number of Adults", min_value=1, max_value=10, value=2)
        
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        
        destination_type = st.selectbox(
            "Destination Type", 
            ["Any", "Beach", "Hill Station", "Historical", "Nature", "Spiritual", "Adventure", "Cultural"]
        )
        
        state = st.selectbox(
            "Preferred State", 
            ["Any", "Goa", "Himachal Pradesh", "Rajasthan", "Kerala", "Uttar Pradesh", 
             "West Bengal", "Uttarakhand", "Karnataka"]
        )
        
        best_time = st.selectbox(
            "Best Time to Visit", 
            ["Any", "January", "February", "March", "April", "May", "June", 
             "July", "August", "September", "October", "November", "December"]
        )
        
        preferences = st.multiselect(
            "Your Preferences",
            ["Relaxing", "Adventure", "Cultural", "Food", "Shopping", "Photography", "Wildlife"]
        )
        
        if st.button("Get Recommendations", type="primary"):
            st.session_state.get_recommendations = True
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["Recommendations", "Destination Explorer", "Travel Insights"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Personalized Recommendations</h2>', unsafe_allow_html=True)
        
        if st.session_state.get_recommendations:
            with st.spinner("Finding the best destinations for you..."):
                # Prepare user input
                user_input = {
                    'Name_x': name,
                    'Type': destination_type,
                    'State': state,
                    'BestTimeToVisit': best_time,
                    'Preferences': ', '.join(preferences) if preferences else 'Not specified',
                    'Gender': gender,
                    'NumberOfAdults': adults,
                    'NumberOfChildren': children
                }
                
                # Get recommendations
                if model_loaded:
                    # Use actual model if available
                    pass  # You would implement your model prediction here
                else:
                    # Use demo recommendations
                    collab_recommendations = collaborative_recommend(
                        user_id, user_similarity, user_item_matrix, destinations_df
                    )
                    content_recommendations = content_based_recommend(user_input, destinations_df)
                
                # Display collaborative filtering recommendations
                st.subheader("Based on Similar Travelers")
                if not collab_recommendations.empty:
                    for _, dest in collab_recommendations.iterrows():
                        with st.container():
                            st.markdown(f"""
                            <div class="destination-card">
                                <h3>{dest['Name']} ⭐ {dest['Popularity']}/5</h3>
                                <p><strong>Location:</strong> {dest['State']} | <strong>Type:</strong> {dest['Type']}</p>
                                <p><strong>Best Time to Visit:</strong> {dest['BestTimeToVisit']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No recommendations based on similar users. Try expanding your preferences.")
                
                # Display content-based recommendations
                st.subheader("Based on Your Preferences")
                if not content_recommendations.empty:
                    for _, dest in content_recommendations.iterrows():
                        with st.container():
                            st.markdown(f"""
                            <div class="destination-card">
                                <h3>{dest['Name']} ⭐ {dest['Popularity']}/5</h3>
                                <p><strong>Location:</strong> {dest['State']} | <strong>Type:</strong> {dest['Type']}</p>
                                <p><strong>Best Time to Visit:</strong> {dest['BestTimeToVisit']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No recommendations based on your preferences. Try expanding your criteria.")
        else:
            st.info("Please fill out the form on the left and click 'Get Recommendations' to see personalized suggestions.")
    
    with tab2:
        st.markdown('<h2 class="sub-header">Explore All Destinations</h2>', unsafe_allow_html=True)
        
        # Filters for destination explorer
        col1, col2, col3 = st.columns(3)
        with col1:
            explore_type = st.selectbox(
                "Filter by Type", 
                ["All", "Beach", "Hill Station", "Historical", "Nature", "Spiritual", "Adventure", "Cultural"],
                key="explore_type"
            )
        with col2:
            explore_state = st.selectbox(
                "Filter by State", 
                ["All", "Goa", "Himachal Pradesh", "Rajasthan", "Kerala", "Uttar Pradesh", 
                 "West Bengal", "Uttarakhand", "Karnataka"],
                key="explore_state"
            )
        with col3:
            min_popularity = st.slider("Minimum Popularity", 3.0, 5.0, 3.5, 0.1)
        
        # Filter destinations
        filtered_destinations = destinations_df.copy()
        if explore_type != "All":
            filtered_destinations = filtered_destinations[filtered_destinations['Type'] == explore_type]
        if explore_state != "All":
            filtered_destinations = filtered_destinations[filtered_destinations['State'] == explore_state]
        filtered_destinations = filtered_destinations[filtered_destinations['Popularity'] >= min_popularity]
        
        # Display filtered destinations
        if not filtered_destinations.empty:
            for _, dest in filtered_destinations.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="destination-card">
                        <h3>{dest['Name']} ⭐ {dest['Popularity']}/5</h3>
                        <p><strong>Location:</strong> {dest['State']} | <strong>Type:</strong> {dest['Type']}</p>
                        <p><strong>Best Time to Visit:</strong> {dest['BestTimeToVisit']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No destinations match your filters. Try adjusting your criteria.")
    
    with tab3:
        st.markdown('<h2 class="sub-header">Travel Insights & Trends</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Popularity by destination type
            st.subheader("Popularity by Destination Type")
            type_popularity = destinations_df.groupby('Type')['Popularity'].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=type_popularity.values, y=type_popularity.index, ax=ax, palette="viridis")
            ax.set_xlabel("Average Popularity Rating")
            ax.set_ylabel("Destination Type")
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Destination distribution by state
            st.subheader("Destinations by State")
            state_counts = destinations_df['State'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=state_counts.values, y=state_counts.index, ax=ax, palette="magma")
            ax.set_xlabel("Number of Destinations")
            ax.set_ylabel("State")
            plt.tight_layout()
            st.pyplot(fig)
        
        # Popularity distribution
        st.subheader("Popularity Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(destinations_df['Popularity'], bins=10, kde=True, ax=ax)
        ax.set_xlabel("Popularity Rating")
        ax.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)

# Initialize session state
if 'get_recommendations' not in st.session_state:
    st.session_state.get_recommendations = False

# Run the app
if __name__ == "__main__":
    main()
