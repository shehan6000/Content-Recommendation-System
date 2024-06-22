import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample data with ratings
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    'item_id': [101, 102, 103, 101, 104, 102, 103, 105, 104, 105],
    'rating': [5, 3, 4, 2, 5, 4, 5, 3, 4, 2]  # Ratings from 1 to 5
}

df = pd.DataFrame(data)
# Save the data to a CSV file
csv_file_path = 'user_ratings.csv'
df.to_csv(csv_file_path, index=False)

# Load the data from the CSV file
df = pd.read_csv(csv_file_path)

# Create user-item interaction matrix
interaction_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
print("User-Item Interaction Matrix:")
print(interaction_matrix)

# Compute cosine similarity between users
user_similarity = cosine_similarity(interaction_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=interaction_matrix.index, columns=interaction_matrix.index)
print("\nUser Similarity Matrix:")
print(user_similarity_df)

# Function to recommend items
def recommend_items(user_id, interaction_matrix, user_similarity_df, top_n=2):
    # Get the user's interactions
    user_interactions = interaction_matrix.loc[user_id]
    # Find similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index
    # Aggregate items from similar users
    item_scores = {}
    for similar_user in similar_users:
        if similar_user == user_id:
            continue
        similar_user_interactions = interaction_matrix.loc[similar_user]
        for item in similar_user_interactions.index:
            if similar_user_interactions[item] > 0 and user_interactions[item] == 0:
                if item not in item_scores:
                    item_scores[item] = 0
                item_scores[item] += similar_user_interactions[item] * user_similarity_df[user_id][similar_user]
    
    # Sort items by score
    recommended_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [item for item, score in recommended_items]

def get_user_recommendations():
    user_id = int(input("Enter your user ID: "))
    top_n = int(input("Enter the number of recommendations you want: "))
    recommended_items = recommend_items(user_id, interaction_matrix, user_similarity_df, top_n=top_n)
    print(f"\nRecommended Items for User {user_id}:")
    for item in recommended_items:
        print(item)

# Run the user interface
get_user_recommendations()
