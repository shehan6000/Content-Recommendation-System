# Content Recommendation System

This project is a simple content recommendation system that suggests articles, videos, or music based on user preferences and behavior. The system uses collaborative filtering with a cosine similarity measure to recommend items to users.

## Features

- Uses user-item interaction data with ratings.
- Computes cosine similarity between users to find similar users.
- Recommends items based on interactions of similar users.
- Simple user interface for inputting user ID and getting recommendations.

## Dataset

The dataset is stored in a CSV file (`user_ratings.csv`) with the following columns:
- `user_id`: The ID of the user.
- `item_id`: The ID of the item (article, video, music).
- `rating`: The user's rating for the item (1 to 5).

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn

