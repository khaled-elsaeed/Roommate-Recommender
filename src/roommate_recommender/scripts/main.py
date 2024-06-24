import os
import numpy as np
import pandas as pd
from scripts.data_processing import read_data
from scripts.similarity_calculation import compute_user_similarity
from scripts.recommendation_system import intra_recommendation, inter_recommendation, recommendation

# File path to the data
file_path = 'data/clustered/Roommates.csv'

# Read and process the data
df, cluster_labels = read_data(file_path)

# Compute user similarity
user_similarity_df = compute_user_similarity(df)

# Example: Get recommendations for an existing user (user_id = 1)
user_id = 1
print(f"Top 3 intra-cluster similar users to user {user_id}:")
print(intra_recommendation(user_id, cluster_labels, user_similarity_df, n=3))

print(f"Top 3 inter-cluster similar users to user {user_id}:")
print(inter_recommendation(user_id, user_similarity_df, cluster_labels, n=3))



3.0,1.0,1.0,0.0,1.0,2.0,2.0,2,2,0,0,0.2786342473088685 #61
2.0,3.0,3.0,2.0,1.0,2.0,0.0,2,2,1,0,0.2786342473088685 #62

# Example: Get recommendations for a newcomer
new_user_data = np.array([2.0,1.0,2.0,1.0,1.0,1.0,0.0,3,3,0,2,0,1.0624844375404108])  # Example new user data
print("Top 3 recommendations for the newcomer:")
print(recommendation(df.values, cluster_labels, user_similarity_df, new_user_data=new_user_data, n=3))
# print(df.iloc[94])
