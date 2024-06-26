import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Function to compute user-user similarity matrix
def compute_similarity_matrix(df):
    # Select relevant columns for similarity computation
    features = ['Bedtime_Preference', 'Wake_Up_Time_Preference', 'Planned_Study_Time', 
                'Private_Time_Requirements', 'Guest_Frequency_Preference', 
                'Attitude_towards_Roommate_Smoking', 'Ideal_Study_Environment_Description_encoded', 
                'Attitude_towards_Borrowing_Sharing', 'Description_of_Personal_Room_At_Home_encoded', 
                'Desired_Room_Attributes_encoded', 'Study_Time_Preference', 
                'Conflict_Handling_Method_encoded', 'Communication_Preference_with_Roommate', 
                'Age_normalized']

    # Compute similarity matrix using cosine similarity
    similarity_matrix = cosine_similarity(df[features])

    # Convert similarity matrix to dataframe for easier manipulation
    similarity_df = pd.DataFrame(similarity_matrix, index=df['id'], columns=df['id'])
    return similarity_df

# Function to get top N similar users with their similarity scores
def get_top_similar_users(similarity_df, user_id, N=5):
    # Get top N similar users (excluding the user itself)
    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:N+1]
    similar_users_list = list(similar_users.index)
    similar_scores_list = list(similar_users.values)
    return list(zip(similar_users_list, similar_scores_list))

# Function to recommend users within and outside the same cluster with scores
def recommend_users(df, similarity_df, user_id, N=3):
    user_cluster_label = df[df['id'] == user_id]['cluster_label'].values[0]
    
    # Get top similar users
    similar_users = get_top_similar_users(similarity_df, user_id, N*2)

    # Separate similar users into intra-cluster and inter-cluster
    intra_cluster_recommendations = []
    inter_cluster_recommendations = []

    for similar_user, score in similar_users:
        similar_user_cluster_label = df[df['id'] == similar_user]['cluster_label'].values[0]
        if similar_user_cluster_label == user_cluster_label:
            intra_cluster_recommendations.append((similar_user, score))
        else:
            inter_cluster_recommendations.append((similar_user, score))
        
        # Stop when we have enough recommendations
        if len(intra_cluster_recommendations) >= N and len(inter_cluster_recommendations) >= N:
            break

    # Ensure we have exactly N recommendations from each category
    intra_cluster_recommendations = intra_cluster_recommendations[:N]
    inter_cluster_recommendations = inter_cluster_recommendations[:N]

    return intra_cluster_recommendations, inter_cluster_recommendations

