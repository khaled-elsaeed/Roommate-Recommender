import pandas as pd
from scripts.data_processing import encode_selected_features
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier

# Load your clustered data (replace with your actual data loading method)
df = pd.read_csv('data/clustered/roommates.csv')

# Assign the next available unique ID to the new person
def get_next_id(df):
    max_id = df['id'].max()
    return max_id + 1

# Function to compute user-user similarity matrix
def compute_similarity_matrix(df):
    features = ['Bedtime_Preference', 'Wake_Up_Time_Preference', 'Planned_Study_Time', 
                'Private_Time_Requirements', 'Guest_Frequency_Preference', 
                'Attitude_towards_Roommate_Smoking', 'Ideal_Study_Environment_Description_encoded', 
                'Attitude_towards_Borrowing_Sharing', 'Description_of_Personal_Room_At_Home_encoded', 
                'Desired_Room_Attributes_encoded', 'Study_Time_Preference', 
                'Conflict_Handling_Method_encoded', 'Communication_Preference_with_Roommate', 
                'Age_normalized']
    similarity_matrix = cosine_similarity(df[features])
    similarity_df = pd.DataFrame(similarity_matrix, index=df['id'], columns=df['id'])
    return similarity_df

# Assign cluster to the new person using k-NN
def assign_cluster_knn(df, new_person_df, n_neighbors=5):
    features = ['Bedtime_Preference', 'Wake_Up_Time_Preference', 'Planned_Study_Time', 
                'Private_Time_Requirements', 'Guest_Frequency_Preference', 
                'Attitude_towards_Roommate_Smoking', 'Ideal_Study_Environment_Description_encoded', 
                'Attitude_towards_Borrowing_Sharing', 'Description_of_Personal_Room_At_Home_encoded', 
                'Desired_Room_Attributes_encoded', 'Study_Time_Preference', 
                'Conflict_Handling_Method_encoded', 'Communication_Preference_with_Roommate', 
                'Age_normalized']
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(df[features], df['cluster_label'])
    new_person_cluster_label = knn.predict(new_person_df[features])[0]
    return new_person_cluster_label

# Get top N similar users with their similarity scores
def get_top_similar_users(similarity_df, user_id, N=5):
    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:N+1]
    similar_users_list = list(similar_users.index)
    similar_scores_list = list(similar_users.values)
    return list(zip(similar_users_list, similar_scores_list))

# Recommend users within and outside the same cluster with scores
def recommend_users(df, similarity_df, user_id, user_cluster_label, N=3):
    similar_users = get_top_similar_users(similarity_df, user_id, N*2)
    intra_cluster_recommendations = []
    inter_cluster_recommendations = []

    for similar_user, score in similar_users:
        similar_user_cluster_label = df[df['id'] == similar_user]['cluster_label'].values[0]
        if similar_user_cluster_label == user_cluster_label:
            intra_cluster_recommendations.append((similar_user, score))
        else:
            inter_cluster_recommendations.append((similar_user, score))
        if len(intra_cluster_recommendations) >= N and len(inter_cluster_recommendations) >= N:
            break

    intra_cluster_recommendations = intra_cluster_recommendations[:N]
    inter_cluster_recommendations = inter_cluster_recommendations[:N]

    return intra_cluster_recommendations, inter_cluster_recommendations

# Main function to recommend users based on new or existing person
def recommend_for_user(df, user_data=None, user_id=None, N=3, n_neighbors=5):
    if user_data is not None:  # For a new person
        user_data['id'] = get_next_id(df)

        new_person_df = pd.DataFrame([user_data])
        df_with_new_person = pd.concat([df, new_person_df], ignore_index=True)
        new_person_cluster_label = assign_cluster_knn(df, new_person_df, n_neighbors)
        df_with_new_person.loc[df_with_new_person['id'] == user_data['id'], 'cluster_label'] = new_person_cluster_label
        similarity_df = compute_similarity_matrix(df_with_new_person)
        intra_cluster_recs, inter_cluster_recs = recommend_users(df_with_new_person, similarity_df, user_data['id'], new_person_cluster_label, N)
    elif user_id is not None:  # For an existing person
        similarity_df = compute_similarity_matrix(df)
        user_cluster_label = df[df['id'] == user_id]['cluster_label'].values[0]
        intra_cluster_recs, inter_cluster_recs = recommend_users(df, similarity_df, user_id, user_cluster_label, N)
    else:
        raise ValueError("Either user_data or user_id must be provided.")
    
    return intra_cluster_recs, inter_cluster_recs


# Example data to encode (subset of features)
features_subset = [
    'Bedtime_Preference', 'Wake_Up_Time_Preference', 'Planned_Study_Time', 'Private_Time_Requirements',
    'Guest_Frequency_Preference', 'Faculty', 'Attitude_towards_Roommate_Smoking',
    'Ideal_Study_Environment_Description_encoded', 'Attitude_towards_Borrowing_Sharing',
    'Description_of_Personal_Room_At_Home_encoded', 'Desired_Room_Attributes_encoded',
    'Study_Time_Preference', 'Conflict_Handling_Method_encoded', 'Communication_Preference_with_Roommate',
    'Age'
]

values_subset = [
    "Before midnight", "at any time", "Morning", "Significant", "Never",
    "Pharmacy", "No", "Very quiet", "Ask before sharing", "Neat",
    "Study oriented", "Total quiet", "Hint jokingly", "Face-to-face", 19
]

# Encode the subset of values
encoded_values_subset = encode_selected_features(features_subset, values_subset)



# For a new person
intra_cluster_recs_new, inter_cluster_recs_new = recommend_for_user(df, user_data=encoded_values_subset)
print("___________________________________________________________________________________")
print("Intra-cluster recommendations for new person with scores:")
print(intra_cluster_recs_new)

print("\nInter-cluster recommendations for new person with scores:")
print(inter_cluster_recs_new)

print("___________________________________________________________________________________")

# For an existing person
existing_user_id = 2  # Replace with the existing user ID you're interested in
intra_cluster_recs_existing, inter_cluster_recs_existing = recommend_for_user(df, user_id=existing_user_id)
print("Intra-cluster recommendations for existing person with scores:")
print(intra_cluster_recs_existing)

print("\nInter-cluster recommendations for existing person with scores:")
print(inter_cluster_recs_existing)
