import pandas as pd
import numpy as np
from scripts.similarity_calculation import cosine_similarity

from sklearn.neighbors import KNeighborsClassifier

def get_top_n_similar_users(user_id, user_similarity_df, cluster_labels, n=3):
    """
    Get the top N users most similar to the given user.

    Parameters:
    user_id (int): The target user ID.
    user_similarity_df (pandas.DataFrame): The user similarity DataFrame.
    cluster_labels (numpy.ndarray): Array of cluster labels for each user.
    n (int): Number of top similar users to return.

    Returns:
    pandas.DataFrame: DataFrame containing top N similar users with their cluster labels and similarity scores.
    """
    similar_users = user_similarity_df.iloc[user_id].sort_values(ascending=False)
    similar_users = similar_users.drop(user_id)  # Exclude the target user
    top_similar_users = similar_users.head(n)
    result = pd.DataFrame({
        'UserID': top_similar_users.index,
        'ClusterLabel': cluster_labels[top_similar_users.index],
        'SimilarityScore': top_similar_users.values
    })
    return result

def intra_recommendation(user_id, cluster_labels, user_similarity_df, n=3):
    """
    Get intra-cluster recommendations for the given user.

    Parameters:
    user_id (int): The target user ID.
    cluster_labels (numpy.ndarray): Array of cluster labels for each user.
    user_similarity_df (pandas.DataFrame): The user similarity DataFrame.
    n (int): Number of top similar users to return.

    Returns:
    pandas.DataFrame: DataFrame containing top N intra-cluster similar users with their cluster labels and similarity scores.
    """
    cluster_id = cluster_labels[user_id]
    same_cluster_users = np.where(cluster_labels == cluster_id)[0]
    similar_users = user_similarity_df.iloc[user_id, same_cluster_users].sort_values(ascending=False)
    similar_users = similar_users.drop(user_id)  # Exclude the target user
    top_similar_users = similar_users.head(n)
    result = pd.DataFrame({
        'UserID': top_similar_users.index,
        'ClusterLabel': cluster_labels[top_similar_users.index],
        'SimilarityScore': top_similar_users.values
    })
    return result

def inter_recommendation(user_id, user_similarity_df, cluster_labels, n=3):
    """
    Get inter-cluster recommendations for the given user.

    Parameters:
    user_id (int): The target user ID.
    user_similarity_df (pandas.DataFrame): The user similarity DataFrame.
    cluster_labels (numpy.ndarray): Array of cluster labels for each user.
    n (int): Number of top similar users to return.

    Returns:
    pandas.DataFrame: DataFrame containing top N inter-cluster similar users with their cluster labels and similarity scores.
    """
    return get_top_n_similar_users(user_id, user_similarity_df, cluster_labels, n)

def recommendation(user_data, cluster_labels, user_similarity_df, new_user_data=None, n=3):
    """
    Get recommendations for a user, handling both existing and newcomer cases.

    Parameters:
    user_data (numpy.ndarray): The user data matrix.
    cluster_labels (numpy.ndarray): Array of cluster labels for each user.
    user_similarity_df (pandas.DataFrame): The user similarity DataFrame.
    new_user_data (numpy.ndarray, optional): The data for the newcomer.
    n (int): Number of top similar users to return.

    Returns:
    pandas.DataFrame: DataFrame containing top N recommendations with their cluster labels and similarity scores, 
                      and the cluster label for the newcomer.
    """
    if new_user_data is not None:
        if len(new_user_data) != user_data.shape[1]:
            raise ValueError(f"new_user_data must have {user_data.shape[1]} features.")

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(user_data, cluster_labels)
        new_user_cluster = knn.predict([new_user_data])[0]

        extended_data = np.vstack([user_data, new_user_data])
        extended_similarity = cosine_similarity(extended_data)
        extended_similarity_df = pd.DataFrame(extended_similarity)

        new_user_id = len(user_data)
        intra_recommended_users = intra_recommendation(new_user_id, np.append(cluster_labels, new_user_cluster), extended_similarity_df, n)

        if intra_recommended_users.empty:
            inter_users = inter_recommendation(new_user_id, extended_similarity_df, np.append(cluster_labels, new_user_cluster), n)
            inter_users['NewUserClusterLabel'] = new_user_cluster
            return inter_users
        else:
            intra_users = intra_recommendation(new_user_id, np.append(cluster_labels, new_user_cluster), extended_similarity_df, n)
            intra_users['NewUserClusterLabel'] = new_user_cluster
            return intra_users
    else:
        user_id = user_data
        intra_recommended_users = intra_recommendation(user_id, cluster_labels, user_similarity_df, n)
        if intra_recommended_users.empty:
            inter_users = inter_recommendation(user_id, user_similarity_df, cluster_labels, n)
            inter_users['NewUserClusterLabel'] = cluster_labels[user_id]
            return inter_users
        else:
            intra_users = intra_recommendation(user_id, cluster_labels, user_similarity_df, n)
            intra_users['NewUserClusterLabel'] = cluster_labels[user_id]
            return intra_users