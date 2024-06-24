import numpy as np
import pandas as pd

def cosine_similarity(matrix):
    """
    Calculate the cosine similarity for a given matrix.

    Parameters:
    matrix (numpy.ndarray): The input matrix.

    Returns:
    numpy.ndarray: Cosine similarity matrix.
    """
    norm_matrix = np.linalg.norm(matrix, axis=1, keepdims=True)
    similarity_matrix = np.dot(matrix, matrix.T) / (norm_matrix * norm_matrix.T)
    return similarity_matrix

def compute_user_similarity(df):
    """
    Compute the user similarity matrix.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.

    Returns:
    pandas.DataFrame: User similarity DataFrame.
    """
    user_similarity = cosine_similarity(df.values)
    user_similarity_df = pd.DataFrame(user_similarity)
    return user_similarity_df
