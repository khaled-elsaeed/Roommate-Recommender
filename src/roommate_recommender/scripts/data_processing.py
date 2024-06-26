import pandas as pd
import numpy as np
from data.mappings import ordinal_mappings, nominal_mappings

def read_data(file_path):
    """
    Read the CSV file and process the data.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pandas.DataFrame: The processed DataFrame.
    numpy.ndarray: The cluster labels.
    """
    # Read the CSV file using pandas
    df = pd.read_csv(file_path)

    # Ensure all columns are numeric
    df = df.apply(pd.to_numeric)

    return df


def standardize_age(age):
    mean_age = 19.64453125  # Replace with the actual mean from your dataset
    std_age = 1.27825301087223  # Replace with the actual standard deviation from your dataset

    normalized_age = (age - mean_age) / std_age
    return normalized_age

# Function to encode values based on mappings for selected features
def encode_selected_features(features, values):
    encoded_values_dict = {}
    
    for feature, value in zip(features, values):
        if feature in ordinal_mappings:
            mapping = ordinal_mappings[feature]
            if value in mapping:
                encoded_value = mapping[value]
                encoded_values_dict[feature] = encoded_value
            else:
                encoded_values_dict[feature] = None  # Handle unknown values if needed
        elif feature in nominal_mappings:
            mapping = nominal_mappings[feature]
            if value in mapping:
                encoded_value = mapping[value]
                encoded_values_dict[feature] = encoded_value
            else:
                encoded_values_dict[feature] = None  # Handle unknown values if needed
        elif feature == 'Age':
            normalized_age_value = standardize_age(value)
            encoded_values_dict['Age_normalized'] = normalized_age_value  # Age_normalized is already normalized
        else:
            encoded_values_dict[feature] = None  # Handle unknown features if needed
    
    return encoded_values_dict
