import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder
import csv

def read_data(file_path):
    """
    Read the CSV file and process the data.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pandas.DataFrame: The processed DataFrame.
    numpy.ndarray: The cluster labels.
    """
    data = []

    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        columns = next(csv_reader)  # Read the header
        for row in csv_reader:
            data.append([float(x) for x in row])

    # Convert data to a DataFrame
    df = pd.DataFrame(data, columns=columns)
    df = df.apply(pd.to_numeric)

    # Extract cluster labels from the last column
    cluster_labels = df.iloc[:, -1].values.astype(int)
    df = df.iloc[:, :-1]  # Drop the last column from the DataFrame

    return df, cluster_labels

