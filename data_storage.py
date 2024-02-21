import numpy as np

def build_nparray(dataArray):
    # Extract headers and data points
    headers = dataArray[0]
    dataRow = dataArray[1:]

    # Separate features and labels
    features = []  # declaring features as empty array
    labels = []   # declaring lable's as an empty array  

    for row in dataRow:
        # Convert feature values to float
        feature_row = [float(value) for value in row[:-1]]
        features.append(feature_row)

        # Convert label to int
        label = float(row[-1])
        labels.append(label)

    # Convert the lists to NumPy arrays
    feature_array = np.array(features, dtype=float)
    label_array = np.array(labels, dtype=int)

    return feature_array, label_array

def build_list(data):
    # Extract headers and data points
    headers = data[0]
    data = data[1:]

    # Separate features and labels
    featuresA = []
    labelsA = []

    for row in data:
        # Convert feature values to float and append as a list
        feature_row = [float(value) for value in row[:-1]]
        featuresA.append(feature_row)

        # Convert label to int and append to labels list
        label = int(row[-1])
        labelsA.append(label)

    return featuresA, labelsA



def build_dict(data):
    # Extract headers and data points
    headers = data[0]
    data = data[1:]

    # Initialize dictionaries for features and labels
    feature_dicts = []
    label_dict = {}

    for row_index, row in enumerate(data):
        # Initialize a dictionary for each data point
        data_point = {}

        for i, value in enumerate(row[:-1]):
            feature_name = headers[i]
            feature_value = float(value)
            data_point[feature_name] = feature_value

        # Append the data point dictionary to the list of feature dictionaries
        feature_dicts.append(data_point)

        # Convert label to int and use the index as the key in the label dictionary
        label_index = row_index
        label_value = int(row[len(headers) - 1])
        label_dict[label_index] = label_value

    return feature_dicts, label_dict

