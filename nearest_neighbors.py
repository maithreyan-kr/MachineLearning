import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def KNN_test(X_train, Y_train, X_test, Y_test, K):
    predictions = []

    for x_test in X_test:
        # Calculate distances to all training examples
        distances = [euclidean_distance(x_test, x_train) for x_train in X_train]

        # Get indices of K-nearest neighbors
        nearest_indices = np.argsort(distances)[:K]

        # Get labels of K-nearest neighbors
        nearest_labels = Y_train[nearest_indices]

        # Predict the label by taking the majority vote
        prediction = np.sign(np.sum(nearest_labels))
        predictions.append(prediction)

    # Calculate accuracy
    correct_predictions = np.sum(predictions == Y_test)
    total_predictions = len(Y_test)
    accuracy = correct_predictions / total_predictions
    return accuracy

def calculate_accuracy(Y_true, Y_pred):
    return np.mean(Y_true == Y_pred)

def choose_K(X_train, Y_train, X_val, Y_val, max_K):
    best_accuracy = 0
    best_K = 0

    for K in range(1, max_K + 1):
        Y_val_pred = []

        for x_val in X_val:
            # Calculate distances from the validation point to all training points
            distances = np.sqrt(np.sum((X_train - x_val) ** 2, axis=1))

            # Get the indices of the K-nearest neighbors
            nearest_indices = np.argsort(distances)[:K]

            # Get the labels of the K-nearest neighbors
            nearest_labels = Y_train[nearest_indices]

            # Predict the label by taking the majority vote
            prediction = np.sign(np.sum(nearest_labels))
            Y_val_pred.append(prediction)

        accuracy = calculate_accuracy(Y_val, np.array(Y_val_pred))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_K = K

    return best_K

# Sample to test both the functions.....

# Example training data and labels
X_train = np.random.rand(100, 2)  # 100 samples with 2 features
Y_train = np.random.choice([-1, 1], 100)  # Binary labels

# Example validation data and labels
X_val = np.random.rand(20, 2)  # 20 samples with 2 features
Y_val = np.random.choice([-1, 1], 20)  # Binary labels

max_K = 10  # Set your desired maximum K value
best_K = choose_K(X_train, Y_train, X_val, Y_val, max_K)
print(f'Best K value: {best_K}')


# Read the training data CSV file
X_train = []
Y_train = []

with open('nearest_neighbors_2.csv', 'r') as file:
    lines = file.readlines()
    for line in lines[1:]:  # Skip the header row
        data = line.strip().split(',')
        features = list(map(float, data[:-1]))
        label = int(data[-1])
        X_train.append(features)
        Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Read the test data CSV file
X_test = []
Y_test = []

with open('nearest_neighbors_2.csv', 'r') as file:
    lines = file.readlines()
    for line in lines[1:]:  # Skip the header row
        data = line.strip().split(',')
        features = list(map(float, data[:-1]))
        label = int(data[-1])
        X_test.append(features)
        Y_test.append(label)

X_test = np.array(X_test)
Y_test = np.array(Y_test)

# K is the number of neighbors you want to consider
K = 3

# Calculate accuracy
accuracy = KNN_test(X_train, Y_train, X_test, Y_test, K)
print(f'Accuracy on test data when K = {K} : {accuracy * 100:.2f}%')

