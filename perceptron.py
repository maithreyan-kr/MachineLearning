import numpy as np

def perceptron_train(X, Y, max_iterations=1000, learning_rate=0.1):
    # Initialize weights and bias to zeros
    num_features = X.shape[1]
    w = np.zeros(num_features)
    b = 0

    for _ in range(max_iterations):
        misclassified = False
        for i in range(X.shape[0]):
            xi = X[i]
            yi = Y[i]

            # Calculate the activation
            activation = np.dot(w, xi) + b

            # Update weights and bias if misclassification occurs
            if yi * activation <= 0:
                w += learning_rate * yi * xi
                b += learning_rate * yi
                misclassified = True

        # If no misclassifications, exit the loop
        if not misclassified:
            break

    return w, b

# Example training data
X = np.array([[2, 3], [1, 4], [4, 1], [3, 3]])
Y = np.array([1, -1, 1, -1])

# Train the perceptron
w, b = perceptron_train(X, Y)

print("Weights:", w)
print("Bias:", b)


def perceptron_test(X_test, Y_test, w, b):
    num_samples = X_test.shape[0]
    correct_predictions = 0

    for i in range(num_samples):
        xi = X_test[i]
        yi = Y_test[i]

        # Calculate the predicted class (1 or -1)
        prediction = np.sign(np.dot(w, xi) + b)

        # Check if the prediction matches the true label
        if prediction == yi:
            correct_predictions += 1

    accuracy = correct_predictions / num_samples
    return accuracy

# Example testing data
X_test = np.array([[2, 3], [1, 4], [4, 1], [3, 3]])
Y_test = np.array([1, -1, 1, -1])

# Example perceptron weights and bias
w = np.array([0.5, -0.5])
b = -1.0

# Test the perceptron
accuracy = perceptron_test(X_test, Y_test, w, b)
print("Accuracy:", accuracy)

# To print decision boundary for perceptron_2.csv
def predict(x):
    return np.sign(np.dot(w, x) + b)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
step = 0.1
print("Decision Boundary :")
for y in np.arange(y_max, y_min, -step):
    for x in np.arange(x_min, x_max, step):
        prediction = predict([x, y])
        if prediction == 1:
            print("o", end=" ")  # Class 1
        else:
            print("x", end=" ")  # Class -1
    print()

