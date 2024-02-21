import numpy as np
import matplotlib.pyplot as plt
import helpers as hlp

def distance_point_to_hyperplane(pt, w, b):
    w_norm = np.linalg.norm(w)
    dist = np.abs(np.dot(w, pt) + b) / w_norm
    return dist

def compute_margin(data, w, b):
    X = data[:, :2]
    y = data[:, 2]

    # calculate margin
    w_norm = np.linalg.norm(w)
    margin = 2 / w_norm

    return margin

def svm_test_brute(w, b, x):
     # calculate decision value
    decision_value = np.dot(w, x) + b

    # predict class based on sign of decision value
    y = np.sign(decision_value).astype(int)

    return y

def svm_train_brute(training_data):
    # initialise
    max_margin = 0
    best_w = None
    best_b = None
    best_support_vectors = None

    # iterate through all pairs of positive and negative samples
    for i in range(len(training_data)):
        for j in range(len(training_data)):
            if i == j or training_data[i, 2] == training_data[j, 2]:
                continue

            # calculate decision boundary parameters
            w = training_data[j, :2] - training_data[i, :2]
            b = 0.5 * (np.dot(training_data[i, :2], training_data[i, :2]) -
                       np.dot(training_data[j, :2], training_data[j, :2]))
            margin = compute_margin(training_data, w, b)

            if margin > max_margin:
                max_margin = margin
                best_w = w
                best_b = b
                best_support_vectors = [training_data[i, :2], training_data[j, :2]]

    return best_w, best_b, np.array(best_support_vectors)




num = 3
data = hlp.generate_training_data_binary(num)
hlp.plot_training_data_binary(data)

# ttain SVM and get decision boundary and support vectors
pt = 0
x=3
w, b, s = svm_train_brute(data)
distance = distance_point_to_hyperplane(pt, w, b)
margin = compute_margin(data, w, b)
y = svm_test_brute(w, b, x)

plt.scatter(data[data[:, 2] == 1][:, 0], data[data[:, 2] == 1][:, 1], label='Class 1', marker='o')
plt.scatter(data[data[:, 2] == -1][:, 0], data[data[:, 2] == -1][:, 1], label='Class -1', marker='x')
plt.scatter(s[:, 0], s[:, 1], c='red', label='Support Vectors', marker='s')


w_norm = np.linalg.norm(w)
decision_boundary_slope = w[0] / w[1]
decision_boundary_intercept = -b / w[1]
margin_multiplier = margin / (2 * w_norm)

# calculate points on decision boundary
x_vals = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 100)
y_vals = decision_boundary_slope * x_vals + decision_boundary_intercept
upper_margin_line = y_vals + margin_multiplier
lower_margin_line = y_vals - margin_multiplier

# plot decision boundary and margin lines
plt.plot(x_vals, y_vals, label='Decision Boundary', color='green', linestyle='solid')
plt.plot(x_vals, upper_margin_line, color='green', linestyle='dashed')
plt.plot(x_vals, lower_margin_line, color='green', linestyle='dashed')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
