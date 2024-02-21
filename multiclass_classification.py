import numpy as np
import helpers as hlp
import matplotlib.pyplot as plt

def distance_point_to_hyperplane(pt, w, b):
    w_norm = np.linalg.norm(w)
    dist = np.abs(np.dot(w, pt) + b) / w_norm
    return dist

def svm_test_multiclass(W, B, x):

    num_classes = len(W)

    distances = np.zeros(num_classes)

    for i in range(num_classes):
        distances[i] = np.dot(W[i], x) + B[i]


    if np.all(distances <= 0):
        return -1  

    predicted_class = np.argmax(distances) + 1  

    if np.sum(distances == distances[predicted_class - 1]) > 1:
        return -1  

    return predicted_class

def compute_margin(data, w, b, support_vectors=None):
    X = data[:, :2]
    y = data[:, 2]

    # calculate margin
    w_norm = np.linalg.norm(w)
    margin = 2 / w_norm

    if support_vectors is not None:
        support_margin = [distance_point_to_hyperplane(pt, w, b) for pt in support_vectors]
        return margin, support_margin

    return margin, None

def svm_train_brute(data, labels):

    labels = np.array(labels)
    if np.isscalar(labels):
        labels = np.array([labels])

    # initialize
    max_margin = 0
    best_w = None
    best_b = None
    best_support_vectors = None

    for i in range(len(data)):
        for j in range(len(data)):
            if i == j or labels[j] == labels[i]:
                continue

            w = data[j, :2] - data[i, :2]
            b = 0.5 * (np.dot(data[i, :2], data[i, :2]) -
                       np.dot(data[j, :2], data[j, :2]))
            margin = compute_margin(data, w, b)

            if margin > max_margin:
                max_margin = margin
                best_w = w
                best_b = b
                best_support_vectors = [data[i, :2], data[j, :2]]

    return best_w, best_b, np.array(best_support_vectors)

def svm_train_multiclass(training_data):
   
    data = training_data[0]
    Y = training_data[1]

    num_classes = np.max(Y)

    W = []
    B = []

    
    for i in range(1, num_classes + 1):
       
        binary_labels = np.array(Y == i, dtype=int)

        w, b = svm_train_brute(data, binary_labels)

        W.append(w)
        B.append(b)

    return np.array(W), np.array(B)

def plot_decision_boundary(data, W, B, support_vectors=True):
    plt.scatter(data[data[:, 2] == 1, 0], data[data[:, 2] == 1, 1], marker='o', label='Class 1', color='blue')
    plt.scatter(data[data[:, 2] == 2, 0], data[data[:, 2] == 2, 1], marker='o', label='Class 2', color='red')

    for i in range(len(W)):
        w = W[i]
        b = B[i]

        xx = np.linspace(min(data[:, 0]), max(data[:, 0]), 100)
        yy = (-w[0] / w[1]) * xx - b / w[1]

        plt.plot(xx, yy, '--', label=f'Decision Boundary Class {i + 1}')

        if support_vectors:
            plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap=plt.cm.Paired, edgecolors='k', marker='o')
            plt.scatter(data[:, 0], data[:, 1], c='none', edgecolors='k', s=150, marker='o', label='Support Vectors')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('SVM Decision Boundary and Support Vectors')
    plt.show()

num =2
training_data = hlp.generate_training_data_multi(num)
hlp.plot_training_data_multi(training_data[0])
W, B = svm_train_multiclass(training_data)
plot_decision_boundary(training_data[0], W, B)
best_w, best_b, support_vectors = svm_train_brute(training_data)
margin, support_margin = compute_margin(training_data[0], best_w, best_b, support_vectors)
for i in range(len(W)):
    w = W[i]
    b = B[i]

    margin, support_margin = compute_margin(training_data[0], w, b, support_vectors[i])

    print(f"Margin for Class {i + 1} vs Rest: {margin}")
    if support_margin:
        print(f"Support Vector Margins: {support_margin}")

    print()
