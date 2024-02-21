import numpy as np

# define nodes of binary tree
class TreeNode:
    def __init__(self, depth, max_depth):
        self.depth = depth
        self.max_depth = max_depth
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.label = None
    
# Implementing DT_train_binary function
def DT_train_binary(X, Y, max_depth):
    def information_gain(y):
        p1 = np.sum(y) / len(y)
        p0 = 1 - p1
        if p0 == 0 or p1 == 0:
            return 0
        return - (p0 * np.log2(p0) + p1 * np.log2(p1))

    # function to find the best split
    def find_best_split(X, Y):
        best_gain = 0
        best_feature_idx = None
        best_threshold = None

        for feature_idx in range(X.shape[1]):
            unique_values = np.unique(X[:, feature_idx])
            for threshold in unique_values:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = X[:, feature_idx] > threshold

                left_Y = Y[left_mask]
                right_Y = Y[right_mask]

                if len(left_Y) == 0 or len(right_Y) == 0:
                    continue

                gain = information_gain(Y) - (
                    len(left_Y) / len(Y) * information_gain(left_Y) +
                    len(right_Y) / len(Y) * information_gain(right_Y)
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold

    #building a decision tree
    #updating build_tree function
    def build_tree(X, Y, depth):
        if depth == max_depth or np.all(Y == Y[0]):
            node = TreeNode(depth, max_depth)
            node.label = Y[0]
            return node

        feature_idx, threshold = find_best_split(X, Y)

        if feature_idx is None:
            node = TreeNode(depth, max_depth)
            node.label = np.argmax(np.bincount(Y))
            return node

        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold

        node = TreeNode(depth, max_depth)
        node.feature_idx = feature_idx
        node.threshold = threshold
        node.left = build_tree(X[left_mask], Y[left_mask], depth + 1)
        node.right = build_tree(X[right_mask], Y[right_mask], depth + 1)

        return node

    return build_tree(X, Y, 0)

def DT_test_binary(X, Y, DT):
    def predict(node, sample):
        if node.label is not None:
            return node.label
        if sample[node.feature_idx] <= node.threshold:
            return predict(node.left, sample)
        else:
            return predict(node.right, sample)

    correct_predictions = 0
    total_samples = len(X)

    for i in range(total_samples):
        prediction = predict(DT, X[i])
        if prediction == Y[i]:
            correct_predictions += 1

    accuracy = correct_predictions / total_samples
    return accuracy

# Implementing DT_make_prediction function
def DT_make_prediction(x, DT):
    def predict(node, sample):
        if node.label is not None:
            return node.label
        if sample[node.feature_idx] <= node.threshold:
            return predict(node.left, sample)
        else:
            return predict(node.right, sample)

    return predict(DT, x)

# Implementing DT_train_real function
def DT_train_real(X, Y, max_depth=-1):
    decision_tree = TreeNode(max_depth=max_depth)
    decision_tree.fit(X, Y)
    return decision_tree.tree

# Implementing DT_test_real function
def DT_test_real(X, Y, DT):
    def predict_example(node, example):
        if isinstance(node, dict):
            feature_idx = node['feature']
            threshold = node['threshold']

            if example[feature_idx] <= threshold:
                return predict_example(node['left'], example)
            else:
                return predict_example(node['right'], example)
        else:
            return node  # Leaf node (class prediction)

    correct_predictions = 0

    for i in range(len(X)):
        prediction = predict_example(DT, X[i])
        if prediction == Y[i]:
            correct_predictions += 1

    accuracy = correct_predictions / len(X)
    return accuracy

# Implementing RF_build_random_forest function
def RF_build_random_forest(X, Y, max_depth, num_of_trees):
    forest = []
    tree_accuracies = []

    total_samples = len(X)
    subset_size = int(0.1 * total_samples)  # Random subset size (10% of data)

    for _ in range(num_of_trees):
        # Randomly select a subset of data
        random_indices = np.random.choice(total_samples, subset_size, replace=False)
        X_subset = X[random_indices]
        Y_subset = Y[random_indices]

        # Train a decision tree on the random subset
        tree = DT_train_binary(X_subset, Y_subset, max_depth)

        # Test the tree on the same subset for accuracy
        accuracy = DT_test_binary(X_subset, Y_subset, tree)

        # Store the tree and its accuracy
        forest.append(tree)
        tree_accuracies.append(accuracy)

    return forest, tree_accuracies

# Implementing RF_test_random_forest function
def RF_test_random_forest(X, Y, RF):
    num_samples = len(X)
    predictions = []

    for sample in X:
        tree_predictions = [DT_make_prediction(sample, tree) for tree in RF]
        final_prediction = np.bincount(tree_predictions).argmax()
        predictions.append(final_prediction)

    accuracy = np.sum(predictions == Y) / num_samples
    return accuracy