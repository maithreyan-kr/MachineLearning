import numpy as np
from sklearn.tree import DecisionTreeClassifier

def adaboost_train(X, Y, max_iter):
    n_samples = len(X)
    # sample weights as a Numpy array
    sample_weights = np.ones(n_samples) / n_samples  

    f = []  # declaring f[empty array] to store trained stumps
    alpha = []  # declaring alpha[empty array] to store calculated alpha values

    for _ in range(max_iter):
        # train a decision tree(stump) with the weighted dataset
        stump = DecisionTreeClassifier(max_depth=1)
        stump.fit(X, Y, sample_weight=sample_weights)

        # predictions using the trained stump
        predictions = stump.predict(X)

        # make sure data types match
        sample_weights = sample_weights.astype(float)
        Y = np.array(Y, dtype=float)
        predictions = predictions.astype(float)

        # calculate weighted error and alpha value for the stump
        weighted_error = np.sum(sample_weights * (predictions != Y)) / np.sum(sample_weights)
        stump_alpha = 0.5 * np.log((1 - weighted_error) / max(weighted_error, 1e-10))

        # update sample weights for the next iteration
        sample_weights = sample_weights * np.exp(-stump_alpha * Y * predictions)
        sample_weights /= np.sum(sample_weights)  # Normalize the weights

        # append the stump and alpha values to their respective arrays
        f.append(stump)
        alpha.append(stump_alpha)

    # return array of stumps and alpha values
    return f, alpha



def adaboost_test(X, Y, f, alpha):
    n_samples = len(X)
    n_stumps = len(f)
    predictions = np.zeros(n_samples)

    for i in range(n_stumps):
        stump = f[i]
        stump_prediction = stump.predict(X)
        # make sure stump_prediction and alpha are of same data type
        stump_prediction = stump_prediction.astype(alpha[i].dtype)
        predictions += alpha[i] * stump_prediction

    # calculate final AdaBoost predictions by taking sign of the weighted sum
    final_predictions = np.sign(predictions)

    # calculate accuracy
    accuracy = np.mean(final_predictions == Y)

    return accuracy*100 