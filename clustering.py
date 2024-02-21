import numpy as np

# 2) Clustering 
# (a) K_Means function
def K_Means(X, K, mu):
    # Check if mu is empty, and initialize cluster centers randomly if so     
    if len(mu) == 0:
        # Randomly initialize cluster centers by selecting K data points
        mu = X[np.random.choice(X.shape[0], K, replace=False)]

    while True:
        # Assign each data point to the nearest cluster
        distances = np.linalg.norm(X[:, np.newaxis] - mu, axis=2)
        labels = np.argmin(distances, axis=1)

        # Update cluster centers
        new_mu = np.array([X[labels == k].mean(axis=0) for k in range(K)])

        # Check for convergence
        if np.all(mu == new_mu):
            return new_mu

        mu = new_mu
        
# (b) K_Means_better
def K_Means_better(X, K, num_iterations=10):
    best_centers = None
    best_count = 0

    for _ in range(num_iterations):
        # Run K-Means with random or provided initial centers
        mu_initial = np.array([])  
        centers = K_Means(X, K, mu_initial)

        # function to Check if these centers are the same as the best found so far
        count = np.sum(np.all(np.isin(centers, best_centers, invert=True), axis=1))

        if count > best_count:
            best_centers = centers
            best_count = count

    return best_centers

# Test the function with K=2 and K=3
K = 2
mu_initial = np.array([])  

K = 3
mu_initial = np.array([]) 

# Example to check the working of function
if __name__ == "__main__":
    # Generate random data for testing
    np.random.seed(0)
    X = np.random.rand(100, 2)

    # Define the number of clusters and initial cluster centers (mu)
    K = 3
    initial_mu = np.array([[0.2, 0.2], [0.6, 0.6], [0.9, 0.1]])

    cluster_centers = K_Means(X, K, initial_mu)

    print("Final cluster centers:")
    print(cluster_centers)

