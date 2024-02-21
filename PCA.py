import numpy as np
import matplotlib.pyplot as plt

def compute_covariance_matrix(Z):
    # transpose matrix Z
    ZT = np.transpose(Z)

    # compute covariance matrix
    covariance_matrix = np.dot(ZT, Z) / Z.shape[0]

    return covariance_matrix


def find_pcs(cov):
    
    # compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Sort eigenvalues and corresponding eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    pcs = eigenvectors[:, idx]
    L = eigenvalues[idx]

    return pcs, L


def project_data(Z, PCS, L):
    
    # take first principal component
    first_pc = PCS[:, 0]

    # project data onto first principal component
    Z_star = np.dot(Z, first_pc)

    return Z_star


def show_plot(Z, Z_star):
    
    # scatter plot for original data
    plt.scatter(Z[:, 0], Z[:, 1], label='Original Data', marker='o', color='blue')

    # line plot for projected data
    plt.plot(Z_star, label='Projected Data', linestyle='-', marker='o', color='red')

    # set labels and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Original Data and Projected Data')

    # legend
    plt.legend()

    # save plot as a .jpg file
    plt.savefig('plot.jpg')

    # display plot
    plt.show()
