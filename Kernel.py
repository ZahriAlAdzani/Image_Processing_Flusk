import numpy as np


def generate_low_pass(size):
    # Generate random numbers in the matrix
    matrix = np.random.rand(size, size)

    # Normalize the matrix so that the sum of all elements is 1
    normalized_matrix = matrix / np.sum(matrix)

    return normalized_matrix


def generate_high_pass(size):
    # Generate random numbers in the matrix
    matrix = np.random.rand(size, size)

    # Subtract the mean of all elements to make the sum equal to 0
    matrix -= np.mean(matrix)

    return matrix


def generate_band_pass(size):
    while True:
        matrix = np.random.rand(size)
        # Check if the sum of values is not 0 or 1
        matrix_sum = matrix.sum()
        if matrix_sum != 0 and matrix_sum != 1:
            break
    return matrix


