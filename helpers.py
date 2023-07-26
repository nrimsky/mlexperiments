import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt

def orthogonal_complement(eigenvector_matrix):
    """
    eigenvector_matrix: a matrix of eigenvectors (n_eigenvectors x n_weights)

    returns: a matrix of orthogonal vectors (n_weights x n_weights)
    """
    return np.eye(eigenvector_matrix.shape[-1]) - np.matmul(np.transpose(eigenvector_matrix), eigenvector_matrix)
    
def plot_pertubation_results(results):
    t_values = [result[0] for result in results]
    op_05_accuracy_values = [result[1] for result in results]
    pure_num_acc_values = [result[2] for result in results]
    pure_pattern_acc_values = [result[3] for result in results]
    plt.figure(figsize=(10, 6))

    plt.plot(t_values, op_05_accuracy_values, label='Opacity 0.5', color='r')
    plt.plot(t_values, pure_num_acc_values, label='Pure numbers', color='g')
    plt.plot(t_values, pure_pattern_acc_values, label='Pure patterns', color='b')

    plt.xlabel('Perturbation amount t')
    plt.ylabel('Accuracy (%)')
    plt.title('Model accuracy for different t values')
    plt.legend()
    plt.savefig('model_accuracy_graph.png', format='png')
