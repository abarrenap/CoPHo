import numpy as np
import random
import json

def generate_tsp_adjacency_matrix_int(num_nodes, min_weight=1, max_weight=100):
    """
    Generate a complete graph adjacency matrix for TSP with integer edge weights.
    :param num_nodes: number of nodes in the graph
    :param min_weight: minimum edge weight
    :param max_weight: maximum edge weight
    :return: adjacency matrix as a numpy array
    """
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = random.randint(min_weight, max_weight)
            adj_matrix[i][j] = adj_matrix[j][i] = weight
    return adj_matrix

def generate_tsp_dataset_int(n_graphs, max_size, min_weight=1, max_weight=100):
    """
    Generate a dataset of TSP graphs as adjacency matrices with integer weights.
    :param n_graphs: total number of graphs
    :param max_size: maximum number of nodes per graph
    :param min_weight: minimum edge weight
    :param max_weight: maximum edge weight
    :return: list of adjacency matrices
    """
    dataset = []
    for _ in range(n_graphs):
        size = random.randint(5, max_size)  # at least 5 nodes
        adj_matrix = generate_tsp_adjacency_matrix_int(size, min_weight, max_weight)
        dataset.append(adj_matrix.tolist())  # convert to list for JSON compatibility
    return dataset

def save_dataset(dataset, filename="tsp_dataset.json"):
    """
    Save dataset of adjacency matrices to a JSON file.
    """
    with open(filename, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Dataset saved to {filename}")

# Example usage
if __name__ == "__main__":
    n_graphs = 100  # number of graphs
    max_size = 20   # max nodes per graph
    dataset = generate_tsp_dataset_int(n_graphs, max_size)
    save_dataset(dataset)