import torch
import numpy as np
import networkx as nx
from torchmetrics import Metric


class TSPValidity(Metric):
    """Comprueba si los grafos generados son válidos para TSP (completos y conectados)"""
    def __init__(self):
        super().__init__()
        self.add_state("valid_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, adj_matrices):
        """
        Args:
            adj_matrices: tensor (batch_size, n_nodes, n_nodes)
        """
        batch_size = adj_matrices.shape[0]
        n_nodes = adj_matrices.shape[1]
        
        for i in range(batch_size):
            adj = adj_matrices[i].cpu().numpy()
            # Eliminar auto-loops
            np.fill_diagonal(adj, 0)
            
            # Verificar si es un grafo completo (todas las aristas existen)
            # En TSP, el grafo debe ser completo
            expected_edges = n_nodes * (n_nodes - 1)  # sin contar diagonal
            actual_edges = np.count_nonzero(adj)
            
            # También verificar conectividad
            G = nx.from_numpy_array(adj > 0)  # grafo binario
            is_connected = nx.is_connected(G) if len(G.nodes()) > 0 else False
            is_complete = (actual_edges == expected_edges)
            
            if is_connected and is_complete:
                self.valid_count += 1
        
        self.total_count += batch_size
    
    def compute(self):
        if self.total_count == 0:
            return torch.tensor(0.0)
        return self.valid_count.float() / self.total_count


class TSPTourLength(Metric):
    """Calcula la longitud promedio del tour más corto encontrado"""
    def __init__(self):
        super().__init__()
        self.add_state("total_length", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, adj_matrices):
        """
        Args:
            adj_matrices: tensor (batch_size, n_nodes, n_nodes) con pesos
        """
        batch_size = adj_matrices.shape[0]
        
        for i in range(batch_size):
            adj = adj_matrices[i].cpu().numpy()
            n_nodes = adj.shape[0]
            
            # Heurística greedy simple para encontrar un tour
            tour_length = self._greedy_tour(adj)
            
            if tour_length > 0 and tour_length < float('inf'):
                self.total_length += tour_length
                self.total_count += 1
    
    def _greedy_tour(self, adj):
        """Heurística greedy para construir un tour"""
        n = adj.shape[0]
        if n <= 1:
            return 0.0
        
        visited = [False] * n
        current = 0
        visited[current] = True
        tour_length = 0.0
        
        for _ in range(n - 1):
            # Encontrar el nodo más cercano no visitado
            min_dist = float('inf')
            next_node = -1
            
            for j in range(n):
                if not visited[j] and adj[current, j] > 0:
                    if adj[current, j] < min_dist:
                        min_dist = adj[current, j]
                        next_node = j
            
            if next_node == -1:
                return float('inf')  # Tour no válido
            
            tour_length += min_dist
            visited[next_node] = True
            current = next_node
        
        # Regresar al inicio
        if adj[current, 0] > 0:
            tour_length += adj[current, 0]
        else:
            return float('inf')
        
        return tour_length
    
    def compute(self):
        if self.total_count == 0:
            return torch.tensor(float('inf'))
        return self.total_length / self.total_count


class TSPGraphDensity(Metric):
    """Mide la densidad de aristas (debería ser ~1.0 para grafos completos)"""
    def __init__(self):
        super().__init__()
        self.add_state("total_density", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, adj_matrices):
        """
        Args:
            adj_matrices: tensor (batch_size, n_nodes, n_nodes)
        """
        batch_size = adj_matrices.shape[0]
        
        for i in range(batch_size):
            adj = adj_matrices[i].cpu().numpy()
            n = adj.shape[0]
            
            if n > 1:
                # Densidad = aristas_presentes / aristas_posibles
                np.fill_diagonal(adj, 0)
                num_edges = np.count_nonzero(adj)
                max_edges = n * (n - 1)
                density = num_edges / max_edges if max_edges > 0 else 0.0
                
                self.total_density += density
                self.total_count += 1
    
    def compute(self):
        if self.total_count == 0:
            return torch.tensor(0.0)
        return self.total_density / self.total_count
