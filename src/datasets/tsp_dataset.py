
import os
import torch
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, Data
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
import json

class TSPDataset(InMemoryDataset):
    """
    Dataset loader for TSP adjacency matrices stored in a JSON file
    """
    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # Only one file: tsp_dataset.json
        return ['tsp_dataset.json']

    @property
    def processed_file_names(self):
        return [f'{self.split}.pt']

    def download(self):
        pass  # No download needed

    def process(self):
        # Load all graphs from JSON
        json_path = os.path.join(self.root, 'tsp_dataset.json')
        with open(json_path, 'r') as f:
            adj_matrices = json.load(f)
        data_list = []
        for adj in adj_matrices:
            adj = torch.tensor(adj, dtype=torch.float32)
            num_nodes = adj.shape[0]
            # Convert adjacency matrix to edge_index
            edge_index = (adj > 0).nonzero(as_tuple=False).t().contiguous()
            
            # Para TSP: edge_attr solo contiene el peso (1 dimensión)
            # El peso 0 indica que no hay arista
            edge_weights = adj[adj > 0].unsqueeze(1)  # Peso de la arista (bs, 1)
            edge_attr = edge_weights  # Solo el peso, sin encoding adicional
            
            x = torch.ones((num_nodes, 1), dtype=torch.float32)  # Nodo dummy
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
            # y debe existir pero vacío
            data.y = torch.zeros((1, 0), dtype=torch.float32)  # Empty label
            data_list.append(data)
        # Split
        num_graphs = len(data_list)
        test_len = max(1, int(round(num_graphs * 0.2)))
        train_len = max(1, int(round((num_graphs - test_len) * 0.8)))
        val_len = num_graphs - train_len - test_len
        torch.manual_seed(42)
        indices = torch.randperm(num_graphs)
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len + val_len]
        test_indices = indices[train_len + val_len:]
        if self.split == 'train':
            split_indices = train_indices
        elif self.split == 'val':
            split_indices = val_indices
        else:
            split_indices = test_indices
        split_data = [data_list[i] for i in split_indices]
        if self.pre_transform is not None:
            split_data = [self.pre_transform(data) for data in split_data]
        data, slices = self.collate(split_data)
        torch.save((data, slices), self.processed_paths[0])

class TSPDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datasets = {
            'train': TSPDataset(root=cfg.dataset.datadir, split='train'),
            'val': TSPDataset(root=cfg.dataset.datadir, split='val'),
            'test': TSPDataset(root=cfg.dataset.datadir, split='test'),
        }
        super().__init__(cfg, self.datasets)
        self.infos = TSPDatasetInfos(self.datasets)

    def node_types(self):
        return torch.tensor([1.0])

class TSPDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datasets):
        super().__init__()
        self.name = 'tsp'
        self.n_nodes = self._compute_node_distribution(datasets)
        self.node_types = torch.tensor([1.0])
        self.edge_types = torch.tensor([1.0])
        super().complete_infos(self.n_nodes, self.node_types)
    def _compute_node_distribution(self, datasets):
        max_n = 0
        node_counts_list = []
        for split in ['train', 'val', 'test']:
            if split in datasets:
                for data in datasets[split]:
                    max_n = max(max_n, data.num_nodes)
                    node_counts_list.append(data.num_nodes)
        n_nodes = torch.zeros(max_n + 1)
        for count in node_counts_list:
            n_nodes[count] += 1
        n_nodes = n_nodes / n_nodes.sum()
        return n_nodes
    def compute_input_output_dims(self, datamodule, extra_features=None, domain_features=None, graph_generation_model=None, newflag=True):
        if extra_features is None or domain_features is None:
            raise ValueError("extra_features and domain_features must be provided")
        super().compute_input_output_dims(datamodule, extra_features, domain_features)
