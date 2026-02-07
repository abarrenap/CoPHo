import os
import torch
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_undirected
import networkx as nx
from pathlib import Path
from torch.utils.data import random_split
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


class DIMACSDataset(InMemoryDataset):
    """
    Dataset loader for DIMACS graph coloring format (.col files)
    """
    
    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """Get all .col files from the root directory"""
        if not os.path.exists(self.root):
            return []
        col_files = [f for f in os.listdir(self.root) if f.endswith('.col')]
        return sorted(col_files)

    @property
    def processed_file_names(self):
        return [f'{self.split}.pt']

    def download(self):
        """No download needed - data is already provided"""
        pass

    def _parse_dimacs_file(self, filepath):
        """
        Parse a DIMACS .col file and convert to torch_geometric.Data
        Returns a Data object
        """
        edges = []
        num_nodes = 0
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('c'):
                    continue
                if line.startswith('p'):
                    # p edge n m
                    parts = line.split()
                    num_nodes = int(parts[2])
                    continue
                if line.startswith('e'):
                    # e u v (1-indexed)
                    parts = line.split()
                    u, v = int(parts[1]) - 1, int(parts[2]) - 1  # Convert to 0-indexed
                    edges.append([u, v])
                    edges.append([v, u])  # Make undirected
        
        if len(edges) == 0:
            # Graph with no edges
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 2), dtype=torch.float)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.zeros((edge_index.size(1), 2), dtype=torch.float)
            edge_attr[:, 1] = 1.0
        
        # Create node features (one-hot with 2 channels, like other graph datasets)
        x = torch.zeros((num_nodes, 2), dtype=torch.float)
        x[:, 1] = 1.0
        
        # Create Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)

        # Add graph-level label (keep 2D for batching)
        data.y = torch.zeros((1, 1), dtype=torch.long)
        
        return data

    def process(self):
        """Process all .col files and split into train/val/test"""
        raw_files = self.raw_file_names
        data_list = []
        
        # Parse all files
        for filename in raw_files:
            filepath = os.path.join(self.root, filename)
            try:
                data = self._parse_dimacs_file(filepath)
                if data.num_nodes == 0:
                    continue
                data_list.append(data)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        print(f"Loaded {len(data_list)} graphs from DIMACS dataset")
        
        if len(data_list) == 0:
            raise ValueError(f"No .col files found in {self.root}")
        
        # Split into train/val/test
        num_graphs = len(data_list)
        test_len = max(1, int(round(num_graphs * 0.2)))
        train_len = max(1, int(round((num_graphs - test_len) * 0.8)))
        val_len = num_graphs - train_len - test_len
        
        # Create deterministic split
        torch.manual_seed(42)
        indices = torch.randperm(num_graphs)
        
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len + val_len]
        test_indices = indices[train_len + val_len:]
        
        if self.split == 'train':
            split_indices = train_indices
        elif self.split == 'val':
            split_indices = val_indices
        else:  # test
            split_indices = test_indices
        
        split_data = [data_list[i] for i in split_indices]
        
        if self.pre_transform is not None:
            split_data = [self.pre_transform(data) for data in split_data]
        
        data, slices = self.collate(split_data)
        torch.save((data, slices), self.processed_paths[0])


class DIMACSDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datasets = {
            'train': DIMACSDataset(root=cfg.dataset.datadir, split='train'),
            'val': DIMACSDataset(root=cfg.dataset.datadir, split='val'),
            'test': DIMACSDataset(root=cfg.dataset.datadir, split='test'),
        }
        super().__init__(cfg, self.datasets)
        self.infos = DIMACSDatasetInfos(self.datasets)

    def node_types(self):
        """DIMACS graphs have no node types - all nodes are the same"""
        return torch.tensor([1.0])


class DIMACSDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datasets):
        super().__init__()
        self.name = 'dimacs'
        
        # Compute node distribution
        self.n_nodes = self._compute_node_distribution(datasets)
        self.node_types = torch.tensor([1.0])  # Single node type
        self.edge_types = torch.tensor([1.0, 0.0])  # No edge, edge
        
        # Call complete_infos to set up nodes_dist and other attributes
        super().complete_infos(self.n_nodes, self.node_types)
        
    def _compute_node_distribution(self, datasets):
        """Compute distribution of node counts in the dataset"""
        # Find max nodes
        max_n = 0
        node_counts_list = []
        for split in ['train', 'val', 'test']:
            if split in datasets:
                for data in datasets[split]:
                    max_n = max(max_n, data.num_nodes)
                    node_counts_list.append(data.num_nodes)
        
        # Create distribution tensor
        n_nodes = torch.zeros(max_n + 1)
        for count in node_counts_list:
            n_nodes[count] += 1
        
        # Normalize
        n_nodes = n_nodes / n_nodes.sum()
        return n_nodes
        
    def compute_input_output_dims(self, datamodule, extra_features=None, domain_features=None, graph_generation_model=None, newflag=True):
        """Compute input/output dimensions for DIMACS based on actual batch tensors."""
        if extra_features is None or domain_features is None:
            raise ValueError("extra_features and domain_features must be provided")
        super().compute_input_output_dims(datamodule, extra_features, domain_features)
