import os
import torch
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_undirected
import networkx as nx
from pathlib import Path
from torch.utils.data import random_split
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from models.gin import GIN
import torch
from encoder.encoder import load_feature_extractor
from encoder.load_data import graphs_to_dgl

EMBED_DIM = 70
enc = load_feature_extractor(device=torch.device('cpu'), output_dim=EMBED_DIM)

class MISDataset(InMemoryDataset):
    """
    Dataset loader for MIS (Maximum Independent Set) graph format (.txt file)
    """
    
    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """Get the mis.txt file"""
        return ['mis.txt']

    @property
    def processed_file_names(self):
        return [f'{self.split}.pt']

    def download(self):
        """No download needed - data is already provided"""
        pass

    def _parse_mis_file(self, filepath):
        """
        Parse a MIS .txt file and extract individual graphs
        Returns a list of Data objects
        """
        data_list = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line or not line.startswith('N='):
                i += 1
                continue
            
            # Parse number of nodes
            num_nodes = int(line.split('=')[1])
            i += 1
            
            # Parse X (node features)
            if i < len(lines) and lines[i].strip().startswith('X:'):
                i += 1
                if i < len(lines):
                    x_line = lines[i].strip()
                    x_values = list(map(int, x_line.split()))
                    # Node features (all 1s for MIS, create 2D feature)
                    x = torch.ones((num_nodes, 2), dtype=torch.float)
                    i += 1
            else:
                x = torch.ones((num_nodes, 2), dtype=torch.float)
            
            # Parse E (edge matrix)
            e_vals = [0.0] * (num_nodes * num_nodes)
            if i < len(lines) and lines[i].strip().startswith('E:'):
                i += 1
                
                # Read the adjacency matrix (num_nodes x num_nodes flattened)
                edge_matrix = []
                rows_needed = num_nodes
                rows_read = 0
                
                while rows_read < rows_needed and i < len(lines):
                    vals_line = lines[i].strip()
                    if vals_line and vals_line[0].isdigit() or vals_line[0] == '.':
                        # Parse this line of the edge matrix
                        vals = [float(v) for v in vals_line.split()]
                        edge_matrix.extend(vals)
                        rows_read += len(vals) / num_nodes  # Rough estimate of rows
                    i += 1
                
                # Create edge_index from edge_matrix
                edge_matrix = edge_matrix[:num_nodes * num_nodes]
                e_vals = edge_matrix
                adj_matrix = torch.tensor(edge_matrix, dtype=torch.float).reshape(num_nodes, num_nodes)
                
                # Extract edges from adjacency matrix
                edges = []
                for n1 in range(num_nodes):
                    for n2 in range(n1 + 1, num_nodes):  # Upper triangle only
                        if adj_matrix[n1, n2] > 0.5:
                            edges.append([n1, n2])
                            edges.append([n2, n1])  # Undirected
                
                if len(edges) == 0:
                    edge_index = torch.zeros((2, 0), dtype=torch.long)
                    edge_attr = torch.zeros((0, 2), dtype=torch.float)
                else:
                    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                    edge_attr = torch.zeros((edge_index.size(1), 2), dtype=torch.float)
                    edge_attr[:, 1] = 1.0
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 2), dtype=torch.float)
            
            # Create Data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
            graph_data = {"n": num_nodes, "x": [1.0] * num_nodes, "e": e_vals}
            g, h = graphs_to_dgl([graph_data], device=enc.device)
            emb = enc(g, h).cpu()  # Get graph embedding as label
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            elif emb.dim() > 2:
                emb = emb.view(emb.size(0), -1)
            data.y = emb
            data.cond = emb
            data_list.append(data)
        
        return data_list

    def process(self):
        """Process the MIS .txt file and split into train/val/test"""
        filepath = os.path.join(self.root, 'raw/mis.txt')
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"MIS file not found at {filepath}")
        
        data_list = self._parse_mis_file(filepath)
        
        print(f"Loaded {len(data_list)} graphs from MIS dataset")
        
        if len(data_list) == 0:
            raise ValueError(f"No graphs found in {filepath}")
        
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


class MISDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datasets = {
            'train': MISDataset(root=cfg.dataset.datadir, split='train'),
            'val': MISDataset(root=cfg.dataset.datadir, split='val'),
            'test': MISDataset(root=cfg.dataset.datadir, split='test'),
        }
        super().__init__(cfg, self.datasets)
        self.infos = MISDatasetInfos(self.datasets)

    def node_types(self):
        """MIS graphs have no node types - all nodes are the same"""
        return torch.tensor([1.0])


class MISDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datasets):
        super().__init__()
        self.name = 'mis'
        
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
        """Compute input/output dimensions for MIS based on actual batch tensors."""
        if extra_features is None or domain_features is None:
            raise ValueError("extra_features and domain_features must be provided")
        super().compute_input_output_dims(datamodule, extra_features, domain_features)
