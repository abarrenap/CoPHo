import torch
import pickle
import sys

# Remapea 'datasets' a 'src.datasets' (o al módulo actual donde esté la clase)
import src.datasets

class DummyModule:
    pass

sys.modules['datasets'] = src.datasets

from torch_geometric.data import Data

path = '/Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/data/planar/processed/train.pt'
data_tuple = torch.load(path)

g0 = data_tuple[0]
print("\n🧩 First graph (g0):")
print(g0)

print("\n📊 Node features x (first 10 rows):")
print(g0.x[:10])

print("\n🔗 Edge index (first 10 edges):")
print(g0.edge_index[:, :10])

print("\n🏷 Edge attributes (first 10 edges):")
print(g0.edge_attr[:10])

print("\n🎯 Label y:")
print(g0.y)
