import torch

mis_data = torch.load('/Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/data/mis/processed/train.pt', weights_only=False)
dim_data = torch.load('/Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/data/dim/processed/train.pt', weights_only=False)

print("=== MIS Dataset ===")
print(f"Num samples: {len(mis_data)}")
print(f"First sample keys: {mis_data[0].keys if hasattr(mis_data[0], 'keys') else dir(mis_data[0])[:5]}")
if hasattr(mis_data[0], 'x'):
    print(f"X shape: {mis_data[0].x.shape}")
    print(f"X dtype: {mis_data[0].x.dtype}")
if hasattr(mis_data[0], 'edge_index'):
    print(f"Edge index shape: {mis_data[0].edge_index.shape}")

print("\n=== DIMACS Dataset ===")
print(f"Num samples: {len(dim_data)}")
if hasattr(dim_data[0], 'x'):
    print(f"X shape: {dim_data[0].x.shape}")
    print(f"X dtype: {dim_data[0].x.dtype}")
if hasattr(dim_data[0], 'edge_index'):
    print(f"Edge index shape: {dim_data[0].edge_index.shape}")

print("\n=== Node Count Comparison ===")
mis_nodes = [d.x.shape[0] for d in mis_data]
dim_nodes = [d.x.shape[0] for d in dim_data]

print(f"MIS - Min: {min(mis_nodes)}, Max: {max(mis_nodes)}, Avg: {sum(mis_nodes)/len(mis_nodes):.1f}")
print(f"DIMACS - Min: {min(dim_nodes)}, Max: {max(dim_nodes)}, Avg: {sum(dim_nodes)/len(dim_nodes):.1f}")
