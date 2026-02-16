from pathlib import Path

from datasets.mis_dataset import MISDataset
import torch


def format_instance(data):
	return {
		"num_nodes": data.num_nodes,
		"num_edges": data.edge_index.size(1),
		"x_shape": tuple(data.x.shape),
		"edge_attr_shape": tuple(data.edge_attr.shape),
		"y": data.y if hasattr(data, "y") else None,
		"y_shape": tuple(data.y.shape) if hasattr(data, "y") and data.y is not None else None,
	}


def main():
	repo_root = Path(__file__).resolve().parents[1]
	mis_root = repo_root / "data" / "mis"

	train_ds = MISDataset(root=str(mis_root), split="train")
	val_ds = MISDataset(root=str(mis_root), split="val")
	test_ds = MISDataset(root=str(mis_root), split="test")

	print("Train instance:", format_instance(train_ds[0]))
	print("Val instance:", format_instance(val_ds[0]))
	print("Test instance:", format_instance(test_ds[0]))


if __name__ == "__main__":
	main()
