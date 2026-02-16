from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch_geometric
from tqdm import tqdm

from src.analysis.spectre_utils import SBMSamplingMetrics
from src.analysis.visualization import NonMolecularVisualization
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.encoder.encoder import load_feature_extractor
from src.encoder.load_data import graphs_to_dgl
from src.metrics.abstract_metrics import TrainAbstractMetrics, TrainAbstractMetricsDiscrete

OUTPUT = "/Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/outputs/2026-02-16/16-18-47-mis_cond_1"
CKPT_PATH = OUTPUT + "/checkpoints/mis_cond_1/last-v1.ckpt"
USE_CONDITIONAL = True  # Set to True to use conditional model
N_PER_GRAPH = 5
MAX_TEST_GRAPHS = 20
HIST_PATH = OUTPUT + "/embedding_error_hist.png"


def load_cfg_from_checkpoint(ckpt_path):
	ckpt = torch.load(ckpt_path, map_location="cpu")
	if "hyper_parameters" in ckpt and "cfg" in ckpt["hyper_parameters"]:
		return ckpt["hyper_parameters"]["cfg"]
	if "cfg" in ckpt:
		return ckpt["cfg"]
	raise KeyError("Checkpoint does not contain cfg in hyper_parameters or top-level keys.")


def build_model_and_data(cfg, ckpt_path):
	dataset_name = getattr(cfg.dataset, "name", None) or cfg.dataset["name"]
	if dataset_name != "mis":
		raise ValueError(f"This script expects MIS, got: {dataset_name}")

	from src.datasets.mis_dataset import MISDataModule, MISDatasetInfos
	
	if USE_CONDITIONAL:
		from src.diffusion_model_discrete_condition import DiscreteDenoisingDiffusion
	else:
		from src.diffusion_model_discrete import DiscreteDenoisingDiffusion

	datamodule = MISDataModule(cfg)
	sampling_metrics = SBMSamplingMetrics(datamodule)
	dataset_infos = MISDatasetInfos(datamodule.datasets)
	train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == "discrete" else TrainAbstractMetrics()
	visualization_tools = NonMolecularVisualization()

	if cfg.model.type == "discrete" and cfg.model.extra_features is not None:
		extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
	else:
		extra_features = DummyExtraFeatures()
	domain_features = DummyExtraFeatures()

	dataset_infos.compute_input_output_dims(
		datamodule=datamodule, extra_features=extra_features, domain_features=domain_features
	)

	model_kwargs = {
		"dataset_infos": dataset_infos,
		"train_metrics": train_metrics,
		"sampling_metrics": sampling_metrics,
		"visualization_tools": visualization_tools,
		"extra_features": extra_features,
		"domain_features": domain_features,
	}

	model = DiscreteDenoisingDiffusion.load_from_checkpoint(ckpt_path, **model_kwargs)
	return model, datamodule


def build_graph_dicts(molecule_list):
	graphs = []
	for _, edge_types in molecule_list:
		edge_matrix = edge_types.float().clone()
		edge_matrix.fill_diagonal_(0)
		edge_matrix = edge_matrix.clamp(min=0)
		n = edge_matrix.shape[0]
		graphs.append({"n": n, "x": [1.0] * n, "e": edge_matrix.flatten().tolist()})
	return graphs


def edge_matrix_to_networkx(edge_matrix):
	"""Convert edge matrix to networkx graph."""
	g = nx.from_numpy_array(edge_matrix.numpy() if hasattr(edge_matrix, 'numpy') else edge_matrix)
	return g


def plot_original_and_generated(orig_data, gen_molecule_list, idx, save_dir=OUTPUT):
	"""Plot original graph and generated graphs side-by-side."""
	n_gen = len(gen_molecule_list)
	n_cols = n_gen + 1
	fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
	if n_cols == 1:
		axes = [axes]

	# Reconstruct original adjacency matrix from edge_index
	orig_adj = torch_geometric.utils.to_dense_adj(
		orig_data.edge_index, max_num_nodes=orig_data.num_nodes
	).squeeze(0)
	
	# Plot original
	orig_g = edge_matrix_to_networkx(orig_adj)
	pos = nx.spring_layout(orig_g, seed=42)
	nx.draw_networkx_nodes(orig_g, pos, ax=axes[0], node_color="lightblue", node_size=300)
	nx.draw_networkx_edges(orig_g, pos, ax=axes[0], width=1.5)
	axes[0].set_title(f"Original (graph {idx})", fontsize=12, fontweight="bold")
	axes[0].axis("off")

	# Plot generated
	for i, (_, edge_types) in enumerate(gen_molecule_list):
		gen_edge_matrix = edge_types.float().clone()
		gen_edge_matrix.fill_diagonal_(0)
		gen_edge_matrix = gen_edge_matrix.clamp(min=0)
		gen_g = edge_matrix_to_networkx(gen_edge_matrix)
		pos_gen = nx.spring_layout(gen_g, seed=42)
		nx.draw_networkx_nodes(gen_g, pos_gen, ax=axes[i + 1], node_color="lightcoral", node_size=300)
		nx.draw_networkx_edges(gen_g, pos_gen, ax=axes[i + 1], width=1.5)
		axes[i + 1].set_title(f"Generated {i+1}", fontsize=12)
		axes[i + 1].axis("off")

	save_path = Path(save_dir) / f"graph_{idx}.png"
	plt.tight_layout()
	plt.savefig(save_path, dpi=100, bbox_inches="tight")
	plt.close()



def main():
	cfg = load_cfg_from_checkpoint(CKPT_PATH)
	model, datamodule = build_model_and_data(cfg, CKPT_PATH)

	device = torch.device("cpu")
	model.to(device)
	model.eval()
	model.visualization_tools = None  # Disable visualization to run without Trainer

	enc = load_feature_extractor(device=device)
	test_ds = datamodule.datasets["test"]

	errors = []
	test_indices = range(len(test_ds))
	if MAX_TEST_GRAPHS is not None:
		test_indices = range(min(MAX_TEST_GRAPHS, len(test_ds)))
	
	for idx in tqdm(test_indices, desc="Processing test graphs"):
		data = test_ds[idx]

		y_orig = data.y
		if y_orig.dim() == 1:
			y_orig = y_orig.unsqueeze(0)

		y_cond = y_orig.repeat(N_PER_GRAPH, 1).to(device)
		molecule_list, _, _ = model.sample_batch(
			batch_id=0,
			batch_size=N_PER_GRAPH,
			keep_chain=0,
			number_chain_steps=model.number_chain_steps,
			save_final=0,
			num_nodes=data.num_nodes,
			y_cond=y_cond,
		)

		# Visualize first N graphs
		plot_original_and_generated(data, molecule_list, idx)

		graph_dicts = build_graph_dicts(molecule_list)
		g, h = graphs_to_dgl(graph_dicts, device=device)
		gen_emb = enc(g, h).cpu()
		y_repeat = y_orig.repeat(gen_emb.shape[0], 1)
		distances = torch.norm(gen_emb - y_repeat, dim=1)
		errors.extend(distances.tolist())

	if not errors:
		raise RuntimeError("No errors computed; check that the test dataset is non-empty.")

	print(
		"Errors: mean={:.6f}, std={:.6f}, min={:.6f}, max={:.6f}".format(
			float(torch.tensor(errors).mean()),
			float(torch.tensor(errors).std()),
			float(min(errors)),
			float(max(errors)),
		)
	)

	Path(HIST_PATH).parent.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(8, 5))
	plt.hist(errors, bins=30, alpha=0.8)
	plt.title("Embedding error distribution")
	plt.xlabel("L2 error")
	plt.ylabel("Count")
	plt.tight_layout()
	plt.savefig(HIST_PATH, dpi=150)
	print(f"Saved histogram to: {HIST_PATH}")


if __name__ == "__main__":
	main()
