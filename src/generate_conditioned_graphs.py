import os
from pathlib import Path
from typing import List, Optional

import torch

from encoder.encoder import load_feature_extractor
from encoder.load_data import load_graphs_from_txt, graphs_to_dgl
from models import condi_config
from tqdm import tqdm


def load_cfg_from_checkpoint(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "hyper_parameters" in ckpt and "cfg" in ckpt["hyper_parameters"]:
        return ckpt["hyper_parameters"]["cfg"], ckpt
    if "cfg" in ckpt:
        return ckpt["cfg"], ckpt
    raise ValueError("Checkpoint does not contain cfg in hyper_parameters or root")


def build_datamodule_and_infos(cfg):
    dataset_name = cfg.dataset.name

    if dataset_name in ["sbm", "comm20", "planar", "enzymes"]:
        from datasets.spectre_dataset_multi import SpectreGraphDataModule, SpectreDatasetInfos
        from analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
        from analysis.visualization import NonMolecularVisualization
        from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
        from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

        datamodule = SpectreGraphDataModule(cfg, cond_type=condi_config.condition_target)
        if dataset_name == "sbm":
            sampling_metrics = SBMSamplingMetrics(datamodule)
        elif dataset_name == "comm20":
            sampling_metrics = Comm20SamplingMetrics(datamodule)
        else:
            sampling_metrics = PlanarSamplingMetrics(datamodule)

        dataset_infos = SpectreDatasetInfos(datamodule, cfg.dataset)
        train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == "discrete" else TrainAbstractMetrics()
        visualization_tools = NonMolecularVisualization()

        if cfg.model.type == "discrete" and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule,
                                                extra_features=extra_features,
                                                domain_features=domain_features)

        model_kwargs = {
            "dataset_infos": dataset_infos,
            "train_metrics": train_metrics,
            "sampling_metrics": sampling_metrics,
            "visualization_tools": visualization_tools,
            "extra_features": extra_features,
            "domain_features": domain_features,
        }
        return datamodule, model_kwargs

    if dataset_name == "dimacs":
        from datasets.dimacs_dataset import DIMACSDataModule, DIMACSDatasetInfos
        from analysis.spectre_utils import SBMSamplingMetrics
        from analysis.visualization import NonMolecularVisualization
        from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
        from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

        datamodule = DIMACSDataModule(cfg)
        sampling_metrics = SBMSamplingMetrics(datamodule)
        dataset_infos = DIMACSDatasetInfos(datamodule.datasets)
        train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == "discrete" else TrainAbstractMetrics()
        visualization_tools = NonMolecularVisualization()

        if cfg.model.type == "discrete" and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule,
                                                extra_features=extra_features,
                                                domain_features=domain_features)

        model_kwargs = {
            "dataset_infos": dataset_infos,
            "train_metrics": train_metrics,
            "sampling_metrics": sampling_metrics,
            "visualization_tools": visualization_tools,
            "extra_features": extra_features,
            "domain_features": domain_features,
        }
        return datamodule, model_kwargs

    if dataset_name == "mis":
        from datasets.mis_dataset import MISDataModule, MISDatasetInfos
        from analysis.spectre_utils import SBMSamplingMetrics
        from analysis.visualization import NonMolecularVisualization
        from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
        from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

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

        dataset_infos.compute_input_output_dims(datamodule=datamodule,
                                                extra_features=extra_features,
                                                domain_features=domain_features)

        model_kwargs = {
            "dataset_infos": dataset_infos,
            "train_metrics": train_metrics,
            "sampling_metrics": sampling_metrics,
            "visualization_tools": visualization_tools,
            "extra_features": extra_features,
            "domain_features": domain_features,
        }
        return datamodule, model_kwargs

    if dataset_name == "tsp":
        from datasets.tsp_dataset import TSPDataModule, TSPDatasetInfos
        from analysis.spectre_utils import TSPSamplingMetrics
        from analysis.visualization import WeightedVisualization
        from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
        from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

        datamodule = TSPDataModule(cfg)
        sampling_metrics = TSPSamplingMetrics(datamodule)
        dataset_infos = TSPDatasetInfos(datamodule.datasets)
        train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == "discrete" else TrainAbstractMetrics()
        visualization_tools = WeightedVisualization()

        if cfg.model.type == "discrete" and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule,
                                                extra_features=extra_features,
                                                domain_features=domain_features)

        model_kwargs = {
            "dataset_infos": dataset_infos,
            "train_metrics": train_metrics,
            "sampling_metrics": sampling_metrics,
            "visualization_tools": visualization_tools,
            "extra_features": extra_features,
            "domain_features": domain_features,
        }
        return datamodule, model_kwargs

    raise NotImplementedError(f"Unsupported dataset: {dataset_name}")


def build_model(cfg, model_kwargs, checkpoint_path: str, guidance_path: Optional[str] = None):
    if cfg.model.type == "discrete":
        use_persist_homo = cfg.model.get("use_persist_homo", False)
        if use_persist_homo:
            from diffusion_model_discrete_persisthomo import DiscreteDenoisingDiffusion
        else:
            from diffusion_model_discrete import DiscreteDenoisingDiffusion
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    else:
        from diffusion_model import LiftedDenoisingDiffusion
        model = LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("guidance_model.")}
    model.load_state_dict(filtered_state_dict, strict=False)

    if guidance_path is not None:
        from models.GNN_model import GraphStructModel, GraphDistanceModel
        from hydra.utils import to_absolute_path

        if "path" in condi_config.condition_target:
            guidance_model = GraphDistanceModel(hidden_dim=condi_config.HIDDEN_DIM,
                                                num_layers=condi_config.NUM_LAYERS,
                                                dropout=condi_config.DROPOUT)
        else:
            guidance_out_dim = model_kwargs["dataset_infos"].output_dims["y"]
            guidance_model = GraphStructModel(in_dim=1,
                                              hidden_dim=condi_config.HIDDEN_DIM,
                                              num_layers=condi_config.NUM_LAYERS,
                                              dropout=condi_config.DROPOUT,
                                              out_dim=guidance_out_dim)

        guidance_model_path = to_absolute_path(guidance_path)
        if os.path.isdir(guidance_model_path):
            guidance_model_path = os.path.join(
                guidance_model_path,
                f"CLASSIFIER_struct_{condi_config.condition_target[0]}_community.pth",
            )

        if os.path.exists(guidance_model_path):
            guidance_model.load_state_dict(torch.load(guidance_model_path, map_location="cpu"))
            model.assign_guidance_model(guidance_model)
        else:
            print(f"[WARNING] Guidance model not found: {guidance_model_path}")

    return model


def compute_gin_embeddings(graphs_path: str, device: torch.device) -> torch.Tensor:
    graphs = load_graphs_from_txt(graphs_path)
    print(f"Loaded {len(graphs)} graphs from {graphs_path}")
    g, h = graphs_to_dgl(graphs, device=torch.device("cpu"))
    gin_use_pretrained = getattr(condi_config, "gin_use_pretrained", True)
    gin_model_path = getattr(condi_config, "gin_model_path", None)

    encoder_kwargs = {
        "device": torch.device("cpu"),
        "output_dim": 70,
        "use_pretrained": gin_use_pretrained,
    }
    if gin_model_path:
        encoder_kwargs["model_path"] = gin_model_path

    encoder = load_feature_extractor(**encoder_kwargs)
    with torch.no_grad():
        embeddings = encoder(g, h).detach()
    print(f"Generated {embeddings.shape[0]} embeddings")
    return embeddings.to(device)


def save_generated_graphs(samples: List, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Overwrite on each run so evaluation does not mix multiple generations.
    with output.open("w", encoding="utf-8") as handle:
        for item in samples:
            n_nodes = int(item[0].shape[0])
            handle.write(f"N={n_nodes}\n")
            handle.write("X:\n")
            for _ in range(n_nodes):
                handle.write("1 ")
            handle.write("\n")
            handle.write("E:\n")
            for bond_list in item[1]:
                for bond in bond_list:
                    handle.write(f"{int(round(float(bond)))} ")
                handle.write("\n")


def normalize_condition_embeddings(cond: torch.Tensor) -> torch.Tensor:
    """Match MISDataset target preprocessing: per-graph max-abs normalization."""
    if cond.numel() == 0:
        return cond
    max_abs = cond.abs().amax(dim=-1, keepdim=True)
    max_abs = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
    return cond / max_abs


def generate_conditioned_graphs(checkpoint_path: str,
                                 graphs_path: str,
                                 output_path: str,
                                 guidance_path: Optional[str] = None,
                                 device: str = "cpu",
                                 num_samples: Optional[int] = None,
                                 per_embedding: int = 1):
    device_t = torch.device(device)

    cfg, _ = load_cfg_from_checkpoint(checkpoint_path)
    datamodule, model_kwargs = build_datamodule_and_infos(cfg)
    model = build_model(cfg, model_kwargs, checkpoint_path, guidance_path=guidance_path)
    model.to(device_t)
    model.eval()

    guidance_loaded = getattr(model, "guidance_model", None) is not None
    conditioning_enabled = bool(getattr(model, "enable_condition", False))
    conditioning_active = conditioning_enabled and guidance_loaded
    print(
        "Conditioning summary: "
        f"enabled={conditioning_enabled}, "
        f"guidance_loaded={guidance_loaded}, "
        f"active={conditioning_active}"
    )

    cond = compute_gin_embeddings(graphs_path, device_t)
    if cond.dim() == 1:
        cond = cond.unsqueeze(0)
    cond = normalize_condition_embeddings(cond)

    print(f"Condition tensor shape: {cond.shape}")

    if per_embedding < 1:
        raise ValueError("per_embedding must be >= 1")

    if per_embedding > 1:
        cond = cond.repeat_interleave(per_embedding, dim=0)
        print(f"After repeat_interleave with per_embedding={per_embedding}: {cond.shape}")

    if num_samples is None:
        # When conditioning on input graphs, generate samples for all conditions
        num_samples = int(cond.shape[0])
    
    print(f"Number of samples to generate: {num_samples}")

    samples = []
    samples_left_to_generate = num_samples
    samples_left_to_save = num_samples
    chains_left_to_save = int(cfg.general.chains_to_save)
    batch_id = 0
    cond_idx = 0  # Track position in condition tensor

    with tqdm(total=num_samples, desc="Generating graphs", unit="graph") as pbar:
        while samples_left_to_generate > 0:
            bs = int(cfg.train.batch_size)
            to_generate = min(samples_left_to_generate, bs, len(cond) - cond_idx)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            cond_batch = cond[cond_idx:cond_idx + to_generate]

            molecule_list, _, _ = model.sample_batch(
                batch_id=batch_id,
                batch_size=to_generate,
                num_nodes=None,
                save_final=to_save,
                keep_chain=chains_save,
                number_chain_steps=int(cfg.general.number_chain_steps),
                y_cond=cond_batch,
                cond=cond_batch,
            )
            samples.extend(molecule_list)

            batch_id += to_generate
            cond_idx += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save
            
            pbar.update(to_generate)

    save_generated_graphs(samples, output_path)
    print(f"Saved {len(samples)} graphs to {output_path}")


if __name__ == "__main__":
    checkpoint = "/users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/outputs/2026-03-10/08-54-52-barabasi_exp1/checkpoints/barabasi_exp1/epoch=45.ckpt"
    graphs = "/Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/data/barabasi/raw/test_32.txt"

    output = "../generated/45.txt"
    guidance = '/Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/src/weights/CLASSIFIER_struct_embedding_community.pth'
    device = "cpu"
    per_embedding = 1

    generate_conditioned_graphs(
        checkpoint_path=checkpoint,
        graphs_path=graphs,
        output_path=output,
        guidance_path=guidance,
        device=device,
        num_samples=None,
        per_embedding=per_embedding,
    )
