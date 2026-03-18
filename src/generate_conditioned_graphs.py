import os
import inspect
import shutil
import tempfile
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from omegaconf import open_dict
from pytorch_lightning import Trainer

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

    # Safety guard: avoid runtime crashes when the config enables conditioning
    # but no guidance model has been attached (e.g., unconditional generation).
    if bool(getattr(model, "enable_condition", False)) and getattr(model, "guidance_model", None) is None:
        print("[WARNING] Conditioning is enabled but no guidance model is loaded. Disabling conditioning for this run.")
        model.enable_condition = False

    return model


def compute_gin_embeddings(graphs_path: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    graphs = load_graphs_from_txt(graphs_path)
    print(f"Loaded {len(graphs)} graphs from {graphs_path}")
    graph_sizes = torch.tensor([int(g["n"]) for g in graphs], dtype=torch.long)
    g, h = graphs_to_dgl(graphs, device=torch.device("cpu"))
    # Keep conditioning equivalent to MISDataset/main test path.
    encoder = load_feature_extractor(device=torch.device("cpu"), output_dim=70)
    with torch.no_grad():
        embeddings = encoder(g, h).detach()
    print(f"Generated {embeddings.shape[0]} embeddings")
    return embeddings.to(device), graph_sizes.to(device)


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


def summarize_condition_stats(cond: torch.Tensor, label: str) -> None:
    cond = cond.float()
    if cond.numel() == 0:
        print(f"{label}: empty")
        return
    norms = cond.norm(dim=-1)
    print(
        f"{label}: shape={tuple(cond.shape)} "
        f"mean={cond.mean().item():.4f} std={cond.std().item():.4f} "
        f"min={cond.min().item():.4f} max={cond.max().item():.4f} "
        f"norm_mean={norms.mean().item():.4f} norm_std={norms.std().item():.4f}"
    )


def count_graphs_in_txt(graphs_path: str) -> int:
    count = 0
    with open(graphs_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("N="):
                count += 1
    return count


def generate_conditioned_graphs_via_trainer_test(
    checkpoint_path: str,
    graphs_path: str,
    output_path: str,
    guidance_path: Optional[str] = None,
    device: str = "cpu",
    num_samples: Optional[int] = None,
) -> None:
    cfg, _ = load_cfg_from_checkpoint(checkpoint_path)
    if cfg.dataset.name != "mis":
        raise NotImplementedError("trainer.test conditioning path is currently implemented for dataset=mis only.")

    if num_samples is None:
        num_samples = count_graphs_in_txt(graphs_path)
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")

    temp_root = tempfile.mkdtemp(prefix="mis_conditioned_")
    previous_full_test_flag = os.environ.get("MIS_USE_FULL_AS_TEST")
    try:
        os.environ["MIS_USE_FULL_AS_TEST"] = "1"
        raw_dir = os.path.join(temp_root, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        temp_mis_path = os.path.join(raw_dir, "mis.txt")
        shutil.copyfile(graphs_path, temp_mis_path)

        with open_dict(cfg):
            cfg.dataset.datadir = temp_root
            cfg.general.final_model_samples_to_generate = int(num_samples)
            cfg.general.final_model_samples_to_save = int(num_samples)
            cfg.general.wandb = "disabled"
            if int(getattr(cfg.general, "number_chain_steps", 0)) <= 0:
                cfg.general.final_model_chains_to_save = 0

        datamodule, model_kwargs = build_datamodule_and_infos(cfg)
        model = build_model(cfg, model_kwargs, checkpoint_path, guidance_path=guidance_path)

        device_t = torch.device(device)
        accelerator = "gpu" if device_t.type == "cuda" else "cpu"
        trainer = Trainer(
            gradient_clip_val=cfg.train.clip_grad,
            strategy="auto",
            accelerator=accelerator,
            devices=1,
            max_epochs=cfg.train.n_epochs,
            check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
            fast_dev_run=False,
            enable_progress_bar=False,
            callbacks=[],
            log_every_n_steps=50,
            logger=[],
        )

        cwd = os.getcwd()
        before_files = set(glob(os.path.join(cwd, "generated_samples*.txt")))
        trainer.test(model, datamodule=datamodule)
        after_files = set(glob(os.path.join(cwd, "generated_samples*.txt")))

        new_files = sorted(after_files - before_files, key=os.path.getmtime)
        if not new_files:
            all_files = sorted(after_files, key=os.path.getmtime)
            if not all_files:
                raise RuntimeError("trainer.test finished but no generated_samples*.txt file was produced.")
            new_files = [all_files[-1]]

        source_path = new_files[-1]
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, output)
        print(f"Copied trainer.test output from {source_path} to {output_path}")
    finally:
        if previous_full_test_flag is None:
            os.environ.pop("MIS_USE_FULL_AS_TEST", None)
        else:
            os.environ["MIS_USE_FULL_AS_TEST"] = previous_full_test_flag
        shutil.rmtree(temp_root, ignore_errors=True)


def generate_conditioned_graphs(checkpoint_path: str,
                                 graphs_path: str,
                                 output_path: str,
                                 guidance_path: Optional[str] = None,
                                 device: str = "cpu",
                                 num_samples: Optional[int] = None,
                                 per_embedding: int = 1,
                                 use_trainer_test: bool = False):
    if use_trainer_test:
        if per_embedding != 1:
            print("[WARNING] per_embedding is ignored when use_trainer_test=True")
        return generate_conditioned_graphs_via_trainer_test(
            checkpoint_path=checkpoint_path,
            graphs_path=graphs_path,
            output_path=output_path,
            guidance_path=guidance_path,
            device=device,
            num_samples=num_samples,
        )

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

    cond, graph_sizes = compute_gin_embeddings(graphs_path, device_t)
    if cond.dim() == 1:
        cond = cond.unsqueeze(0)
    elif cond.dim() > 2:
        cond = cond.view(cond.size(0), -1)

    summarize_condition_stats(cond, "Condition tensor")

    if per_embedding < 1:
        raise ValueError("per_embedding must be >= 1")

    if per_embedding > 1:
        cond = cond.repeat_interleave(per_embedding, dim=0)
        graph_sizes = graph_sizes.repeat_interleave(per_embedding, dim=0)
        print(f"After repeat_interleave with per_embedding={per_embedding}: {cond.shape}")

    if num_samples is None:
        # When conditioning on input graphs, generate samples for all conditions
        num_samples = int(cond.shape[0])
    
    print(f"Number of samples to generate: {num_samples}")

    samples = []
    samples_left_to_generate = num_samples
    samples_left_to_save = num_samples
    chains_left_to_save = int(
        getattr(cfg.general, "final_model_chains_to_save", getattr(cfg.general, "chains_to_save", 0))
    )
    number_chain_steps = int(cfg.general.number_chain_steps)
    batch_id = 0
    cond_idx = 0  # Track position in condition tensor

    with tqdm(total=num_samples, desc="Generating graphs", unit="graph") as pbar:
        while samples_left_to_generate > 0:
            bs = int(cfg.train.batch_size)
            remaining_cond = len(cond) - cond_idx
            if remaining_cond <= 0:
                break
            to_generate = min(samples_left_to_generate, bs, remaining_cond)
            to_save = min(samples_left_to_save, bs, to_generate)
            chains_save = min(chains_left_to_save, bs, to_generate)
            if number_chain_steps <= 0:
                chains_save = 0
            cond_batch = cond[cond_idx:cond_idx + to_generate]
            graph_sizes_batch = graph_sizes[cond_idx:cond_idx + to_generate]

            sample_kwargs = {
                "batch_id": batch_id,
                "batch_size": to_generate,
                "num_nodes": graph_sizes_batch,
                "save_final": to_save,
                "keep_chain": chains_save,
                "number_chain_steps": number_chain_steps,
                "y_cond": cond_batch,
            }
            # Some model variants use cond=... (persisthomo), others only y_cond.
            if "cond" in inspect.signature(model.sample_batch).parameters:
                sample_kwargs["cond"] = cond_batch

            sample_out = model.sample_batch(**sample_kwargs)
            if isinstance(sample_out, tuple):
                molecule_list = sample_out[0]
            else:
                molecule_list = sample_out
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
    checkpoint = "/Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/outputs/2026-03-16/21-01-09-mixed_exp1/checkpoints/mixed_exp1/epoch=69.ckpt"
    graphs = "/Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/data/mis/raw/test_32.txt"

    output = "../generated/32.txt"
    guidance = '/Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/src/weights/CLASSIFIER_struct_embedding_community.pth'
    device = "cpu"
    per_embedding = 1
    use_trainer_test = True

    generate_conditioned_graphs(
        checkpoint_path=checkpoint,
        graphs_path=graphs,
        output_path=output,
        guidance_path=guidance,
        device=device,
        num_samples=None,
        per_embedding=per_embedding,
        use_trainer_test=use_trainer_test,
    )

    os.system("rm .wandb_run_id")
    os.system("rm generated_adjs.*")
    os.system("rm generated_samples*.txt")
