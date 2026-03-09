import graph_tool as gt
import os
import pathlib
import warnings
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import numpy as np

torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

from diffusion_model import LiftedDenoisingDiffusion
# Import será dinámico para soportar tanto discrete como persist_homo
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from models.GNN_model import GraphDistanceModel, GraphStructModel
from models import condi_config

from pytorch_lightning import seed_everything
from pathlib import Path
seed_everything(42)
from hydra.utils import to_absolute_path, get_original_cwd

warnings.filterwarnings("ignore", category=PossibleUserWarning)


def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    resume = to_absolute_path(resume)
    
    if cfg.model.type == 'discrete':
        use_persist_homo = cfg.model.get('use_persist_homo', False)
        if use_persist_homo:
            from diffusion_model_discrete_persisthomo import DiscreteDenoisingDiffusion
        else:
            from diffusion_model_discrete import DiscreteDenoisingDiffusion
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
        # Cargar checkpoint manualmente para filtrar guidance_model
        ckpt = torch.load(resume)
        state_dict = ckpt["state_dict"]
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("guidance_model.")}
        model.load_state_dict(filtered_state_dict, strict=False)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)

    # Cargar checkpoint para determinar el tipo de modelo guardado
    ckpt = torch.load(resume_path)
    
    if cfg.model.type == 'discrete':
        use_persist_homo = cfg.model.get('use_persist_homo', False)
        if use_persist_homo:
            from diffusion_model_discrete_persisthomo import DiscreteDenoisingDiffusion
        else:
            from diffusion_model_discrete import DiscreteDenoisingDiffusion
        
        # Crear modelo y cargar estado filtrando guidance_model
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
        state_dict = ckpt["state_dict"]
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("guidance_model.")}
        model.load_state_dict(filtered_state_dict, strict=False)
    else:
        model = LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)
        state_dict = ckpt["state_dict"]
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("guidance_model.")}
        model.load_state_dict(filtered_state_dict, strict=False)
    
    new_cfg = ckpt.get("hyper_parameters", {}).get("cfg", cfg) if "hyper_parameters" in ckpt else cfg

    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '_resume'

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model



@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]

    if dataset_config["name"] in ['sbm', 'comm20', 'planar', 'enzymes']:
        from datasets.spectre_dataset_multi import SpectreGraphDataModule, SpectreDatasetInfos
        from analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
        from analysis.visualization import NonMolecularVisualization

        if dataset_config['name'] != 'enzymes':
            datamodule = SpectreGraphDataModule(cfg, cond_type=condi_config.condition_target)

        if dataset_config['name'] == 'enzymes':
            from datasets.enzymes_dataset import SpectreGraphDataModule, SpectreDatasetInfos
            datamodule = SpectreGraphDataModule(cfg, cond_type=condi_config.condition_target)
        if dataset_config['name'] == 'sbm':
            sampling_metrics = SBMSamplingMetrics(datamodule)
        elif dataset_config['name'] == 'comm20':
            sampling_metrics = Comm20SamplingMetrics(datamodule)
        else:
            sampling_metrics = PlanarSamplingMetrics(datamodule)

        dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)
        train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
        visualization_tools = NonMolecularVisualization()

        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}


    elif dataset_config["name"] == 'dimacs':
        from datasets.dimacs_dataset import DIMACSDataModule, DIMACSDatasetInfos
        from analysis.spectre_utils import SBMSamplingMetrics
        from analysis.visualization import NonMolecularVisualization

        datamodule = DIMACSDataModule(cfg)
        sampling_metrics = SBMSamplingMetrics(datamodule)
        dataset_infos = DIMACSDatasetInfos(datamodule.datasets)
        train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
        visualization_tools = NonMolecularVisualization()

        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}

    elif dataset_config["name"] == 'mis':
        from datasets.mis_dataset import MISDataModule, MISDatasetInfos
        from analysis.spectre_utils import SBMSamplingMetrics
        from analysis.visualization import NonMolecularVisualization

        datamodule = MISDataModule(cfg)
        sampling_metrics = SBMSamplingMetrics(datamodule)
        dataset_infos = MISDatasetInfos(datamodule.datasets)
        train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
        visualization_tools = NonMolecularVisualization()

        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}

    elif dataset_config["name"] == 'tsp':
        from datasets.tsp_dataset import TSPDataModule, TSPDatasetInfos
        from analysis.spectre_utils import TSPSamplingMetrics
        from analysis.visualization import WeightedVisualization

        datamodule = TSPDataModule(cfg)
        sampling_metrics = TSPSamplingMetrics(datamodule)
        dataset_infos = TSPDatasetInfos(datamodule.datasets)
        train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
        visualization_tools = WeightedVisualization()

        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}

    elif dataset_config["name"] in ['qm9', 'guacamol', 'moses']:
        from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
        from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from diffusion.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization

        if dataset_config["name"] == 'qm9':
            from datasets import qm9_dataset
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            train_smiles = qm9_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
                                                        dataset_infos=dataset_infos, evaluate_dataset=False)
        elif dataset_config['name'] == 'guacamol':
            from datasets import guacamol_dataset
            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
            train_smiles = None

        elif dataset_config.name == 'moses':
            from datasets import moses_dataset
            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            train_smiles = None
        else:
            raise ValueError("Dataset not implemented")

        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        if cfg.model.type == 'discrete':
            train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
        else:
            train_metrics = TrainMolecularMetrics(dataset_infos)

        # We do not evaluate novelty during training
        sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
        visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        #print("on:", cfg.general.test_only.split('weights')[0])
        cfg, _ = get_resume(cfg, model_kwargs)
        test_dir = cfg.general.test_only
        # Subir directorios hasta encontrar la carpeta del experimento
        test_dir = os.path.dirname(test_dir)  # Remover archivo .ckpt
        test_dir = os.path.dirname(test_dir)  # Remover carpeta del modelo
        test_dir = os.path.dirname(test_dir)  # Remover carpeta checkpoints
        os.chdir(test_dir)
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        resume_dir = cfg.general.resume
        # Subir directorios hasta encontrar la carpeta del experimento
        resume_dir = os.path.dirname(resume_dir)  # Remover archivo .ckpt
        resume_dir = os.path.dirname(resume_dir)  # Remover carpeta del modelo
        resume_dir = os.path.dirname(resume_dir)  # Remover carpeta checkpoints
        os.chdir(resume_dir)

    utils.create_folders(cfg)

    # Importar el modelo dinámicamente según la configuración
    if cfg.model.type == 'discrete':
        # Verificar si se debe usar el modelo Persist Homo
        use_persist_homo = cfg.model.get('use_persist_homo', False)
        if use_persist_homo:
            from diffusion_model_discrete_persisthomo import DiscreteDenoisingDiffusion
        else:
            from diffusion_model_discrete import DiscreteDenoisingDiffusion
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)

    if cfg.general.guidance_path is not None:
        print(cfg.general.guidance_path)
        if "path" in condi_config.condition_target:
            guidance_model = GraphDistanceModel(hidden_dim=condi_config.HIDDEN_DIM, num_layers=condi_config.NUM_LAYERS, dropout=condi_config.DROPOUT)
        else:
            guidance_out_dim = dataset_infos.output_dims['y']
            guidance_model = GraphStructModel(in_dim=1, hidden_dim=condi_config.HIDDEN_DIM, num_layers=condi_config.NUM_LAYERS,
                                              dropout=condi_config.DROPOUT, out_dim=guidance_out_dim)

        guidance_path = to_absolute_path(cfg.general.guidance_path)
        if os.path.isdir(guidance_path):
            guidance_model_path = os.path.join(guidance_path, f"CLASSIFIER_struct_{condi_config.condition_target[0]}_community.pth")
        else:
            guidance_model_path = guidance_path

        if os.path.exists(guidance_model_path):
            guidance_model.load_state_dict(torch.load(guidance_model_path))
            model.assign_guidance_model(guidance_model)
        else:
            print(f"[WARNING] Guidance model not found: {guidance_model_path}")
    callbacks = []
    if cfg.train.save_model:
        ckpt_dir = os.path.join(os.getcwd(), f"checkpoints/{cfg.general.name}")
        os.makedirs(ckpt_dir, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=ckpt_dir, filename='last', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    # Force CPU execution
    accelerator = 'cpu'
    devices = 1

    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy="auto",
                      accelerator=accelerator,
                      devices=devices,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=False,
                      callbacks=callbacks,
                      log_every_n_steps=50 if name != 'debug' else 1,
                      logger = [])

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(model, datamodule=datamodule)
    else:
        # Start by evaluating test_only_path
        ckpt = torch.load(cfg.general.test_only)
        state_dict = ckpt["state_dict"]
        # 过滤掉 guidance model 的参数
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("guidance_model.")}
        model.load_state_dict(filtered_state_dict, strict=False)
        trainer.test(model, datamodule=datamodule)
        # trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12348'
    
    import torch.distributed as dist
    # Use 'gloo' backend for CPU/Mac instead of 'nccl' (GPU only)
    backend = 'gloo' if not torch.cuda.is_available() else 'nccl'
    dist.init_process_group(backend=backend, rank=0, world_size=1)
    main()
