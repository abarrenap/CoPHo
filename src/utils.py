import os
import re
import torch_geometric.utils
from omegaconf import OmegaConf, open_dict
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch
import omegaconf
import wandb


def create_folders(args):
    try:
        # os.makedirs('checkpoints')
        os.makedirs('graphs')
        os.makedirs('chains')
    except OSError:
        pass

    try:
        # os.makedirs('checkpoints/' + args.general.name)
        os.makedirs('graphs/' + args.general.name)
        os.makedirs('chains/' + args.general.name)
    except OSError:
        pass


def normalize(X, E, y, norm_values, norm_biases, node_mask):
    X = (X - norm_biases[0]) / norm_values[0]
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def unnormalize(X, E, y, norm_values, norm_biases, node_mask, collapse=False):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X = (X * norm_values[0] + norm_biases[0])
    E = (E * norm_values[1] + norm_biases[1])
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)


def to_dense(x, edge_index, edge_attr, batch):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    # node_mask = node_mask.float()
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case
    max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E


def update_config_with_new_keys(cfg, saved_cfg):
    saved_general = saved_cfg.general
    saved_train = saved_cfg.train
    saved_model = saved_cfg.model

    for key, val in saved_general.items():
        OmegaConf.set_struct(cfg.general, True)
        with open_dict(cfg.general):
            if key not in cfg.general.keys():
                setattr(cfg.general, key, val)

    OmegaConf.set_struct(cfg.train, True)
    with open_dict(cfg.train):
        for key, val in saved_train.items():
            if key not in cfg.train.keys():
                setattr(cfg.train, key, val)

    OmegaConf.set_struct(cfg.model, True)
    with open_dict(cfg.model):
        for key, val in saved_model.items():
            if key not in cfg.model.keys():
                setattr(cfg.model, key, val)
    return cfg


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self


def setup_wandb(cfg):
    if wandb.run is not None:
        return

    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': f'graph_ddm_{cfg.dataset.name}', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True), 'reinit': 'return_previous', 'mode': cfg.general.wandb}

    configured_run_id = cfg.general.get('wandb_run_id', None) if hasattr(cfg.general, 'get') else None
    run_id = configured_run_id
    is_resumed_execution = cfg.general.get('resume', None) is not None or cfg.general.get('test_only', None) is not None

    if run_id is None and is_resumed_execution:
        run_id = _load_wandb_run_id_from_file()
    if run_id is None and is_resumed_execution:
        run_id = _infer_wandb_run_id_from_local_dir()
    if run_id is not None:
        kwargs['id'] = run_id
        kwargs['resume'] = 'allow'

    wandb.init(**kwargs)
    if wandb.run:
        run_id = getattr(wandb.run, 'id', None)
        if run_id is not None:
            with open_dict(cfg.general):
                cfg.general.wandb_run_id = run_id
            _save_wandb_run_id_to_file(run_id)
        # Track all metrics against explicit epoch when provided.
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")
    wandb.save('*.txt')


def _save_wandb_run_id_to_file(run_id: str) -> None:
    run_id_path = os.path.join(os.getcwd(), '.wandb_run_id')
    with open(run_id_path, 'w', encoding='utf-8') as f:
        f.write(run_id)


def _load_wandb_run_id_from_file():
    run_id_path = os.path.join(os.getcwd(), '.wandb_run_id')
    if not os.path.isfile(run_id_path):
        return None

    with open(run_id_path, 'r', encoding='utf-8') as f:
        run_id = f.read().strip()

    if re.fullmatch(r'[A-Za-z0-9]+', run_id):
        return run_id
    return None


def _extract_wandb_run_id(folder_name: str):
    match = re.fullmatch(r'run-\d{8}_\d{6}-([A-Za-z0-9]+)', folder_name)
    if match:
        return match.group(1)
    return None


def _infer_wandb_run_id_from_local_dir():
    wandb_dir = os.path.join(os.getcwd(), 'wandb')
    if not os.path.isdir(wandb_dir):
        return None

    latest_run_link = os.path.join(wandb_dir, 'latest-run')
    if os.path.exists(latest_run_link):
        latest_name = os.path.basename(os.path.realpath(latest_run_link))
        run_id = _extract_wandb_run_id(latest_name)
        if run_id is not None:
            return run_id

    run_dirs = []
    for name in os.listdir(wandb_dir):
        full_path = os.path.join(wandb_dir, name)
        if os.path.isdir(full_path) and name.startswith('run-'):
            run_dirs.append(full_path)
    run_dirs.sort(key=os.path.getmtime, reverse=True)

    for run_dir in run_dirs:
        run_id = _extract_wandb_run_id(os.path.basename(run_dir))
        if run_id is not None:
            return run_id
    return None
