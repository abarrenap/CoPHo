from models.gin import GIN
import torch
import os

DEFAULT_GIN_WEIGHTS = "/Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/src/weights/random_gin.pt"

def load_feature_extractor(
    device, num_layers=3, hidden_dim=35, neighbor_pooling_type='sum',
        graph_pooling_type='sum', input_dim=1, edge_feat_dim=0,
        dont_concat=False, num_mlp_layers=2, output_dim=1,
        node_feat_loc='attr', edge_feat_loc='attr', init='orthogonal',
        **kwargs):

    model = GIN(
        num_layers=num_layers, hidden_dim=hidden_dim,
        neighbor_pooling_type=neighbor_pooling_type,
        graph_pooling_type=graph_pooling_type, input_dim=input_dim,
        edge_feat_dim=edge_feat_dim, num_mlp_layers=num_mlp_layers,
        output_dim=output_dim, init=init)

    model.node_feat_loc = node_feat_loc
    model.edge_feat_loc = edge_feat_loc

    use_pretrained = kwargs.get('use_pretrained', True)
    if use_pretrained:
        model_path = kwargs.get('model_path', DEFAULT_GIN_WEIGHTS)
        assert model_path is not None, 'Please pass model_path if use_pretrained=True'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GIN pretrained weights not found at: {model_path}")
        print('loaded', model_path)
        saved_model = torch.load(model_path, map_location=device)
        state_dict = saved_model.get('model_state_dict', saved_model) if isinstance(saved_model, dict) else saved_model
        model_state = model.state_dict()
        compatible_state = {
            k: v for k, v in state_dict.items()
            if k in model_state and model_state[k].shape == v.shape
        }
        model_state.update(compatible_state)
        model.load_state_dict(model_state)
        print(f"GIN params loaded: {len(compatible_state)}/{len(model_state)}")

    model.eval()

    if dont_concat:
        model.forward = model.get_graph_embed_no_cat
    else:
        model.forward = model.get_graph_embed

    model.device = device
    return model.to(device)
