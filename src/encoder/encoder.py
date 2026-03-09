from models.gin import GIN
import torch

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

    use_pretrained = kwargs.get('use_pretrained', False)
    if use_pretrained:
        model_path = kwargs.get('model_path')
        assert model_path is not None, 'Please pass model_path if use_pretrained=True'
        print('loaded', model_path)
        saved_model = torch.load(model_path)
        model.load_state_dict(saved_model['model_state_dict'])

    model.eval()

    if dont_concat:
        model.forward = model.get_graph_embed_no_cat
    else:
        model.forward = model.get_graph_embed

    model.device = device
    return model.to(device)
