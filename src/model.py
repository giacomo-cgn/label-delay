from collections import OrderedDict
import src.resnet as resnet
import torch

def init_model(
    device,
    model_name='resnet50',
    use_pred=False,
    output_dim=128,
    cifar=False,
    num_classes=100,
    detach=None
):
    assert detach is not None
    encoder = resnet.__dict__[model_name](cifar=cifar, num_classes=num_classes, detach=detach)
    if model_name == 'resnet18':
        hidden_dim = 512
    else:
        hidden_dim = 2048
        if 'w2' in model_name:
            hidden_dim *= 2
        elif 'w4' in model_name:
            hidden_dim *= 4

    # -- projection head
    encoder.fc = torch.nn.Sequential(OrderedDict([
        ('fc1', torch.nn.Linear(hidden_dim, hidden_dim)),
        ('bn1', torch.nn.BatchNorm1d(hidden_dim)),
        ('relu1', torch.nn.ReLU(inplace=True)),
        ('fc2', torch.nn.Linear(hidden_dim, hidden_dim)),
        ('bn2', torch.nn.BatchNorm1d(hidden_dim)),
        ('relu2', torch.nn.ReLU(inplace=True)),
        ('fc3', torch.nn.Linear(hidden_dim, output_dim))
    ]))

    # -- projection head for feature alignment
    encoder.feat_proj = torch.nn.Sequential(OrderedDict([
        ('fc1', torch.nn.Linear(output_dim, output_dim)),
    ]))


    # -- prediction head
    encoder.pred = None
    if use_pred:
        mx = 4  # 4x bottleneck prediction head
        pred_head = OrderedDict([])
        pred_head['bn1'] = torch.nn.BatchNorm1d(output_dim)
        pred_head['fc1'] = torch.nn.Linear(output_dim, output_dim//mx)
        pred_head['bn2'] = torch.nn.BatchNorm1d(output_dim//mx)
        pred_head['relu'] = torch.nn.ReLU(inplace=True)
        pred_head['fc2'] = torch.nn.Linear(output_dim//mx, output_dim)
        encoder.pred = torch.nn.Sequential(pred_head)

    encoder.to(device)
    return encoder