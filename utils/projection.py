import torch.nn as nn
import torch.nn.functional as F


class Reducer(nn.Module):
    def __init__(self, input_dim=768, hidden_dims=[512, 384, 256], output_dim=128, use_l2_norm=True):

        super().__init__()
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.use_l2_norm = use_l2_norm

    def forward(self, x):
        out = self.network(x)
        if self.use_l2_norm:
            out = F.normalize(out, p=2, dim=-1)
        return out


class ProjectionNet(nn.Module):
    def __init__(self, input_dim=4096, hidden_dims=[1024, 512, 256], output_dim=128):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        return F.normalize(out, p=2, dim=-1)
