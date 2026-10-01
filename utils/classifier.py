import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    def __init__(self, path):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                feature = torch.tensor(item["feature"], dtype=torch.float32)
                label = torch.tensor(float(item["label"]))
                self.samples.append((feature, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class ScoringNetwork(nn.Module):
    def __init__(self, input_dim=384):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)
