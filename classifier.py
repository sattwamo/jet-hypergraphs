import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, global_mean_pool


class LundHGNN(nn.Module):
    def __init__(self, in_channels=5, hidden=64):
        super().__init__()

        self.lin_in = nn.Linear(in_channels, hidden)

        self.hg1 = HypergraphConv(hidden, hidden)
        self.hg2 = HypergraphConv(hidden, hidden)

        self.lin_out = nn.Linear(hidden, 1)

    def forward(self, x, hyperedge_index, batch):
        x = self.lin_in(x)
        x = F.sigmoid(x)

        x = self.hg1(x, hyperedge_index)
        x = F.sigmoid(x)

        x = self.hg2(x, hyperedge_index)
        x = F.sigmoid(x)

        x = global_mean_pool(x, batch)
        return self.lin_out(x).squeeze(-1)
