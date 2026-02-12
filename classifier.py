# import json
# import numpy as np
# import fastjet as fj
# import networkx as nx
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon
# from sklearn.metrics import roc_auc_score, roc_curve


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import HypergraphConv, global_mean_pool
# from torch_geometric.data import Data, Dataset
# from torch_geometric.loader import DataLoader

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
