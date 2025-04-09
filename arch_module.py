from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
import torchvision.models.detection.transform
import torchvision.ops.boxes as box_ops

class MLP(nn.Module):
    def __init__(self, input_dim, num_hiddens, output_dim, nlayers):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, num_hiddens))
        for i in range(nlayers - 2):
            self.layers.append(nn.Linear(num_hiddens, num_hiddens))
        self.layers.append(nn.Linear(num_hiddens, output_dim))
        self.input_dim = input_dim

        return

    def forward(self, x):

        out = x.view(-1, self.input_dim)
        for i in range(len(self.layers) - 1):
            out = self.layers[i](out)
            out = F.relu(out)
        out = self.layers[-1](out)

        return out


class clf(nn.Module):

    def __init__(self):
        super(clf, self).__init__()

        self.q = torch.arange(0, 1.00001, 0.1).cuda()
        q_dim = len(self.q)

        self.encoder = MLP(144 * 2, 512, 512, 2)
        self.detector = MLP(q_dim * 512, 512, 2, 2)

        return

    def forward(self, batch):
        lenn = len(batch)
        out = []
        for i in range(lenn):
            out_i = self.encoder(batch[i].cuda())
            out_i = torch.quantile(out_i, self.q, dim=0).contiguous().view(-1)
            out.append(out_i)

        out = torch.stack(out, dim=0)
        out = self.detector(out)
        out = torch.tanh(out) * 10
        return out

    def ano(self, batch):
        out = self.forward(batch)
        return out[:, 1] - out[:, 0]
