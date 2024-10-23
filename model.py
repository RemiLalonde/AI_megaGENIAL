from time import process_time_ns

import torch
import torch.nn as nn
import norse.torch as snn
from norse.torch import LIFParameters


class SpikingNN(nn.Module):
    def __init__(self):
        super(SpikingNN, self).__init__()

        self.fc1 = nn.Linear(4, 512)
        self.lif1 = snn.LIFCell()

        self.fc2 = nn.Linear(512, 256)
        self.lif2 = snn.LIFCell()

        self.fc3 = nn.Linear(256, 128)
        self.lif3 = snn.LIFCell()

        self.fc4 = nn.Linear(128, 64)
        self.lif4 = snn.LIFCell()

        self.fc5 = nn.Linear(64, 4)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        outputs = []
        lif_state1, lif_state2, lif_state3, lif_state4 = None, None, None, None

        for t in range(seq_len):
            row = x[:, t, :]
            z1 = self.fc1(row)
            z1, lif_state1 = self.lif1(z1, lif_state1)
            z2 = self.fc2(z1)
            z2, lif_state2 = self.lif2(z2, lif_state2)
            z3 = self.fc3(z2)
            z3, lif_state3 = self.lif3(z3, lif_state3)
            z4 = self.fc4(z3)
            z4, lif_state4 = self.lif4(z4, lif_state4)
            z5 = self.fc5(z4)
            outputs.append(z5)

        output = torch.stack(outputs, dim=1).sum(dim=1)
        return output