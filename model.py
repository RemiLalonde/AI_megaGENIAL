from time import process_time_ns
import numpy as np
import torch
import torch.nn as nn
import norse.torch as snn
from norse.torch import LIFParameters
import pandas as pd


class SpikingNN(nn.Module):
    def __init__(self):
        super(SpikingNN, self).__init__()
        # lif_params = LIFParameters(v_th=1.0)  # Set LIF threshold

        self.fc1 = nn.Linear(4, 128)
        self.lif1 = snn.LIFCell()

        self.fc2 = nn.Linear(128, 64)
        self.lif2 = snn.LIFCell()

        self.fc3 = nn.Linear(64, 32)
        self.lif3 = snn.LIFCell()

        self.fc4 = nn.Linear(32, 16)
        self.lif4 = snn.LIFCell()

        self.fc5 = nn.Linear(16, 3)

        # Add QuantStub and DeQuantStub for QAT
        # self.quant = torch.quantization.QuantStub()
        # self.dequant = torch.quantization.DeQuantStub()


    def forward(self, x): # Quantize input
        batch_size = x.size(0)
        seq_len = x.size(1)
        outputs = []
        outputs_potential = []
        lif_state1, lif_state2, lif_state3, lif_state4 = None, None, None, None

        for t in range(seq_len):
            row = x[:, t, :]

            z1 = self.fc1(row)
            z1, lif_state1 = self.lif1(z1, lif_state1)
            z2 = self.fc2(z1)
            z2, lif_state2 = self.lif2(z2, lif_state2)
            #print(lif_state2.v)
            z3 = self.fc3(z2)
            z3, lif_state3 = self.lif3(z3, lif_state3)

            z4 = self.fc4(z3)
            z4, lif_state4 = self.lif4(z4, lif_state4)

            z5 = self.fc5(z4)
            outputs.append(z5)

            # Collect potentials and output in the current timestep
            m = torch.concat((lif_state1.v[0], lif_state2.v[0], lif_state3.v[0], lif_state4.v[0], z5[0]), dim=0).numpy()
            m = (m * 127).astype(int)
            outputs_potential.append(m)

        # Quantize weights after forward pass

        # Process collected potentials
        outputs_potential = np.array(outputs_potential).T
        df = pd.DataFrame(outputs_potential)
        df.to_csv('outputs_potential_left_side_seizure_index_10.csv', index=False)

        output = torch.stack(outputs, dim=1).sum(dim=1)
        return output  # Dequantize output
