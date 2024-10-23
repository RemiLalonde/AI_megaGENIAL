import torch
import norse.torch as snn

# Define your LIFCell
lif_cell = snn.LIFCell()

# Print the parameters of the LIFCell
print(lif_cell.p.tau_mem_inv.item())