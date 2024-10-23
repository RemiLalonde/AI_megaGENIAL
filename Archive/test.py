import torch

# Given parameters
tau_mem_inv = torch.tensor(200.0)  # Example value of tau_mem_inv (1/seconds)
dt = 1e-3  # Time step of 1 ms (you can adjust this based on your simulation)

# Calculate beta
beta = torch.exp(-dt * tau_mem_inv)
beta2 = 1 - (dt * tau_mem_inv)
print(f"Calculated beta: {beta.item()}")
print(beta2)