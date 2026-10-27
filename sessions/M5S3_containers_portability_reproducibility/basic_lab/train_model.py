"""Fit a straight line inside the existing PyTorch container."""
import json
import sys
import torch

# Read the two experiment parameters from the command line.
seed = int(sys.argv[1])
true_slope = float(sys.argv[2])

# Fix the random initialisation and keep the CPU workload within our allocation.
torch.manual_seed(seed)
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)

# Generate the same 256 observations whenever the slope parameter is unchanged.
x = torch.linspace(-2, 2, 256).reshape(-1, 1)
y = true_slope * x + 2.0 + 0.05 * torch.sin(9 * x)

# The model learns one slope and one intercept.
model = torch.nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

for epoch in range(240):
    optimizer.zero_grad()                 # Discard the previous gradients.
    prediction = model(x)
    loss = torch.mean((prediction - y) ** 2)
    loss.backward()                      # Calculate the new gradients.
    optimizer.step()                     # Update the slope and intercept.

# Inspect the learned values without creating more gradient calculations.
with torch.no_grad():
    slope = model.weight.item()
    intercept = model.bias.item()
    error = torch.mean(torch.abs(model(x) - y)).item()

# A program finishing is not enough: check that it learned the intended line.
assert abs(slope - true_slope) < 0.1
assert abs(intercept - 2.0) < 0.1
assert error < 0.1

# Stable, sorted output lets us compare separate executions with cmp.
result = {
    "torch_version": torch.__version__,
    "seed": seed,
    "target_slope": true_slope,
    "learned_slope": round(slope, 6),
    "learned_intercept": round(intercept, 6),
    "mean_absolute_error": round(error, 6),
}
print(json.dumps(result, sort_keys=True))
