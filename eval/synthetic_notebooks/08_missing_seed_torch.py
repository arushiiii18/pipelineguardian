# Injected issue: missing_seed_torch
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
x = torch.randn(4, 10)  # unseeded — no torch.manual_seed anywhere
out = model(x)
