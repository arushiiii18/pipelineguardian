# Clean version — seeded torch
import torch
import torch.nn as nn

torch.manual_seed(42)
model = nn.Linear(10, 1)
x = torch.randn(4, 10)
out = model(x)
