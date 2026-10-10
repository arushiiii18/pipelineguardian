"""
GPU Smoke Test & Verification for Learned Leakage Detector.
Ensures PyTorch is installed with CUDA support, the NVIDIA RTX 2050 GPU is active,
and forward/backward neural network operations execute on CUDA without falling back to CPU.
"""

import json
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim


def run_gpu_smoke_test(report_path: str = None) -> dict:
    results = {
        "interpreter": sys.executable,
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": None,
        "device_capability": None,
        "total_memory_bytes": None,
        "tensor_ops_passed": False,
        "nn_forward_backward_passed": False,
        "params_on_cuda": False,
        "gradient_step_passed": False,
    }

    if not torch.cuda.is_available():
        results["error"] = "CUDA is not available in PyTorch. Refusing CPU fallback."
        return results

    device = torch.device("cuda:0")
    results["device_name"] = torch.cuda.get_device_name(0)
    results["device_capability"] = torch.cuda.get_device_capability(0)
    results["total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory

    # 1. Basic tensor ops on CUDA
    a = torch.randn(100, 100, device=device)
    b = torch.randn(100, 100, device=device)
    c = torch.matmul(a, b)
    assert c.is_cuda, "Tensor computation must be on CUDA"
    results["tensor_ops_passed"] = True

    # 2. Neural Network Forward and Backward
    class TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(32, 16)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(16, 2)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    model = TinyMLP().to(device)

    # Verify all parameters are on CUDA
    all_cuda = all(p.is_cuda and p.device.type == "cuda" for p in model.parameters())
    assert all_cuda, "All model parameters must reside on CUDA"
    results["params_on_cuda"] = True

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Track initial weights
    initial_weight = model.fc1.weight.clone().detach()

    # Synthetic batch
    x = torch.randn(16, 32, device=device)
    y = torch.randint(0, 2, (16,), device=device)
    assert x.is_cuda and y.is_cuda, "Batch tensors must reside on CUDA"

    # Forward
    out = model(x)
    loss = criterion(out, y)
    assert torch.isfinite(loss).item(), "Loss must be finite"

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Check gradients exist and are finite
    grads_valid = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert grads_valid, "All gradients must be computed and finite"
    results["nn_forward_backward_passed"] = True

    # Optimizer step
    optimizer.step()
    weight_changed = not torch.equal(initial_weight, model.fc1.weight)
    assert weight_changed, "Parameters must update after optimizer step"
    results["gradient_step_passed"] = True

    # Memory statistics
    results["allocated_memory_bytes"] = torch.cuda.memory_allocated(0)
    results["reserved_memory_bytes"] = torch.cuda.memory_reserved(0)

    if report_path:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results



def test_cuda_smoke():
    res = run_gpu_smoke_test()
    assert res["cuda_available"] is True
    assert res["tensor_ops_passed"] is True
    assert res["params_on_cuda"] is True
    assert res["nn_forward_backward_passed"] is True
    assert res["gradient_step_passed"] is True


if __name__ == "__main__":
    rep_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "reports",
        "gpu_verification.json"
    )
    res = run_gpu_smoke_test(report_path=rep_path)
    print(json.dumps(res, indent=2))
    if not res.get("gradient_step_passed", False):
        sys.exit(1)

