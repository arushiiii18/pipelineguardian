"""
Tests for Plain-PyTorch GIN Model and Graph Construction.
Verifies AST graph dimensions, forward and backward pass, and the feasibility gate.
"""

import os
import sys
import pytest
import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from graph_builder import build_ast_graph, collate_graph_batch, NODE_FEAT_DIM
from gnn_model import PlainPyTorchGIN, run_gnn_feasibility_gate


def test_graph_builder():
    code = "import numpy as np\nX = np.random.randn(10, 2)\nscaler.fit(X)"
    node_feats, adj, n = build_ast_graph(code, max_nodes=50)

    assert node_feats.shape == (50, NODE_FEAT_DIM)
    assert adj.shape == (50, 50)
    assert n > 0
    assert adj[0, 0] == 1.0  # self-loop present


def test_gnn_forward_backward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graphs = [build_ast_graph("scaler.fit(X)"), build_ast_graph("scaler.fit(X_train)")]
    bx, badj, bmask = collate_graph_batch(graphs, device=device)

    model = PlainPyTorchGIN(hidden_dim=16, num_layers=2).to(device)
    logits = model(bx, badj, bmask)

    assert logits.shape == (2, 2)
    loss = logits.sum()
    loss.backward()

    # Verify gradients computed
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad


def test_gnn_feasibility_gate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = run_gnn_feasibility_gate(device=device)
    assert res["gate_passed"] is True
    assert res["overfit_passed"] is True
    assert res["weights_updated"] is True
