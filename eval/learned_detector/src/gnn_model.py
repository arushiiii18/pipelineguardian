"""
gnn_model.py - Plain-PyTorch Graph Isomorphism Network (GIN) for ML Leakage Detection.
Executes on NVIDIA RTX 2050 GPU (CUDA) without PyTorch Geometric dependencies.
Features GIN message passing, masked graph pooling, checkpointing, and feasibility gate test.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn

import torch.optim as optim
import numpy as np

# Ensure src in sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from graph_builder import build_ast_graph, collate_graph_batch, NODE_FEAT_DIM
from data_loader import get_train_val_and_test_splits
from metrics import evaluate_predictions
from sklearn.model_selection import GroupKFold


class GINLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU()
        )
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, h: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: [B, N, in_dim], adj: [B, N, N], mask: [B, N, 1]
        agg = torch.bmm(adj, h) + self.eps * h
        B, N, D = agg.shape
        out = self.mlp(agg.view(B * N, D)).view(B, N, -1)
        return out * mask


class PlainPyTorchGIN(nn.Module):
    def __init__(
        self,
        in_dim: int = NODE_FEAT_DIM,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            GINLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, N, in_dim]
        h = torch.relu(self.input_proj(x)) * mask

        for layer in self.layers:
            h = layer(h, adj, mask)

        # Masked Global Mean Pooling
        # sum over node dimension
        sum_pooled = torch.sum(h * mask, dim=1)  # [B, hidden_dim]
        num_nodes = torch.sum(mask, dim=1).clamp(min=1.0)  # [B, 1]
        graph_emb = sum_pooled / num_nodes

        graph_emb = self.dropout(graph_emb)
        logits = self.classifier(graph_emb)
        return logits


def run_gnn_feasibility_gate(device: str = "cuda") -> Dict[str, Any]:
    """
    Mandatory Day 6 Feasibility Gate:
    Tests tiny batch forward/backward pass, finite loss, weight update, overfit capability,
    and checkpoint save/load on NVIDIA GPU.
    """
    res = {
        "gate_passed": False,
        "device_used": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "overfit_passed": False,
        "weights_updated": False,
        "checkpoint_passed": False,
        "vram_allocated_mb": None,
        "duration_sec": None
    }

    start_time = time.time()
    dummy_codes = [
        "scaler.fit(X)\nX_tr, X_te = split(X)",
        "X_tr, X_te = split(X)\nscaler.fit(X_tr)",
        "imputer.fit(X)\nX_tr, X_te = split(X)",
        "X_tr, X_te = split(X)\nimputer.fit(X_tr)"
    ]
    dummy_labels = torch.tensor([1, 0, 1, 0], dtype=torch.long, device=device)

    graphs = [build_ast_graph(c) for c in dummy_codes]
    batch_x, batch_adj, batch_mask = collate_graph_batch(graphs, device=device)

    model = PlainPyTorchGIN(hidden_dim=32, num_layers=2).to(device)
    assert next(model.parameters()).is_cuda, "Model must reside on CUDA"

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    initial_param = next(model.parameters()).clone().detach()

    # Overfit loop (max 60 steps)
    overfitted = False
    for step in range(60):
        optimizer.zero_grad()
        logits = model(batch_x, batch_adj, batch_mask)
        loss = criterion(logits, dummy_labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        if torch.equal(preds, dummy_labels):
            overfitted = True
            break

    param_changed = not torch.equal(initial_param, next(model.parameters()))
    res["overfit_passed"] = overfitted
    res["weights_updated"] = param_changed

    # Test checkpoint save and load
    ckpt_path = os.path.join(SRC_DIR, "tiny_gate_ckpt.pt")
    torch.save(model.state_dict(), ckpt_path)
    model_loaded = PlainPyTorchGIN(hidden_dim=32, num_layers=2).to(device)
    model_loaded.load_state_dict(torch.load(ckpt_path, weights_only=True))
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    res["checkpoint_passed"] = True


    if torch.cuda.is_available():
        res["vram_allocated_mb"] = float(torch.cuda.memory_allocated(0) / (1024 * 1024))

    res["duration_sec"] = time.time() - start_time
    res["gate_passed"] = overfitted and param_changed and res["checkpoint_passed"]
    return res


def train_gnn_epoch(model, optimizer, criterion, graphs_data, labels, batch_size=32, device="cuda"):
    model.train()
    indices = np.random.permutation(len(graphs_data))
    total_loss = 0.0

    for i in range(0, len(graphs_data), batch_size):
        batch_idx = indices[i:i + batch_size]
        sub_graphs = [graphs_data[j] for j in batch_idx]
        sub_labels = torch.tensor(labels[batch_idx], dtype=torch.long, device=device)

        bx, badj, bmask = collate_graph_batch(sub_graphs, device=device)

        optimizer.zero_grad()
        logits = model(bx, badj, bmask)
        loss = criterion(logits, sub_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_idx)

    return total_loss / len(graphs_data)


def eval_gnn(model, graphs_data, labels, batch_size=32, device="cuda"):
    model.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(graphs_data), batch_size):
            sub_graphs = graphs_data[i:i + batch_size]
            bx, badj, bmask = collate_graph_batch(sub_graphs, device=device)
            logits = model(bx, badj, bmask)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs)

    return np.array(all_preds, dtype=int), np.array(all_probs, dtype=float)


def run_grouped_cv_gnn(graphs_data, labels, groups, config, device="cuda", n_splits=5, epochs=15):
    gkf = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros(len(labels), dtype=int)
    oof_probs = np.zeros(len(labels), dtype=float)
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(graphs_data, labels, groups)):
        tr_graphs = [graphs_data[i] for i in train_idx]
        tr_labels = labels[train_idx]
        val_graphs = [graphs_data[i] for i in val_idx]
        val_labels = labels[val_idx]

        model = PlainPyTorchGIN(
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"]
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        for ep in range(epochs):
            train_gnn_epoch(model, optimizer, criterion, tr_graphs, tr_labels, batch_size=32, device=device)

        preds, probs = eval_gnn(model, val_graphs, val_labels, batch_size=32, device=device)
        oof_preds[val_idx] = preds
        oof_probs[val_idx] = probs

        val_eval = evaluate_predictions(val_labels, preds, probs)
        fold_f1s.append(val_eval["macro_f1"])

    oof_eval = evaluate_predictions(labels, oof_preds, oof_probs)
    oof_eval["fold_macro_f1s"] = fold_f1s
    oof_eval["fold_f1_std"] = float(np.std(fold_f1s))
    return oof_eval, oof_preds, oof_probs


def run_gnn_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing GNN on device: {device} (GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Feasibility Gate
    gate_res = run_gnn_feasibility_gate(device=device)
    print("Feasibility Gate Result:")
    print(json.dumps(gate_res, indent=2))
    assert gate_res["gate_passed"], "GNN Feasibility Gate Failed!"

    # 2. Data Loading & Graph Parsing
    splits = get_train_val_and_test_splits()
    train_val = splits["train_val"]
    held_out = splits["held_out_test"]

    print("Parsing AST graphs for train/val pool...")
    tr_graphs = [build_ast_graph(c) for c in train_val["texts"]]
    print("Parsing AST graphs for held-out test set...")
    te_graphs = [build_ast_graph(c) for c in held_out["texts"]]

    # 3. Tuning Grid
    param_grid = [
        {"hidden_dim": 32, "num_layers": 2, "dropout": 0.1, "lr": 0.005},
        {"hidden_dim": 32, "num_layers": 3, "dropout": 0.1, "lr": 0.005},
        {"hidden_dim": 64, "num_layers": 2, "dropout": 0.2, "lr": 0.003},
    ]

    tuning_records = []
    best_config = None
    best_macro_f1 = -1.0
    best_oof_eval = None

    print(f"\nStarting GNN tuning across {len(param_grid)} configurations...")
    for idx, cfg in enumerate(param_grid):
        start_time = time.time()
        oof_eval, _, _ = run_grouped_cv_gnn(tr_graphs, train_val["labels"], train_val["groups"], cfg, device=device, epochs=12)
        duration = time.time() - start_time

        record = {
            "trial_id": idx + 1,
            "config": cfg,
            "macro_f1": oof_eval["macro_f1"],
            "fold_std": oof_eval["fold_f1_std"],
            "positive_f1": oof_eval["positive_class"]["f1"],
            "negative_f1": oof_eval["negative_class"]["f1"],
            "pr_auc": oof_eval.get("pr_auc"),
            "duration_sec": duration
        }
        tuning_records.append(record)
        print(f"Trial {idx+1}: cfg={cfg} -> Macro-F1={oof_eval['macro_f1']:.4f} (std={oof_eval['fold_f1_std']:.4f}) in {duration:.2f}s")

        if oof_eval["macro_f1"] > best_macro_f1:
            best_macro_f1 = oof_eval["macro_f1"]
            best_config = cfg
            best_oof_eval = oof_eval

    # 4. Final Training on Full Train/Val Pool
    print(f"\nTraining final GNN model with best config: {best_config}")
    start_fit = time.time()
    final_model = PlainPyTorchGIN(
        hidden_dim=best_config["hidden_dim"],
        num_layers=best_config["num_layers"],
        dropout=best_config["dropout"]
    ).to(device)
    optimizer = optim.AdamW(final_model.parameters(), lr=best_config["lr"], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for ep in range(15):
        train_gnn_epoch(final_model, optimizer, criterion, tr_graphs, train_val["labels"], batch_size=32, device=device)
    fit_duration = time.time() - start_fit

    # 5. Final Evaluation on Synthetic Held-Out Test Set
    start_infer = time.time()
    held_out_preds, held_out_probs = eval_gnn(final_model, te_graphs, held_out["labels"], batch_size=32, device=device)
    infer_duration = time.time() - start_infer

    held_out_eval = evaluate_predictions(
        held_out["labels"],
        held_out_preds,
        held_out_probs,
        records=held_out["records"]
    )
    print(f"\nGNN Synthetic Held-Out Test Macro-F1 = {held_out_eval['macro_f1']:.4f}")
    print(f"Held-out Confusion Matrix: {held_out_eval['confusion_matrix']}")

    # 6. Save Artifacts
    res_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, "gnn_gin.pt")
    torch.save(final_model.state_dict(), ckpt_path)

    full_report = {
        "model_name": "Plain-PyTorch GIN",
        "feasibility_gate": gate_res,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "best_config": best_config,
        "fit_time_sec": fit_duration,
        "infer_time_sec": infer_duration,
        "grouped_cv_eval": best_oof_eval,
        "held_out_test_eval": held_out_eval,
        "tuning_history": tuning_records
    }

    report_path = os.path.join(res_dir, "gnn_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"Saved artifacts to {report_path} and {ckpt_path}")
    return full_report


if __name__ == "__main__":
    run_gnn_experiment()
