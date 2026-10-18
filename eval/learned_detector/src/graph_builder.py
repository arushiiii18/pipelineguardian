"""
graph_builder.py - AST Data-flow graph representation for Python ML pipelines.
Constructs plain-PyTorch graph tensors (node feature matrix, adjacency matrix, node mask)
without requiring PyTorch Geometric.
"""

import ast
from typing import List, Dict, Any, Tuple
import torch
import numpy as np

# Node Type Vocabularies
NODE_TYPES = [
    "Module", "FunctionDef", "Assign", "Call", "Name", "Attribute",
    "Constant", "Import", "ImportFrom", "Subscript", "Expr", "Other"
]
NODE_TYPE_TO_IDX = {t: i for i, t in enumerate(NODE_TYPES)}

API_CATEGORIES = [
    "split_api", "fit_api", "transform_api", "scaler_api",
    "imputer_api", "pipeline_api", "data_var", "general"
]
API_CAT_TO_IDX = {c: i for i, c in enumerate(API_CATEGORIES)}

NODE_FEAT_DIM = len(NODE_TYPES) + len(API_CATEGORIES)  # 12 + 8 = 20


def _get_api_category(node: ast.AST) -> str:
    name_str = ""
    if isinstance(node, ast.Name):
        name_str = node.id
    elif isinstance(node, ast.Attribute):
        name_str = node.attr
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            name_str = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name_str = node.func.attr

    name_lower = name_str.lower()
    if "train_test_split" in name_lower:
        return "split_api"
    if name_str in ["fit", "fit_transform"]:
        return "fit_api"
    if name_str == "transform":
        return "transform_api"
    if any(s in name_lower for s in ["scaler", "normalizer"]):
        return "scaler_api"
    if "imputer" in name_lower:
        return "imputer_api"
    if any(p in name_lower for p in ["pipeline", "columntransformer"]):
        return "pipeline_api"
    if name_str in ["X", "X_train", "X_test", "y", "y_train", "y_test", "df", "data", "X_combined"]:
        return "data_var"
    return "general"


def build_ast_graph(code_str: str, max_nodes: int = 120) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Parses code_str into AST graph.
    Returns:
    - node_features: [max_nodes, NODE_FEAT_DIM]
    - adj_matrix: [max_nodes, max_nodes] (symmetric adjacency + self-loops)
    - num_nodes: actual node count (clipped to max_nodes)
    """
    try:
        tree = ast.parse(code_str)
    except Exception:
        tree = ast.parse("pass")

    node_list = []
    edges = []

    def traverse(curr, parent_idx=None):
        nonlocal node_list, edges
        if len(node_list) >= max_nodes:
            return
        curr_idx = len(node_list)
        node_list.append(curr)

        if parent_idx is not None:
            edges.append((parent_idx, curr_idx))
            edges.append((curr_idx, parent_idx))

        for child in ast.iter_child_nodes(curr):
            traverse(child, curr_idx)

    traverse(tree)
    num_nodes = len(node_list)

    # Feature matrix
    node_feats = np.zeros((max_nodes, NODE_FEAT_DIM), dtype=np.float32)
    adj = np.zeros((max_nodes, max_nodes), dtype=np.float32)

    for i, node in enumerate(node_list):
        ntype = type(node).__name__
        ntype_idx = NODE_TYPE_TO_IDX.get(ntype, NODE_TYPE_TO_IDX["Other"])
        node_feats[i, ntype_idx] = 1.0

        api_cat = _get_api_category(node)
        api_idx = API_CAT_TO_IDX[api_cat]
        node_feats[i, len(NODE_TYPES) + api_idx] = 1.0

        # Self-loop
        adj[i, i] = 1.0

    # Sequential flow edges between top-level statements
    for i in range(min(num_nodes - 1, 30)):
        edges.append((i, i + 1))
        edges.append((i + 1, i))

    for u, v in edges:
        if u < max_nodes and v < max_nodes:
            adj[u, v] = 1.0
            adj[v, u] = 1.0

    return node_feats, adj, num_nodes


def collate_graph_batch(graphs_data: List[Tuple[np.ndarray, np.ndarray, int]], device: str = "cpu"):
    """
    Collates a list of graph tuples into batched PyTorch CUDA tensors.
    """
    B = len(graphs_data)
    max_nodes = graphs_data[0][0].shape[0]
    feat_dim = graphs_data[0][0].shape[1]

    batch_x = np.zeros((B, max_nodes, feat_dim), dtype=np.float32)
    batch_adj = np.zeros((B, max_nodes, max_nodes), dtype=np.float32)
    batch_mask = np.zeros((B, max_nodes, 1), dtype=np.float32)

    for i, (x, adj, n) in enumerate(graphs_data):
        batch_x[i] = x
        batch_adj[i] = adj
        batch_mask[i, :n] = 1.0

    x_tensor = torch.tensor(batch_x, dtype=torch.float32, device=device)
    adj_tensor = torch.tensor(batch_adj, dtype=torch.float32, device=device)
    mask_tensor = torch.tensor(batch_mask, dtype=torch.float32, device=device)

    return x_tensor, adj_tensor, mask_tensor
