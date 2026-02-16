from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union
import json
import torch
import dgl


PathLike = Union[str, Path]
Graph = Dict[str, Union[int, List[float]]]


def load_txt(path: PathLike, *, encoding: str = "utf-8") -> List[str]:
    """Load a .txt file and return non-empty lines.

    Args:
        path: Path to the .txt file.
        encoding: File encoding to use.

    Returns:
        List of non-empty, stripped lines.
    """
    file_path = Path(path)
    if file_path.suffix.lower() != ".txt":
        raise ValueError(f"Expected a .txt file, got: {file_path}")

    with file_path.open("r", encoding=encoding) as handle:
        return [line.strip() for line in handle if line.strip()]


def load_graphs_from_txt(path: PathLike, *, encoding: str = "utf-8") -> List[Graph]:
    """Load multiple graphs from a .txt file.

    Expected format per graph:
        N=<int>
        X:
        <n values possibly spanning multiple lines>
        E:
        <n*n values possibly spanning multiple lines>

    Returns:
        List of dicts with keys: "n", "x", "e". The "e" list is flat length n*n.
    """
    lines = load_txt(path, encoding=encoding)
    graphs: List[Graph] = []
    idx = 0

    while idx < len(lines):
        line = lines[idx]
        if not line.startswith("N="):
            idx += 1
            continue

        n_str = line.split("=", 1)[1].strip()
        if not n_str.isdigit():
            raise ValueError(f"Invalid N value: {line}")
        n = int(n_str)
        idx += 1

        if idx >= len(lines) or lines[idx] != "X:":
            raise ValueError("Expected 'X:' line after N=...")
        idx += 1

        x_tokens: List[str] = []
        while idx < len(lines) and lines[idx] != "E:":
            if lines[idx].startswith("N="):
                raise ValueError("Missing 'E:' section before next graph")
            x_tokens.extend(lines[idx].split())
            idx += 1

        if idx >= len(lines) or lines[idx] != "E:":
            raise ValueError("Expected 'E:' line after X section")
        idx += 1

        e_tokens: List[str] = []
        while idx < len(lines) and not lines[idx].startswith("N="):
            e_tokens.extend(lines[idx].split())
            idx += 1

        if len(x_tokens) != n:
            raise ValueError(f"Expected {n} X values, got {len(x_tokens)}")
        if len(e_tokens) != n * n:
            raise ValueError(f"Expected {n * n} E values, got {len(e_tokens)}")

        x_vals = [float(token) for token in x_tokens]
        e_vals = [float(token) for token in e_tokens]

        graphs.append({"n": n, "x": x_vals, "e": e_vals})

    return graphs


def load_data_from_dir(data_dir: PathLike, *, encoding: str = "utf-8") -> Dict[str, List[str]]:
    """Load all .txt files from a directory.

    Args:
        data_dir: Directory containing .txt files.
        encoding: File encoding to use.

    Returns:
        Mapping from file stem to list of non-empty lines.
    """
    base_dir = Path(data_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {base_dir}")
    if not base_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {base_dir}")

    results: Dict[str, List[str]] = {}
    for txt_path in sorted(base_dir.glob("*.txt")):
        results[txt_path.stem] = load_txt(txt_path, encoding=encoding)

    return results


def load_graphs_from_dir(data_dir: PathLike, *, encoding: str = "utf-8") -> Dict[str, List[Graph]]:
    """Load graphs from all .txt files in a directory.

    Returns:
        Mapping from file stem to list of graphs (see load_graphs_from_txt).
    """
    base_dir = Path(data_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {base_dir}")
    if not base_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {base_dir}")

    results: Dict[str, List[Graph]] = {}
    for txt_path in sorted(base_dir.glob("*.txt")):
        results[txt_path.stem] = load_graphs_from_txt(txt_path, encoding=encoding)

    return results


def load_graphs_from_json(path: PathLike, *, encoding: str = "utf-8") -> List[Graph]:
    """Load graphs from JSON lists of adjacency/weight matrices.

    Expected JSON format:
        [
          [[...], [...], ...],  # graph 1 (NxN)
          [[...], [...], ...],  # graph 2 (MxM)
          ...
        ]

    Returns:
        List of dicts with keys: "n", "x", "e". The "e" list is flat length n*n.
    """
    file_path = Path(path)
    if file_path.suffix.lower() != ".json":
        raise ValueError(f"Expected a .json file, got: {file_path}")

    with file_path.open("r", encoding=encoding) as handle:
        data = json.load(handle)

    if not isinstance(data, list) or not data:
        raise ValueError("Expected a non-empty list of graphs in JSON")

    graphs: List[Graph] = []
    for idx, matrix in enumerate(data):
        if not isinstance(matrix, list) or not matrix:
            raise ValueError(f"Graph {idx} is not a non-empty 2D list")

        n = len(matrix)
        for row in matrix:
            if not isinstance(row, list) or len(row) != n:
                raise ValueError(f"Graph {idx} is not an NxN matrix")

        e_vals = [float(value) for row in matrix for value in row]
        x_vals = [1.0] * n
        graphs.append({"n": n, "x": x_vals, "e": e_vals})

    return graphs


def write_graphs_to_txt(path: PathLike, graphs: List[Graph], *, encoding: str = "utf-8") -> None:
    """Write graphs to a .txt file in N/X/E format.

    Each graph is written as:
        N=<int>
        X:
        <n values on one line>
        E:
        <n*n values on one line>
    """
    file_path = Path(path)
    if file_path.suffix.lower() != ".txt":
        raise ValueError(f"Expected a .txt file, got: {file_path}")

    lines: List[str] = []
    for graph in graphs:
        if "n" not in graph or "x" not in graph or "e" not in graph:
            raise ValueError("Graph must contain 'n', 'x', and 'e' keys")
        n = int(graph["n"])
        x_vals = graph["x"]
        e_vals = graph["e"]

        if not isinstance(x_vals, list) or not isinstance(e_vals, list):
            raise ValueError("Graph 'x' and 'e' must be lists")
        if len(x_vals) != n:
            raise ValueError(f"Expected {n} X values, got {len(x_vals)}")
        if len(e_vals) != n * n:
            raise ValueError(f"Expected {n * n} E values, got {len(e_vals)}")

        lines.append(f"N={n}")
        lines.append("X:")
        lines.append(" ".join(str(value) for value in x_vals))
        lines.append("E:")
        for row_start in range(0, len(e_vals), n):
            row = e_vals[row_start:row_start + n]
            lines.append(" ".join(str(value) for value in row))

    file_path.write_text("\n".join(lines) + "\n", encoding=encoding)


def graphs_to_dgl(graphs, device):
    """Convert loaded graphs to DGL format with batching."""
    dgl_graphs = []
    for graph_data in graphs:
        n = graph_data['n']
        x = torch.tensor(graph_data['x'], dtype=torch.float32).reshape(-1, 1)  # Node features
        e = graph_data['e']  # Adjacency matrix (flattened n*n)
        
        # Build edge list from adjacency matrix
        edges_src = []
        edges_dst = []
        for i in range(n):
            for j in range(n):
                if e[i * n + j] > 0:  # If there's an edge
                    edges_src.append(i)
                    edges_dst.append(j)
        
        # Create DGL graph
        if len(edges_src) > 0:
            g = dgl.graph((edges_src, edges_dst), num_nodes=n)
        else:
            # Empty graph case
            g = dgl.graph(([], []), num_nodes=n)
        
        dgl_graphs.append(g)
    
    # Batch all graphs together
    batched_graph = dgl.batch(dgl_graphs)
    
    # Get node features for batched graph
    all_node_features = []
    for graph_data in graphs:
        x = torch.tensor(graph_data['x'], dtype=torch.float32).reshape(-1, 1)
        all_node_features.append(x)
    node_features = torch.cat(all_node_features, dim=0).to(device)
    
    return batched_graph.to(device), node_features
