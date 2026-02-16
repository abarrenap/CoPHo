#!/usr/bin/env python3
"""
Script to parse DIMACS .col files to txt format with the structure:

N=<num_nodes>
X:
<node_features>
E:
<adjacency_matrix>

(separate graphs with blank lines)
"""

import os
import struct
import numpy as np
from pathlib import Path


def parse_dimacs_text_file(filepath):
    """
    Parse a DIMACS .col text file and return nodes and edges
    
    Returns:
        tuple: (num_nodes, edge_list)
    """
    edges = []
    num_nodes = 0
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            if line.startswith('p'):
                # p edge n m
                parts = line.split()
                num_nodes = int(parts[2])
                continue
            if line.startswith('e'):
                # e u v (1-indexed in DIMACS)
                parts = line.split()
                u, v = int(parts[1]) - 1, int(parts[2]) - 1  # Convert to 0-indexed
                edges.append((u, v))
    
    return num_nodes, edges


def parse_dimacs_binary_file(filepath):
    """
    Parse a DIMACS binary .col.b file format
    Binary format structure:
    - int (4 bytes): num_nodes
    - int (4 bytes): num_edges
    - For each edge: two ints (4 bytes each) for u, v (1-indexed)
    
    Returns:
        tuple: (num_nodes, edge_list)
    """
    edges = []
    
    try:
        with open(filepath, 'rb') as f:
            # Read number of nodes and edges
            header = f.read(8)
            if len(header) < 8:
                raise ValueError("File too small")
            
            num_nodes = struct.unpack('>I', header[0:4])[0]
            num_edges = struct.unpack('>I', header[4:8])[0]
            
            # Read edges
            for _ in range(num_edges):
                edge_bytes = f.read(8)
                if len(edge_bytes) < 8:
                    break
                u = struct.unpack('>I', edge_bytes[0:4])[0] - 1  # Convert to 0-indexed
                v = struct.unpack('>I', edge_bytes[4:8])[0] - 1
                edges.append((u, v))
    except Exception as e:
        # Fallback: try little-endian format
        try:
            with open(filepath, 'rb') as f:
                header = f.read(8)
                if len(header) < 8:
                    raise ValueError("File too small")
                
                num_nodes = struct.unpack('<I', header[0:4])[0]
                num_edges = struct.unpack('<I', header[4:8])[0]
                
                for _ in range(num_edges):
                    edge_bytes = f.read(8)
                    if len(edge_bytes) < 8:
                        break
                    u = struct.unpack('<I', edge_bytes[0:4])[0] - 1
                    v = struct.unpack('<I', edge_bytes[4:8])[0] - 1
                    edges.append((u, v))
        except Exception as e2:
            raise ValueError(f"Could not parse binary file: {e2}")
    
    # Validate nodes
    if edges:
        max_node = max(max(u, v) for u, v in edges)
        if max_node >= num_nodes:
            num_nodes = max_node + 1
    
    return num_nodes, edges


def create_adjacency_list(num_nodes, edges):
    """
    Create adjacency list (more memory efficient than full matrix)
    Returns a dict where adj_list[i] is a set of neighbors for node i
    """
    adj_list = {i: set() for i in range(num_nodes)}
    for u, v in edges:
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj_list[u].add(v)
            adj_list[v].add(u)  # Make it undirected
    return adj_list


def parse_dimacs_file(filepath):
    """
    Parse a DIMACS .col file (text or binary) and return nodes and edges
    
    Returns:
        tuple: (num_nodes, edge_list)
    """
    if filepath.endswith('.col.b'):
        return parse_dimacs_binary_file(filepath)
    else:
        return parse_dimacs_text_file(filepath)


def create_txt_format_efficient(num_nodes, edges):
    """
    Create the txt format representation of a graph using edge list
    This is memory-efficient for large graphs
    
    Returns:
        str: Formatted string with N, X, and E sections
    """
    output = []
    
    # N: number of nodes
    output.append(f"N={num_nodes}")
    
    # X: node features (all ones since DIMACS doesn't have explicit features)
    output.append("X:")
    node_features = [1] * num_nodes  # All nodes have feature value 1
    # Write all node features in a single line
    output.append(" ".join(str(feat) for feat in node_features))
    
    # E: adjacency matrix (built from edge list)
    output.append("E:")
    
    # Create adjacency list from edges
    adj_list = {i: set() for i in range(num_nodes)}
    for u, v in edges:
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj_list[u].add(v)
            adj_list[v].add(u)  # Make it undirected
    
    # Write adjacency matrix row by row
    for i in range(num_nodes):
        row = [1 if j in adj_list[i] else 0 for j in range(num_nodes)]
        output.append(" ".join(str(val) for val in row))
    
    return "\n".join(output)


def create_txt_format(num_nodes, adj_matrix):
    """
    Create the txt format representation of a graph
    
    Returns:
        str: Formatted string with N, X, and E sections
    """
    output = []
    
    # N: number of nodes
    output.append(f"N={num_nodes}")
    
    # X: node features (all ones since DIMACS doesn't have explicit features)
    output.append("X:")
    node_features = [1] * num_nodes  # All nodes have feature value 1
    # Write node features in rows (10 per row for readability)
    features_str = ""
    for i, feat in enumerate(node_features):
        features_str += str(feat)
        if (i + 1) % 10 == 0 or i == len(node_features) - 1:
            if features_str.strip():
                output.append(features_str.strip())
            features_str = ""
        else:
            features_str += " "
    if features_str.strip():
        output.append(features_str.strip())
    
    # E: adjacency matrix
    output.append("E:")
    for row in adj_matrix:
        output.append(" ".join(str(val) for val in row))
    
    return "\n".join(output)


def process_dimacs_dataset(input_dir, output_file):
    """
    Process all .col files (text format only) in the input directory and save to a single txt file
    
    Args:
        input_dir: Path to directory containing .col files
        output_file: Path to output txt file
    """
    col_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.col') and not f.endswith('.col.b')])
    
    if not col_files:
        print(f"❌ No .col files found in {input_dir}")
        return
    
    print(f"📊 Found {len(col_files)} DIMACS files")
    
    all_graphs = []
    failed_files = []
    
    for i, filename in enumerate(col_files, 1):
        filepath = os.path.join(input_dir, filename)
        try:
            print(f"  [{i}/{len(col_files)}] Processing {filename}...", end=" ")
            num_nodes, edges = parse_dimacs_file(filepath)
            graph_txt = create_txt_format_efficient(num_nodes, edges)
            all_graphs.append(graph_txt)
            print(f"✓ ({num_nodes} nodes)")
        except Exception as e:
            print(f"❌ Error: {e}")
            failed_files.append(filename)
    
    # Write to output file
    print(f"\n💾 Writing to {output_file}...")
    with open(output_file, 'w') as f:
        f.write("\n\n".join(all_graphs))
    
    print(f"\n✅ Complete!")
    print(f"   Graphs processed: {len(all_graphs)}")
    print(f"   Failed files: {len(failed_files)}")
    
    if failed_files:
        print(f"\n⚠️  Failed files:")
        for filename in failed_files:
            print(f"   - {filename}")
    
    print(f"\n📁 Output file: {output_file}")
    print(f"   Size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")


if __name__ == "__main__":
    # Default paths
    dimacs_dir = "/Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/data/DIMACS"
    output_file = "/Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/input_data/DIMACS.txt"
    
    # Check if input directory exists
    if not os.path.isdir(dimacs_dir):
        print(f"❌ Input directory not found: {dimacs_dir}")
        exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Process the dataset
    process_dimacs_dataset(dimacs_dir, output_file)
