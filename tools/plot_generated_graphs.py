import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

def parse_generated_samples(file_path, num_graphs=3):
    """Parse generated_samples.txt and extract graphs"""
    graphs = []
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split by graph entries
    entries = content.strip().split('\n\n')
    
    for entry in entries[:num_graphs]:
        lines = entry.strip().split('\n')
        if not lines or not lines[0].startswith('N='):
            continue
        
        # Parse number of nodes
        n_nodes = int(lines[0].split('=')[1])
        
        # Parse node features (X)
        x_idx = next(i for i, l in enumerate(lines) if 'X:' in l)
        x_line = lines[x_idx + 1]
        x_features = list(map(int, x_line.split()))
        
        # Parse edges (E)
        e_idx = next(i for i, l in enumerate(lines) if 'E:' in l)
        e_lines = lines[e_idx + 1:]
        
        # Reconstruct adjacency matrix from edge list rows
        edges = []
        for i, e_line in enumerate(e_lines):
            if not e_line.strip():
                break
            row = list(map(int, e_line.split()))
            for j, val in enumerate(row):
                if val == 1 and i < j:  # Avoid duplicates for undirected graph
                    edges.append((i, j))
        
        graphs.append({
            'n_nodes': n_nodes,
            'edges': edges,
            'features': x_features[:n_nodes]
        })
    
    return graphs

def plot_graphs(graphs, output_dir=None):
    """Plot each graph individually and save them"""
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"💾 Saving plots to: {output_dir}")
    
    for idx, graph_data in enumerate(graphs):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Create NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(graph_data['n_nodes']))
        G.add_edges_from(graph_data['edges'])
        
        # Draw graph
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=100, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              width=0.5, ax=ax)
        
        ax.set_title(f'Generated Graph {idx+1}\n({graph_data["n_nodes"]} nodes, {len(graph_data["edges"])} edges)', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        if output_dir:
            file_path = output_dir / f'graph_{idx}.png'
            plt.savefig(file_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved: graph_{idx}.png")
        else:
            plt.show()
        
        plt.close(fig)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot generated graphs from generated_samples.txt')
    parser.add_argument('--file', type=str, 
                       default='outputs/2026-02-07/12-51-53-dimacs_exp1/generated_samples1.txt',
                       help='Path to generated_samples.txt')
    parser.add_argument('--num', type=int, default=None,
                       help='Number of graphs to plot (default: all)')
    
    args = parser.parse_args()
    
    print(f"📊 Parsing generated graphs from: {args.file}")
    
    # Parse all graphs (or limited number)
    all_graphs = parse_generated_samples(args.file, num_graphs=10000)
    graphs = all_graphs[:args.num] if args.num else all_graphs
    
    if graphs:
        print(f"✓ Successfully parsed {len(graphs)} graphs")
        
        # Create output folder next to the txt file
        txt_path = Path(args.file)
        output_dir = txt_path.parent / 'plots'
        
        plot_graphs(graphs, output_dir)
        print(f"\n✅ All {len(graphs)} graphs saved in: {output_dir}")
    else:
        print("❌ No graphs found in file")
