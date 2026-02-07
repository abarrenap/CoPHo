#!/usr/bin/env python3
"""
Script para visualizar grafos generados o del dataset
"""
import sys
import os
sys.path.insert(0, '/Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo')

import networkx as nx
import matplotlib.pyplot as plt
from src.datasets.dimacs_dataset import DIMACSDataset
import argparse

def visualize_dimacs_dataset(num_graphs=5):
    """Visualiza algunos grafos del dataset DIMACS"""
    print(f"📊 Visualizando {num_graphs} grafos del dataset DIMACS...")
    
    dataset = DIMACSDataset('/Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/data/DIMACS', split='train')
    
    num_to_show = min(num_graphs, len(dataset))
    
    fig, axes = plt.subplots(1, num_to_show, figsize=(5*num_to_show, 5))
    if num_to_show == 1:
        axes = [axes]
    
    for idx in range(num_to_show):
        data = dataset[idx]
        
        # Convertir a NetworkX
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        
        edge_index = data.edge_index.numpy()
        edges = [(edge_index[0, i], edge_index[1, i]) for i in range(edge_index.shape[1])]
        G.add_edges_from(edges)
        
        # Dibujar
        ax = axes[idx]
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, ax=ax, node_color='lightblue', 
                node_size=100, with_labels=False, edge_color='gray',
                width=0.5)
        ax.set_title(f'Grafo {idx+1}\n{data.num_nodes} nodos, {data.num_edges//2} aristas')
        ax.axis('off')
    
    plt.tight_layout()
    output_path = '/Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/dimacs_samples.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualización guardada en: {output_path}")
    plt.show()

def visualize_generated_graphs(graphs_dir):
    """Visualiza grafos generados por el modelo"""
    print(f"📊 Visualizando grafos generados de: {graphs_dir}")
    
    # Buscar archivos .npz o .pt
    import glob
    graph_files = glob.glob(os.path.join(graphs_dir, "*.npz")) + \
                  glob.glob(os.path.join(graphs_dir, "*.pt"))
    
    if not graph_files:
        print(f"❌ No se encontraron grafos en {graphs_dir}")
        return
    
    print(f"✓ Encontrados {len(graph_files)} archivos")
    # Aquí se puede implementar la lectura y visualización según el formato usado

def show_graph_statistics(dataset_path):
    """Muestra estadísticas del dataset"""
    dataset = DIMACSDataset(dataset_path, split='train')
    
    print("\n📊 Estadísticas del Dataset DIMACS")
    print("=" * 50)
    print(f"Número de grafos: {len(dataset)}")
    
    node_counts = [data.num_nodes for data in dataset]
    edge_counts = [data.num_edges // 2 for data in dataset]
    
    print(f"\nNodos:")
    print(f"  - Mínimo: {min(node_counts)}")
    print(f"  - Máximo: {max(node_counts)}")
    print(f"  - Promedio: {sum(node_counts)/len(node_counts):.1f}")
    
    print(f"\nAristas:")
    print(f"  - Mínimo: {min(edge_counts)}")
    print(f"  - Máximo: {max(edge_counts)}")
    print(f"  - Promedio: {sum(edge_counts)/len(edge_counts):.1f}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualizar grafos DIMACS')
    parser.add_argument('--mode', choices=['dataset', 'generated', 'stats'], 
                        default='dataset',
                        help='Qué visualizar: dataset, generated, o stats')
    parser.add_argument('--num', type=int, default=5,
                        help='Número de grafos a visualizar')
    parser.add_argument('--path', type=str, 
                        default='/Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/data/DIMACS',
                        help='Ruta al dataset o grafos generados')
    
    args = parser.parse_args()
    
    if args.mode == 'dataset':
        visualize_dimacs_dataset(args.num)
    elif args.mode == 'generated':
        visualize_generated_graphs(args.path)
    elif args.mode == 'stats':
        show_graph_statistics(args.path)
