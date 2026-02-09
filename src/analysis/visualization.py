class WeightedVisualization:
    def visualize_chain(self, path, nodes_list, adjacency_matrix):
        import imageio
        import numpy as np
        import os
        # Convertir grafos a networkx
        graphs = [self.to_networkx(nodes_list[i], adjacency_matrix[i]) for i in range(nodes_list.shape[0])]
        # Usar layout fijo para todos los frames
        final_graph = graphs[-1]
        final_pos = nx.spring_layout(final_graph, seed=0)
        save_paths = []
        num_frames = nodes_list.shape[0]
        for frame in range(num_frames):
            file_name = os.path.join(path, f'fram_{frame}.png')
            self.visualize_weighted(graph=graphs[frame], pos=final_pos, path=file_name)
            save_paths.append(file_name)
        imgs = [imageio.imread(fn) for fn in save_paths]
        gif_path = os.path.join(os.path.dirname(path), f'{os.path.basename(path)}.gif')
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True, duration=20)
        if wandb.run:
            wandb.log({'chain': [wandb.Video(gif_path, caption=gif_path, format="gif")]})

    def to_networkx(self, node_list, adjacency_matrix):
        """
        Convertir a networkx y añadir pesos como atributos de arista
        node_list: nodos (bs x n)
        adjacency_matrix: matriz de adyacencia con pesos (bs x n x n)
        """
        graph = nx.Graph()
        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            graph.add_node(i, number=i, symbol=node_list[i], color_val=node_list[i])
        rows, cols = np.where(adjacency_matrix > 0)
        edges = zip(rows.tolist(), cols.tolist())
        for edge in edges:
            weight = adjacency_matrix[edge[0]][edge[1]]
            graph.add_edge(edge[0], edge[1], weight=weight)
        return graph

    def visualize_weighted(self, graph, pos, path, iterations=100, node_size=100, largest_component=False):
        if largest_component:
            CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            graph = CGs[0]
        if pos is None:
            pos = nx.spring_layout(graph, iterations=iterations)
        plt.figure()
        # Dibujar nodos
        nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color='lightblue')
        # Dibujar aristas con grosor proporcional al peso
        weights = [d['weight'] for u, v, d in graph.edges(data=True)]
        nx.draw_networkx_edges(graph, pos, width=[max(1, w/np.max(weights)*4) for w in weights], edge_color='gray')
        # Etiquetas de nodos
        nx.draw_networkx_labels(graph, pos, font_size=10)
        # Etiquetas de pesos
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red', font_size=8)
        plt.tight_layout()
        plt.savefig(path)
        plt.close("all")

    def visualize(self, path: str, graphs: list, num_graphs_to_visualize: int, log='graph'):
        if not os.path.exists(path):
            os.makedirs(path)
        for i in range(num_graphs_to_visualize):
            file_path = os.path.join(path, f'graph_{i}.png')
            graph = self.to_networkx(graphs[i][0].numpy(), graphs[i][1].numpy())
            self.visualize_weighted(graph=graph, pos=None, path=file_path)
            im = plt.imread(file_path)
            if wandb.run and log is not None:
                wandb.log({log: [wandb.Image(im, caption=file_path)]})
import os

#from rdkit import Chem
#from rdkit.Chem import Draw, AllChem
#from rdkit.Geometry import Point3D
#from rdkit import RDLogger
import imageio
import networkx as nx
import numpy as np
#import rdkit.Chem
import wandb
import matplotlib.pyplot as plt


class NonMolecularVisualization:
    def to_networkx(self, node_list, adjacency_matrix):
        """
        Convert graphs to networkx graphs
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        graph = nx.Graph()

        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            graph.add_node(i, number=i, symbol=node_list[i], color_val=node_list[i])

        rows, cols = np.where(adjacency_matrix >= 1)
        edges = zip(rows.tolist(), cols.tolist())
        for edge in edges:
            edge_type = adjacency_matrix[edge[0]][edge[1]]
            graph.add_edge(edge[0], edge[1], color=float(edge_type), weight=3 * edge_type)

        return graph

    def visualize_non_molecule(self, graph, pos, path, iterations=100, node_size=100, largest_component=False):
        if largest_component:
            CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            graph = CGs[0]

        # Plot the graph structure with colors
        if pos is None:
            pos = nx.spring_layout(graph, iterations=iterations)

        # Set node colors based on the eigenvectors
        w, U = np.linalg.eigh(nx.normalized_laplacian_matrix(graph).toarray())
        vmin, vmax = np.min(U[:, 1]), np.max(U[:, 1])
        m = max(np.abs(vmin), vmax)
        vmin, vmax = -m, m

        plt.figure()
        nx.draw(graph, pos, font_size=10, node_size=node_size, with_labels=True, node_color=U[:, 1],
                cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax, edge_color='grey')

        plt.tight_layout()
        plt.savefig(path)
        plt.close("all")

    def visualize(self, path: str, graphs: list, num_graphs_to_visualize: int, log='graph'):
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final molecules
        for i in range(num_graphs_to_visualize):
            file_path = os.path.join(path, 'graph_{}.png'.format(i))
            graph = self.to_networkx(graphs[i][0].numpy(), graphs[i][1].numpy())
            self.visualize_non_molecule(graph=graph, pos=None, path=file_path)
            im = plt.imread(file_path)
            if wandb.run and log is not None:
                wandb.log({log: [wandb.Image(im, caption=file_path)]})

    def visualize_chain(self, path, nodes_list, adjacency_matrix):
        # convert graphs to networkx
        graphs = [self.to_networkx(nodes_list[i], adjacency_matrix[i]) for i in range(nodes_list.shape[0])]
        # find the coordinates of atoms in the final molecule
        final_graph = graphs[-1]
        #ori_graph = graphs[0]
        final_pos = nx.spring_layout(final_graph, seed=0)
        #final_pos = nx.spring_layout(ori_graph, seed=0)

        # draw gif
        save_paths = []
        num_frams = nodes_list.shape[0]

        for frame in range(num_frams):
            file_name = os.path.join(path, 'fram_{}.png'.format(frame))
            self.visualize_non_molecule(graph=graphs[frame], pos=final_pos, path=file_name)
            save_paths.append(file_name)

        imgs = [imageio.imread(fn) for fn in save_paths]
        gif_path = os.path.join(os.path.dirname(path), '{}.gif'.format(path.split('/')[-1]))
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True, duration=20)
        if wandb.run:
            wandb.log({'chain': [wandb.Video(gif_path, caption=gif_path, format="gif")]})
