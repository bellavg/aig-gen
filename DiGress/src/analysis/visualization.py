import os
import warnings
import imageio
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Union
import wandb  # For logging to Weights & Biases

# Assuming your AIG configuration and custom conversion function are accessible
from src.aig_config import NODE_TYPE_KEYS, EDGE_TYPE_KEYS, \
    NUM_NODE_FEATURES, NUM_EDGE_FEATURES

# Your preferred AIG visualization function (from your old project)
# For clarity, I'm embedding its core logic here, but you could also import it
# if it's in a separate, importable file.

# --- Visualization Constants (from your script) ---
VALID_AIG_NODE_TYPES = set(NODE_TYPE_KEYS)
VALID_AIG_EDGE_TYPES = set(EDGE_TYPE_KEYS)
NODE_TYPE_COLOR_MAP = {
    "NODE_CONST0": 'gold',
    "NODE_PI": 'palegreen',
    "NODE_AND": 'lightskyblue',
    "NODE_PO": 'lightcoral',
    "UNKNOWN": 'lightgrey',
}
DEFAULT_NODE_COLOR = 'white'  # Should not be used if types are correct


def visualize_aig_structure_matplotlib(G: nx.DiGraph, output_file='generated_aig_structure.png', title_prefix=""):
    """ Visualize the AIG structure using Matplotlib and NetworkX. """
    if G is None or not isinstance(G, nx.DiGraph) or G.number_of_nodes() == 0:
        warnings.warn(f"Skipping visualization for empty/invalid graph: {output_file}")
        return False

    plt.figure(figsize=(16, 14))
    pos = None
    layout_engine = "spring"

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        layout_engine = "dot"
    except Exception:  # Catches ImportError or other graphviz errors
        warnings.warn("Graphviz layout failed or not available. Using spring_layout.")
        pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

    node_colors = []
    node_labels = {}
    for node in sorted(G.nodes()):
        node_data = G.nodes[node]
        node_type = node_data.get('type', 'UNKNOWN')
        if node_type not in VALID_AIG_NODE_TYPES:
            node_type = "UNKNOWN"
        node_colors.append(NODE_TYPE_COLOR_MAP.get(node_type, DEFAULT_NODE_COLOR))
        node_labels[node] = f"{node}\n({node_type})"

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, alpha=0.9)

    regular_edges = [(u, v) for u, v, data in G.edges(data=True) if data.get('type') == "EDGE_REG"]
    inverted_edges = [(u, v) for u, v, data in G.edges(data=True) if data.get('type') == "EDGE_INV"]
    unknown_edges = [(u, v) for u, v, data in G.edges(data=True)
                     if data.get('type') not in VALID_AIG_EDGE_TYPES and data.get('type') is not None]

    nx.draw_networkx_edges(G, pos, edgelist=regular_edges, width=1.5, edge_color='black', style='solid',
                           arrows=True, arrowsize=12, node_size=700, connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_edges(G, pos, edgelist=inverted_edges, width=1.5, edge_color='red', style='dashed',
                           arrows=True, arrowsize=12, node_size=700, connectionstyle='arc3,rad=0.1')
    if unknown_edges:
        nx.draw_networkx_edges(G, pos, edgelist=unknown_edges, width=1.0, edge_color='grey', style='dotted',
                               arrows=True, arrowsize=10, node_size=700, connectionstyle='arc3,rad=0.1')

    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight='bold')

    legend_elements = [
        plt.Line2D([0], [0], color='black', lw=1.5, linestyle='solid', label='EDGE_REG'),
        plt.Line2D([0], [0], color='red', lw=1.5, linestyle='dashed', label='EDGE_INV')
    ]
    if unknown_edges:  # Only add unknown edge to legend if present
        legend_elements.append(plt.Line2D([0], [0], color='grey', lw=1.0, linestyle='dotted', label='Unknown Edge'))

    for node_type_key, color in NODE_TYPE_COLOR_MAP.items():
        legend_elements.append(plt.scatter([], [], s=80, color=color, label=node_type_key))

    plt.legend(handles=legend_elements, loc='upper right', fontsize='small',
               bbox_to_anchor=(1.15, 1.05), frameon=True, facecolor='white', framealpha=0.8)

    base_filename = os.path.basename(output_file)
    plt.title(f'{title_prefix}{base_filename} ({layout_engine} layout)', fontsize=14)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        # print(f"Custom AIG visualization saved to {output_file}")
        return True
    except Exception as e:
        warnings.warn(f"Error saving custom AIG visualization {output_file}: {e}")
        return False
    finally:
        plt.close()


class AIGVisualization:
    def __init__(self, cfg=None, dataset_infos=None):
        """
        Custom visualization tool for AIGs, integrating user's preferred style.
        Args:
            cfg: Hydra configuration object (optional).
            dataset_infos: Dataset information object (optional).
        """
        self.cfg = cfg
        self.dataset_infos = dataset_infos
        # NUM_NODE_FEATURES and NUM_EDGE_FEATURES are imported directly from aig_config

    def _convert_sampled_to_nx(self, node_class_indices: np.ndarray, edge_class_indices_dense: np.ndarray) -> Union[
        nx.DiGraph, None]:
        """
        Converts sampled class indices (nodes and dense N x N edges) to a NetworkX DiGraph
        with 'type' attributes for nodes and edges.
        """
        num_nodes = node_class_indices.shape[0]
        nx_graph = nx.DiGraph()

        # Process nodes
        for i in range(num_nodes):
            class_idx = node_class_indices[i]
            if not (0 <= class_idx < NUM_NODE_FEATURES):
                warnings.warn(
                    f"Node {i} has invalid class index {class_idx}. Max is {NUM_NODE_FEATURES - 1}. Assigning UNKNOWN.")
                node_type_str = "UNKNOWN"
            else:
                node_type_str = NODE_TYPE_KEYS[class_idx]
            nx_graph.add_node(i, type=node_type_str)

        # Process edges from the dense N x N class index matrix
        # edge_class_indices_dense has shape (N, N), values are class indices for edges
        # where index 0 means "no specific AIG type / no edge"
        # and indices 1 to NUM_EDGE_FEATURES map to actual AIG edge types.
        for u in range(num_nodes):
            for v in range(num_nodes):
                if u == v: continue  # No self-loops

                edge_class_idx_shifted = edge_class_indices_dense[u, v]  # This is the "shifted_type_index"

                if edge_class_idx_shifted == 0:  # Channel 0: No specific AIG type or no edge
                    continue

                actual_aig_type_index = edge_class_idx_shifted - 1  # Convert back to 0-based index for EDGE_TYPE_KEYS

                if not (0 <= actual_aig_type_index < NUM_EDGE_FEATURES):
                    warnings.warn(
                        f"Edge ({u}->{v}) has invalid actual AIG type index {actual_aig_type_index} (from shifted index {edge_class_idx_shifted}). Max is {NUM_EDGE_FEATURES - 1}. Skipping edge."
                    )
                    edge_type_str = "UNKNOWN_EDGE_TYPE"  # Should not happen if model output is correct
                else:
                    edge_type_str = EDGE_TYPE_KEYS[actual_aig_type_index]

                nx_graph.add_edge(u, v, type=edge_type_str)

        return nx_graph

    def visualize(self, path: str, graphs_data: list, num_graphs_to_visualize: int, log='graph'):
        """
        Visualizes individual sampled AIGs.
        Args:
            path (str): Directory to save the visualizations.
            graphs_data (list): List of sampled graphs. Each item is a list/tuple:
                                [node_class_indices_tensor, edge_class_indices_dense_tensor]
                                - node_class_indices_tensor: (num_nodes,) tensor of node type indices.
                                - edge_class_indices_dense_tensor: (num_nodes, num_nodes) tensor of edge type indices.
            num_graphs_to_visualize (int): Number of graphs to visualize from the list.
            log (str, optional): Wandb key for logging. Defaults to 'graph'.
        """
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        num_to_draw = min(num_graphs_to_visualize, len(graphs_data))
        if num_to_draw == 0:
            print("No graphs provided or requested for visualization.")
            return

        print(f"Custom AIG Visualization: Visualizing {num_to_draw} of {len(graphs_data)} graphs...")

        for i in range(num_to_draw):
            node_indices_np = graphs_data[i][0].cpu().numpy()  # Should be (num_nodes,)
            edge_indices_dense_np = graphs_data[i][1].cpu().numpy()  # Should be (num_nodes, num_nodes)

            if node_indices_np.ndim == 0:  # Handle case where graph might have 0 nodes (e.g. n_nodes was 0)
                warnings.warn(f"Graph {i} seems to have 0 nodes based on node_indices_np. Skipping visualization.")
                continue
            if node_indices_np.shape[0] == 0:  # num_nodes is 0
                warnings.warn(f"Graph {i} has 0 nodes. Skipping visualization.")
                continue

            nx_graph = self._convert_sampled_to_nx(node_indices_np, edge_indices_dense_np)

            if nx_graph:
                file_path = os.path.join(path, f'aig_custom_{i:03d}.png')
                title = f"Sampled AIG {i} - "
                success = visualize_aig_structure_matplotlib(nx_graph, output_file=file_path, title_prefix=title)

                if success and wandb.run and log is not None:
                    try:
                        wandb.log({f"{log}/aig_custom_{i:03d}": wandb.Image(file_path)},
                                  commit=False)  # commit=False if in a loop
                    except Exception as e:
                        warnings.warn(f"Failed to log image {file_path} to wandb: {e}")
            else:
                warnings.warn(f"Could not convert graph {i} to NetworkX format for visualization.")
        if wandb.run and log is not None and num_to_draw > 0:
            wandb.log({}, commit=True)  # Final commit for the batch of images

    def visualize_chain(self, path: str, chain_nodes_tensor: torch.Tensor, chain_E_tensor: torch.Tensor):
        """
        Visualizes the generation process (chain) for AIGs as a GIF.
        Args:
            path (str): Base path for saving intermediate frames and the final GIF.
                        The GIF will be saved in the parent directory of `path`.
            chain_nodes_tensor (torch.Tensor): Trajectory of node class indices.
                                               Shape: (num_steps, num_chains_saved, max_nodes)
            chain_E_tensor (torch.Tensor): Trajectory of edge class indices (dense N x N).
                                           Shape: (num_steps, num_chains_saved, max_nodes, max_nodes)
        """
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        num_steps, num_chains_saved, max_nodes = chain_nodes_tensor.shape

        if num_chains_saved == 0 or num_steps == 0:
            print("No chains or steps to visualize.")
            return

        print(f"Custom AIG Chain Visualization: Visualizing {num_chains_saved} chain(s) with {num_steps} steps each.")

        for chain_idx in range(min(num_chains_saved, 1)):  # Visualize only the first chain for simplicity
            frame_paths = []
            chain_specific_path = os.path.join(path, f"chain_{chain_idx}")
            os.makedirs(chain_specific_path, exist_ok=True)

            # Heuristic to find actual number of nodes for this chain (can be tricky if n_nodes varies per sample in original batch)
            # For chains, n_max is used. We need to find where padding starts if possible, or assume all are used.
            # A simple way: check the last frame for active nodes (non -1 if that's the padding convention)
            # However, DiGress sample_batch gives tensors sliced up to n_max, not original n_nodes for each item in chain.
            # We'll assume n_max is the number of nodes to visualize for the chain.
            # A more robust way might involve passing n_nodes for each chain item if available.

            # The input tensors chain_nodes_tensor and chain_E_tensor from DiGress's sample_batch
            # are already class indices (not one-hot).
            # X.size(1) is n_max. E.size(1) is n_max.
            # The tensors are (number_chain_steps, keep_chain, n_max) for nodes
            # and (number_chain_steps, keep_chain, n_max, n_max) for edges.

            for frame_idx in range(num_steps):
                node_indices_np = chain_nodes_tensor[frame_idx, chain_idx].cpu().numpy()  # (n_max,)
                edge_indices_dense_np = chain_E_tensor[frame_idx, chain_idx].cpu().numpy()  # (n_max, n_max)

                # Determine actual number of nodes for this frame (e.g. by finding last non-padding node)
                # If padding value is -1 for nodes (common in some DiGress processing steps before one-hot)
                # active_node_mask = node_indices_np != -1 # Or some other padding indicator
                # if not np.any(active_node_mask): # All padding
                #     num_actual_nodes_in_frame = 0
                # else:
                #     num_actual_nodes_in_frame = np.where(active_node_mask)[0][-1] + 1
                # For now, assume n_max is used, or that node_indices are already for valid nodes up to n_max
                num_actual_nodes_in_frame = node_indices_np.shape[0]  # This is n_max

                if num_actual_nodes_in_frame == 0:
                    # print(f"Frame {frame_idx} of chain {chain_idx} has 0 active nodes. Skipping frame.")
                    continue

                # Slice to actual nodes if a more precise count was determined
                # node_indices_np_sliced = node_indices_np[:num_actual_nodes_in_frame]
                # edge_indices_dense_np_sliced = edge_indices_dense_np[:num_actual_nodes_in_frame, :num_actual_nodes_in_frame]

                nx_graph = self._convert_sampled_to_nx(node_indices_np, edge_indices_dense_np)

                if nx_graph:
                    frame_file = os.path.join(chain_specific_path, f'frame_{frame_idx:03d}.png')
                    title = f"Chain {chain_idx} Frame {frame_idx} - "
                    success = visualize_aig_structure_matplotlib(nx_graph, output_file=frame_file, title_prefix=title)
                    if success:
                        frame_paths.append(frame_file)
                else:
                    warnings.warn(f"Could not convert frame {frame_idx} of chain {chain_idx} to NetworkX.")

            if frame_paths:
                gif_filename = f'{os.path.basename(path)}_chain_{chain_idx}.gif'
                gif_path = os.path.join(os.path.dirname(path), gif_filename)  # Save GIF one level up

                imgs = [imageio.v2.imread(fn) for fn in frame_paths]
                # Add duplicates of the last frame to make it pause
                if imgs:  # Ensure imgs is not empty
                    for _ in range(10):  # Number of times to repeat the last frame
                        imgs.append(imgs[-1])

                try:
                    imageio.mimsave(gif_path, imgs, duration=0.2, loop=0)  # loop=0 means infinite loop
                    print(f"Saved chain GIF to {gif_path}")
                    if wandb.run:
                        try:
                            wandb.log({f"chain_visuals/chain_{chain_idx}": wandb.Video(gif_path, fps=5, format="gif")},
                                      commit=True)
                        except Exception as e:
                            warnings.warn(f"Failed to log GIF {gif_path} to wandb: {e}")
                except Exception as e:
                    warnings.warn(f"Failed to save GIF {gif_path}: {e}")
            else:
                print(f"No frames generated for GIF for chain {chain_idx}.")


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
        nx.draw(graph, pos, font_size=5, node_size=node_size, with_labels=False, node_color=U[:, 1],
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
        final_pos = nx.spring_layout(final_graph, seed=0)

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
