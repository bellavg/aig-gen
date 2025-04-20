# visualize_g2pt_aigs.py
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import os
import logging
from typing import Dict, Any # Added for type hinting

# --- Constants ---
# Define node/edge types and colors (matching your previous script)
# These are used for coloring/styling based on the 'type' attribute from the graph
VALID_AIG_NODE_TYPES = {'NODE_CONST0', 'NODE_PI', 'NODE_AND', 'NODE_PO'}
VALID_AIG_EDGE_TYPES = {'EDGE_INV', 'EDGE_REG'}
EDGE_TYPE_MAP = {"EDGE_REG": 1, "EDGE_INV": 2, "UNKNOWN": 0} # Map names back to numbers if needed by logic
NODE_TYPE_COLOR_MAP = {
    "NODE_CONST0": 'gold', # Special color for const0
    "NODE_PI": 'palegreen',
    "NODE_AND": 'lightskyblue',
    "NODE_PO": 'lightcoral',
    "UNKNOWN": 'lightgrey', # Fallback if type attribute is missing or invalid
}
DEFAULT_NODE_COLOR = 'white'

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("visualize_g2pt_aigs")

# --- Visualization Function (Adapted from your evaluate_aigs.py) ---
def visualize_aig_structure(G: nx.DiGraph, output_file='generated_aig_structure.png'):
    """ Visualize the generated AIG structure, using assigned types and edge types. """
    if G is None or not isinstance(G, nx.DiGraph) or G.number_of_nodes() == 0:
        logger.info(f"Skipping visualization for empty/invalid graph: {output_file}")
        return

    plt.figure(figsize=(16, 14)) # Keep figure size
    pos = None
    layout_engine = "spring" # Default layout

    # Try graphviz layout first
    try:
        # Requires pygraphviz or pydot and graphviz installed
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        logger.debug("Using graphviz 'dot' layout for visualization.")
        layout_engine = "dot"
    except ImportError:
        logger.warning("pygraphviz/pydot not found. Using spring_layout (less structured).")
        pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    except Exception as e: # Catch other layout errors
        logger.warning(f"Graphviz layout failed ('{e}'). Using spring_layout.")
        pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

    # --- Node colors and labels based on ASSIGNED types ---
    node_colors = []
    node_labels = {}

    for node in sorted(G.nodes()): # Sort nodes for consistency
        node_data = G.nodes[node]
        node_type = node_data.get('type', 'UNKNOWN') # Get type from node attribute

        # Ensure type is recognized, otherwise use UNKNOWN color
        if node_type not in VALID_AIG_NODE_TYPES:
            node_type = "UNKNOWN"
            logger.warning(f"Node {node} has unrecognized type '{node_data.get('type')}' in graph for {output_file}. Using UNKNOWN color.")

        # Get color from map
        node_colors.append(NODE_TYPE_COLOR_MAP.get(node_type, DEFAULT_NODE_COLOR))
        # Create label including node index and type
        node_labels[node] = f"{node}\n({node_type})"

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, alpha=0.9)

    # --- Separate edges by ASSIGNED type ---
    regular_edges = []
    inverted_edges = []
    unknown_edges = []
    for u, v, data in G.edges(data=True):
        edge_type = data.get('type', 'UNKNOWN_EDGE_TYPE') # Get type from edge attribute
        if edge_type == "EDGE_INV":
            inverted_edges.append((u, v))
        elif edge_type == "EDGE_REG":
            regular_edges.append((u, v))
        else:
             unknown_edges.append((u,v))
             logger.warning(f"Edge ({u}->{v}) has unrecognized type '{edge_type}' in graph for {output_file}.")


    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=regular_edges,
                           width=1.5, edge_color='black', style='solid',
                           arrows=True, arrowsize=12, node_size=700,
                           connectionstyle='arc3,rad=0.1') # Slight curve
    nx.draw_networkx_edges(G, pos, edgelist=inverted_edges,
                           width=1.5, edge_color='red', style='dashed',
                           arrows=True, arrowsize=12, node_size=700,
                           connectionstyle='arc3,rad=0.1')
    # Draw unknown edges differently
    if unknown_edges:
         nx.draw_networkx_edges(G, pos, edgelist=unknown_edges,
                                width=1.0, edge_color='grey', style='dotted',
                                arrows=True, arrowsize=10, node_size=700,
                                connectionstyle='arc3,rad=0.1')

    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight='bold')

    # Create legend (similar to your previous one)
    legend_elements = [
        plt.Line2D([0], [0], color='black', lw=1.5, linestyle='solid', label='EDGE_REG'),
        plt.Line2D([0], [0], color='red', lw=1.5, linestyle='dashed', label='EDGE_INV'),
        plt.Line2D([0], [0], color='grey', lw=1.0, linestyle='dotted', label='Unknown Edge Type'),
        plt.scatter([], [], s=80, color=NODE_TYPE_COLOR_MAP["NODE_CONST0"], label='NODE_CONST0'),
        plt.scatter([], [], s=80, color=NODE_TYPE_COLOR_MAP["NODE_PI"], label='NODE_PI'),
        plt.scatter([], [], s=80, color=NODE_TYPE_COLOR_MAP["NODE_AND"], label='NODE_AND'),
        plt.scatter([], [], s=80, color=NODE_TYPE_COLOR_MAP["NODE_PO"], label='NODE_PO'),
        plt.scatter([], [], s=80, color=NODE_TYPE_COLOR_MAP["UNKNOWN"], label='Unknown Node Type')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize='small',
               bbox_to_anchor=(1.15, 1.05), frameon=True, facecolor='white', framealpha=0.8) # Adjust anchor slightly


    plt.title(f'Generated AIG Structure ({layout_engine} layout) - {os.path.basename(output_file)}', fontsize=14)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout rect for legend space
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving visualization {output_file}: {e}")
    finally:
        plt.close() # Close the figure to free memory


# --- Main Execution Logic ---
def main(args):
    logger.info(f"Loading generated AIGs from: {args.input_pickle_file}")
    try:
        with open(args.input_pickle_file, 'rb') as f:
            generated_graphs = pickle.load(f)
        if not isinstance(generated_graphs, list):
            logger.error("Pickle file does not contain a list of graphs.")
            return
        logger.info(f"Loaded {len(generated_graphs)} graphs.")
    except FileNotFoundError:
        logger.error(f"Input pickle file not found: {args.input_pickle_file}")
        return
    except Exception as e:
        logger.error(f"Error loading pickle file: {e}")
        return

    if not generated_graphs:
        logger.warning("No graphs found in the pickle file. Nothing to visualize.")
        return

    # Determine how many graphs to visualize
    num_to_visualize = min(args.num_visualize, len(generated_graphs))
    if num_to_visualize <= 0:
        logger.info("Number of graphs to visualize is zero. Exiting.")
        return

    logger.info(f"Preparing to visualize the first {num_to_visualize} graphs...")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    visualized_count = 0
    for i, graph in enumerate(generated_graphs[:num_to_visualize]):
        output_filename = f"aig_graph_{i:04d}.png"
        output_path = os.path.join(args.output_dir, output_filename)
        logger.debug(f"Visualizing graph {i} -> {output_path}")
        try:
            visualize_aig_structure(graph, output_file=output_path)
            visualized_count += 1
        except Exception as e:
            logger.error(f"Failed to visualize graph {i}: {e}", exc_info=True)

    logger.info(f"Finished visualization. Successfully saved {visualized_count}/{num_to_visualize} images to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize generated AIGs saved in a pickle file.")

    parser.add_argument('input_pickle_file', type=str,
                        help='Path to the pickle file containing the list of generated NetworkX DiGraphs (e.g., generated_aigs.pkl).')
    parser.add_argument('--output-dir', type=str, default="visualizations/g2pt_aigs",
                        help='Directory to save the visualization images.')
    parser.add_argument('--num-visualize', type=int, default=10,
                        help='Maximum number of AIGs to visualize (from the start of the list).')

    parsed_args = parser.parse_args()
    main(parsed_args)