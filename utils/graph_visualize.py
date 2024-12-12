import networkx as nx
import torch
from torch_geometric.utils import to_networkx
import matplotsoccer
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
def visualize_graph_simple(data, node_color='lightblue', edge_color='gray'):
    """
    Visualize a single graph from the PyG dataset without showing vectors.
    """
    # Convert PyG data to a NetworkX graph
    G = to_networkx(data, to_undirected=True)

    # Set up node labels as simple indices
    node_labels = {i: i for i in range(data.num_nodes)}

    # Plotting
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G)  # Spring layout for better visualization

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=500, alpha=0.8)

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, alpha=0.5)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color='black')

    plt.title("Sample Graph from PyG Dataset")
    plt.show()



def visualize_head_tail_pyg_data(pyg_data_list, n=10, figsize=(20, 8)):
    # Get first n and last n graphs
    head_graphs = pyg_data_list[:n]
    tail_graphs = pyg_data_list[-n:]
    
    # Create subplot for head graphs
    plt.figure(figsize=figsize)
    plt.suptitle("First 10 Graphs", y=1.02, fontsize=12)
    for idx, data in enumerate(head_graphs, 1):
        plt.subplot(2, 5, idx)
        G = nx.Graph()
        for i in range(data.num_nodes):
            G.add_node(i)
        edge_index = data.edge_index.numpy()
        edges = list(zip(edge_index[0], edge_index[1]))
        G.add_edges_from(edges)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, node_color='lightblue', 
                with_labels=True, node_size=300,
                font_size=6, font_weight='bold')
        plt.title(f'Graph {idx}\nNodes: {data.num_nodes}\nLabel: {data.y.item()}')
    plt.tight_layout()
    plt.show()

    # Create subplot for tail graphs
    plt.figure(figsize=figsize)
    plt.suptitle("Last 10 Graphs", y=1.02, fontsize=12)
    for idx, data in enumerate(tail_graphs, 1):
        plt.subplot(2, 5, idx)
        G = nx.Graph()
        for i in range(data.num_nodes):
            G.add_node(i)
        edge_index = data.edge_index.numpy()
        edges = list(zip(edge_index[0], edge_index[1]))
        G.add_edges_from(edges)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, node_color='lightblue', 
                with_labels=True, node_size=300,
                font_size=6, font_weight='bold')
        plt.title(f'Graph {len(pyg_data_list)-n+idx}\nNodes: {data.num_nodes}\nLabel: {data.y.item()}')
    plt.tight_layout()
    plt.show()

    # Print basic statistics
    print(f"\nDataset Statistics:")
    print(f"Total number of graphs: {len(pyg_data_list)}")
    print(f"Features per node: {pyg_data_list[0].x.shape[1]}")
    print(f"Graph labels distribution: ")
    labels = torch.tensor([data.y for data in pyg_data_list])
    unique_labels, counts = torch.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Label {label.item()}: {count.item()} graphs")
        
def visualize_soccer_graph(data, save_path, field_length=105, field_width=68, node_size=2):
    """
    Create and save a visualization of a soccer graph with labeled nodes.
    
    Args:
        data: PyG data object containing node features and edge information
        save_path: Path where the visualization should be saved
        field_length: Length of the soccer field in meters (default: 105m - standard)
        field_width: Width of the soccer field in meters (default: 68m - standard)
        node_size: Size of the nodes in the visualization (default: 2)
    """
    # Calculate the aspect ratio for the field
    aspect_ratio = field_length / field_width
    
    # Set up the plot with correct proportions
    fig, ax = plt.subplots(figsize=(15, 15/aspect_ratio))
    
    # Create soccer field using matplotsoccer
    matplotsoccer.field("green", show=False, ax=ax)
    
    # Set field dimensions
    ax.set_xlim(0, field_length)  # Length is along x-axis (105m)
    ax.set_ylim(0, field_width)   # Width is along y-axis (68m)
    
    # Create graph representation
    G = to_networkx(data, to_undirected=True)
    
    # Initialize position dictionaries
    pos = {}
    defense_pos = {}
    offense_pos = {}
    ball_pos = {}
    
    # Process node positions with correct scaling
    for i in range(data.num_nodes):
        x = data.x[i][0].item() * field_length
        y = data.x[i][1].item() * field_width
        pos[i] = (x, y)
        
        team_value = data.x[i][-9].item()
        if team_value == -1:  # Ball
            ball_pos[i] = (x, y)
        elif team_value == 1:  # Offense
            offense_pos[i] = (x, y)
        else:  # Defense
            defense_pos[i] = (x, y)
    
    # Draw edges
    edge_list = list(G.edges())
    if edge_list:
        edge_coords = [(pos[e[0]], pos[e[1]]) for e in edge_list]
        edge_collection = plt.matplotlib.collections.LineCollection(
            edge_coords, colors='gray', alpha=0.3, linewidth=1
        )
        ax.add_collection(edge_collection)
    
    # Draw nodes and labels for each team
    for pos_dict, color, label in [(defense_pos, 'blue', 'D'), (offense_pos, 'red', 'O')]:
        if pos_dict:
            x_coords = [coord[0] for coord in pos_dict.values()]
            y_coords = [coord[1] for coord in pos_dict.values()]
            
            # Draw white-filled circles with colored edges
            ax.scatter(x_coords, y_coords, s=node_size*100, 
                      facecolor='white', edgecolor=color, linewidth=2)
            
            # Add labels inside the circles
            for x, y in zip(x_coords, y_coords):
                ax.text(x, y, label, color=color, 
                       ha='center', va='center', 
                       fontweight='bold', fontsize=8)
    
    # Draw ball
    if ball_pos:
        x_coords = [coord[0] for coord in ball_pos.values()]
        y_coords = [coord[1] for coord in ball_pos.values()]
        ax.scatter(x_coords, y_coords, s=node_size*100, 
                  facecolor='black', edgecolor='black')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                  markeredgecolor='red', label='Offense', markersize=8, linewidth=0),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                  markeredgecolor='blue', label='Defense', markersize=8, linewidth=0),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                  label='Ball', markersize=8, linewidth=0)
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Save the visualization with proper aspect ratio
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_sample_visualizations(pyg_data_list, save_dir='visualizations', num_samples=3):
    """
    Save visualizations of sample graphs from the dataset.
    
    Args:
        pyg_data_list: List of PyG data objects
        save_dir: Directory where visualizations should be saved
        num_samples: Number of random samples to visualize
    """
    # Create save directory if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Select random samples
    indices = np.random.choice(len(pyg_data_list), num_samples, replace=False)
    
    for idx in indices:
        data = pyg_data_list[idx]
        save_path = save_dir / f'soccer_graph_sample_{idx}.png'
        visualize_soccer_graph(data, save_path)
        print(f'Saved visualization to {save_path}')