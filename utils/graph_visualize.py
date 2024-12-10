import networkx as nx
import torch
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
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