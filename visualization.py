import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
from datasets import load_data
from datasets import preprocess_pyg


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

    plt.title("Sample Graph Visualization")
    plt.show()

    # Save the plot
    plt.savefig('sample_graph.png')

def draw_soccer_field(ax=None):
    """
    Draws a 2D soccer field on a Matplotlib axis.
    """
    if ax is None:
        ax = plt.gca()

    # Soccer field dimensions (in meters)
    field_length = 105
    field_width = 68

    # Outline & center line
    ax.plot([0, 0, field_length, field_length, 0], [0, field_width, field_width, 0, 0], color="black")
    ax.plot([field_length / 2, field_length / 2], [0, field_width], color="black")

    # Center circle
    center_circle = plt.Circle((field_length / 2, field_width / 2), 9.15, color="black", fill=False)
    ax.add_artist(center_circle)

    # Penalty area (left & right)
    penalty_area_width = 40.3
    penalty_area_length = 16.5
    ax.plot([0, penalty_area_length, penalty_area_length, 0],
            [(field_width - penalty_area_width) / 2, (field_width - penalty_area_width) / 2,
             (field_width + penalty_area_width) / 2, (field_width + penalty_area_width) / 2],
            color="black")
    ax.plot([field_length, field_length - penalty_area_length, field_length - penalty_area_length, field_length],
            [(field_width - penalty_area_width) / 2, (field_width - penalty_area_width) / 2,
             (field_width + penalty_area_width) / 2, (field_width + penalty_area_width) / 2],
            color="black")

    # Set field limits and aspect ratio
    ax.set_xlim(0, field_length)
    ax.set_ylim(0, field_width)
    ax.set_aspect('equal')

def identify_teams_and_ball(data):
    """
    Identifies the ball and teams based on the attacking team flag.
    - Ball: The node with the highest degree.
    - Team 1: Nodes with Attacking Team Flag = 1 (attacking team).
    - Team 2: Nodes with Attacking Team Flag = 0 (defensive team).
    """
    # Convert PyG data to a NetworkX graph
    G = to_networkx(data, to_undirected=True)

    # Find the node with the highest degree (ball node)
    degrees = dict(G.degree())
    ball_node = max(degrees, key=degrees.get)

    # Team 1 and Team 2 based on Attacking Team Flag
    team1_nodes = [i for i in range(data.num_nodes) if data.x[i][-2].item() == 1]  # Attacking team
    team2_nodes = [i for i in range(data.num_nodes) if data.x[i][-2].item() == 0]  # Defensive team

    return G, ball_node, team1_nodes, team2_nodes

def visualize_soccer_graph(data):
    """
    Visualizes the soccer graph on a 2D soccer field with a legend.
    - Identifies the ball and teams based on the attacking team flag.
    """
    # Identify the ball and teams based on the flag
    G, ball_node, team1_nodes, team2_nodes = identify_teams_and_ball(data)

    # Extract node positions from node features and scale to field dimensions
    field_length = 105
    field_width = 68
    node_positions = {i: (data.x[i][0].item() * field_length, data.x[i][1].item() * field_width)
                      for i in range(data.num_nodes)}

    # Define colors for ball, attacking team, and defensive team
    node_colors = []
    for i in range(data.num_nodes):
        if i == ball_node:
            node_colors.append('orange')  # Ball color
        elif i in team1_nodes:
            node_colors.append('blue')  # Attacking team color
        else:
            node_colors.append('red')  # Defensive team color

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Draw the soccer field
    draw_soccer_field(ax)

    # Draw nodes (players/ball)
    nx.draw_networkx_nodes(G, node_positions, node_color=node_colors, node_size=300, ax=ax)

    # # Draw edges within teams and between team members and the ball
    # for u, v in G.edges():
    #     if (u == ball_node or v == ball_node) or (u in team1_nodes and v in team1_nodes) or (u in team2_nodes and v in team2_nodes):
    #         nx.draw_networkx_edges(G, node_positions, edgelist=[(u, v)], edge_color='gray', ax=ax)

    # Add labels for node indices
    nx.draw_networkx_labels(G, node_positions, ax=ax, font_size=10, font_color='black')

    # Add legend
    legend_labels = {'orange': 'Ball', 'blue': 'Attacking Team', 'red': 'Defensive Team'}
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[color],
                                 markerfacecolor=color, markersize=10) for color in legend_labels]
    ax.legend(handles=legend_handles, loc='upper right')

    plt.title("Sample Graph Visualization in Soccer Field")
    plt.show()
    
    # Save the plot
    plt.savefig('soccer_graph.png')


# Load the data
data = load_data.get_data('./datasets/women.pkl', local=True)
filtered_data = preprocess_pyg.filter_features(data.copy())
data_list = preprocess_pyg.process_data(filtered_data)

# Select a sample graph from the preprocessed dataset
sample_graph = data_list[0]

# Visualize the selected sample graph
visualize_graph_simple(sample_graph)
visualize_soccer_graph(sample_graph)
