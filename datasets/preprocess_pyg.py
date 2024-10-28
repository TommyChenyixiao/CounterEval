import numpy as np
import torch
from torch_geometric.data import Data

# Edge feature selection (set to True/False)
use_player_dist = True
use_speed_diff_matrix = True
use_pos_sin_angle = False
use_pos_cos_angle = False
use_vel_sin_angle = True
use_vel_cos_angle = True

# Node feature selection (set to True/False)
use_x_coord = True
use_y_coord = True
use_vx = True
use_vy = True
use_v = False
use_velocity_angle = False
use_dist_goal = True
use_goal_angle = True
use_dist_ball = False
use_ball_angle = False
use_is_attacking = True
use_potential_receiver = True

def filter_features(data, adj_matrix='normal'):
    """
    Filter the node and edge features based on user-defined flags.
    """
    # Define edge and node feature indices based on selection flags
    edge_feature_flags = [
        use_player_dist, use_speed_diff_matrix, use_pos_sin_angle, 
        use_pos_cos_angle, use_vel_sin_angle, use_vel_cos_angle
    ]
    node_feature_flags = [
        use_x_coord, use_y_coord, use_vx, use_vy, use_v, 
        use_velocity_angle, use_dist_goal, use_goal_angle, 
        use_dist_ball, use_ball_angle, use_is_attacking, 
        use_potential_receiver
    ]

    # Get indices of selected features
    edge_feature_idxs = [idx for idx, flag in enumerate(edge_feature_flags) if flag]
    node_feature_idxs = [idx for idx, flag in enumerate(node_feature_flags) if flag]

    # Check for valid feature selection
    if not edge_feature_idxs and not node_feature_idxs:
        print("\nCannot have zero edge and zero node features.\n")
        return data

    # Filter node and edge features in the data
    data[adj_matrix]['e'] = [x[:, edge_feature_idxs] for x in data[adj_matrix]['e']]
    data[adj_matrix]['x'] = [x[:, node_feature_idxs] for x in data[adj_matrix]['x']]

    return data

def process_data(data, adj_matrix='normal'):
    """
    Converts the filtered dataset to PyTorch Geometric format.
    """
    data_mat = data[adj_matrix]
    graph_list = []

    for x, a, e, y in zip(data_mat['x'], data_mat['a'], data_mat['e'], data['binary']):
        # Convert sparse matrix to COO format (edge_index)
        a_coo = a.tocoo()  # Convert sparse matrix to COO format

        # Convert edge indices to numpy array and then to tensor
        edge_index = np.vstack([a_coo.row, a_coo.col])
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        # Create PyG Data object
        graph_data = Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=torch.tensor(e, dtype=torch.float),
            y=torch.tensor(y, dtype=torch.float).view(1, -1)
        )
        graph_list.append(graph_data)

    return graph_list

# Example usage 
# Load the data
if __name__ == "__main__":
    import load_data
    data = load_data.get_data('./datasets/women.pkl', local=True)
    filtered_data = filter_features(data.copy())
    data_list = process_data(filtered_data)

    # Display the number of graphs processed
    print(f"Number of graphs processed: {len(data_list)}")