import pandas as pd
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.pyg_converter import create_pyg_dataset, save_graph_list
df = pd.read_parquet('processed_data/men_imbalanced_node_features_checked.parquet')
df.drop(columns=['player_num_label'], inplace=True)

pyg_data_list = create_pyg_dataset(df)

# Save the PyG dataset
save_graph_list(pyg_data_list, 'processed_data', "men_imbalanced_graph_dataset") 
length_data_list = len(pyg_data_list)
print(f"Save {length_data_list} graphs to processed_data.")