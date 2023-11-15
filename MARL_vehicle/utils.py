import os
import pandas as pd
import pickle

def check_and_create_path(filename):
    file_dir = os.path.split(filename)[0]
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)

def load_edge_ids(edge_data_file):
    edge_data = pd.read_csv(edge_data_file,dtype={"edge_id": str,"edge_pos_x":float, "edge_pos_y":float})
    return edge_data["edge_id"].tolist()

def load_neighbor_edges(data_file):
    fr = open(data_file, "rb")
    result = pickle.load(fr)
    return result

def load_simulation_data(data_file):
    fr = open(data_file, "rb")
    result = pickle.load(fr)
    return result