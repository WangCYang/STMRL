import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import pickle

# 汇总每个时刻的流量数据
def generate_flow_data(data_file, edge_num, edge_index_map, save_file):
    data = pd.read_csv(data_file, dtype={"time":int,"vehicle_id": str, "edge_id":str, "distance": float})
    time_grouped_data = data.groupby("time")
    time_len = len(time_grouped_data)
    print(time_len)
    flow_array = np.zeros([time_len, edge_num])
    for time, current_data in time_grouped_data:
        if time % 1000 == 0:
            print(time)
        edge_grouped_data = current_data.groupby("edge_id")
        for edge, edge_data in edge_grouped_data:
            flow_array[int(time-1),int(edge_index_map[edge])] = len(edge_data)
    print(flow_array.shape)
    np.save(save_file, flow_array)

def standardlization(input_file, output_file, scalar_file):
    input_data = np.load(input_file, allow_pickle=True)
    # print(input_data.shape)
    # print(input_data)
    scaler = StandardScaler().fit(input_data.reshape(-1, 1))  # StandardScaler会自动将type转换为float 并按列进行求mean 和std，不需要额外进行astype
    print(scaler.mean_)
    scaler_file = open(scalar_file, "wb")
    # pickle.dump(scaler, scaler_file)

    data_ = scaler.transform(input_data)
    print(data_.shape)
    np.save(output_file, data_)

def sequence_generate(input_file,output_file,sequence_len):
    all_data = np.load(input_file, allow_pickle=True)
    all_time  = all_data.shape[0]
    input_list = []
    label_list = []
    for time_i in range(sequence_len,all_time):
        if time_i % 1000 == 0:
            print(time_i)

        input_i = np.array(all_data[time_i-sequence_len:time_i])
        label_i = np.array(all_data[time_i])

        input_list.append(input_i)
        label_list.append(label_i)

    print(np.array(input_list).shape)
    print(np.array(label_list).shape)
    np.savez(output_file,
        input=np.array(input_list),
        label=np.array(label_list)
    )

def split_graph_sequence(input_file, output_file, sequence_len):
    all_data = np.load(input_file, allow_pickle=True)
    print(all_data.shape)
    all_time = all_data.shape[0]
    all_data_tr = all_data.transpose((1,0))
    print(all_data_tr.shape)
    input_list = []
    label_list = []
    for time_i in range(sequence_len, all_time):
        # if time_i % 1000 == 0:
        #     print(time_i)

        input_i = np.array(all_data_tr[:,time_i - sequence_len:time_i])
        label_i = np.array(all_data_tr[:,time_i])

        input_list.append(input_i)
        label_list.append(label_i)

    print(np.array(input_list).shape)
    print(np.array(label_list).shape)
    np.savez(output_file,
        input=np.array(input_list),
        label=np.array(label_list)
    )

def generate_adjacent_matrix(neighibors, edge_indices, saved_file):
    edge_num = len(edge_indices.keys())
    adjacent_matrix = np.zeros((edge_num,edge_num))
    print(adjacent_matrix.shape)
    for edge_i, neigh_list in neighibors.items():
        for neigh_k in neigh_list:
            adjacent_matrix[edge_indices[edge_i],edge_indices[neigh_k]] = 1
    print(adjacent_matrix)
    np.save(saved_file, adjacent_matrix)

if __name__ == "__main__":
    ### Scenario： 3-3 grid
    # edge_index_map= {"J0": 0, "J1": 1, "J2": 2, "J3": 3, "J4": 4, "J5": 5, "J6": 6, "J7": 7, "J8": 8}
    # edge_num = 9

    ### Scenario： Net4
    # edge_index_map = {'J11':0, 'J12': 1, 'J0':2, 'J2':3, 'J3':4, 'J17':5, 'J6':6, 'J8':7, 'J9':8, 'J14':9, 'J7':10, 'J10':11, 'J15':12}
    # edge_num = 13
    # Simulation_scenario = "Net4"
    # Time_range = "24h"

    ### Scenario： bologna_pasubio
    edge_index_map = {'4': 0, '7': 1, '12': 2, '19': 3, '18': 4, '0': 5, '9': 6, '27': 7, '23': 8, '2': 9, '29': 10, '33': 11, '32': 12, '1': 13, '15': 14, '40': 15,
                      '39': 16, '36': 17}
    edge_num = 18
    Simulation_scenario = "bologna_pasubio"
    Time_range = "24h"

    '''
        ### Step1: 统计流量数据
    '''
    ### training data & test data
    # save_train_file = "flow_data/{}/{}/edge_flow.npy".format(Simulation_scenario,Time_range)
    # train_data_file = "../sumo/data/{}/{}/vehicle_edge_distance.csv".format(Simulation_scenario,Time_range)
    # train_data = generate_flow_data(train_data_file, edge_num, edge_index_map, save_train_file)

    # save_test_file = "flow_data/{}/{}/edge_flow.npy".format(Simulation_scenario,Time_range)
    # test_data_file = "../sumo/data/{}/{}/vehicle_edge_distance.csv".format(Simulation_scenario,Time_range)
    # test_data = generate_flow_data(test_data_file, edge_num, edge_index_map, save_test_file)

    '''
        ### Step2: 数据标准化并保存标准化器 (GCN-LSTM的预处理不再进行数据标准化处理，可以在模型输入前处理) （线性模型仍然需要这部分操作）
    '''
    # origin_file = "flow_data/{}/{}/edge_flow.npy".format(Simulation_scenario,Time_range)
    # standard_data_file = "flow_data/{}/{}/standard_edge_flow.npy".format(Simulation_scenario,Time_range)
    # scalar_file = "flow_data/{}/{}/standard.pkl".format(Simulation_scenario,Time_range)
    # standardlization(origin_file, standard_data_file, scalar_file)

    '''
        ### Step3.1: 划分序列 (非图结构数据，有重叠序列)
    '''
    # input_file = "flow_data/{}/{}/standard_edge_flow.npy".format(Simulation_scenario,Time_range)
    # output_file = "flow_data/{}/{}/flow_sequence.npz".format(Simulation_scenario,Time_range)
    # sequence_generate(input_file, output_file, 15)

    '''
        ### Step3.2: 划分序列 (图结构数据，有重叠序列)
    '''
    # input_file = "flow_data/{}/{}/edge_flow.npy".format(Simulation_scenario,Time_range)
    # output_file = "flow_data/{}/{}/flow_graph.npz".format(Simulation_scenario,Time_range)
    # input_len = 15  # 历史序列长度
    # split_graph_sequence(input_file, output_file, input_len)

    '''
        ### Step4: 邻接矩阵
    '''
    Edge_neighbors_file = "../sumo/data/{}/neighbor_edges.pkl".format(Simulation_scenario)
    Edge_neighbors = pickle.load(open(Edge_neighbors_file, "rb"))
    saved_file = "flow_data/{}/adjacent_matrix.npy".format(Simulation_scenario)
    generate_adjacent_matrix(Edge_neighbors, edge_index_map,saved_file)