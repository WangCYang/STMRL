import pandas as pd
import numpy as np
import pickle
from scipy.spatial import distance

def closest_edge(edge_pos_df, vehicle_pos):
    dis_list = []
    for index, edge_i in edge_pos_df.iterrows():
        edge_i_pos = [edge_i["edge_pos_x"], edge_i["edge_pos_y"]]
        dist_i =  distance.cdist(np.array(edge_i_pos).reshape(1,-1), np.array(vehicle_pos).reshape(1,-1), metric='euclidean')[0][0]
        dis_list.append(dist_i)
    min_index = np.argmin(dis_list)
    closest_edge_id = edge_pos_df.iloc[min_index]["edge_id"]
    return closest_edge_id, dis_list[min_index]

def split_simulation_data(datafile, time_len, save_file):
    simulation_data = pd.read_csv(datafile,dtype={"time":int,"vehicle_id": str, "edge_id":str, "distance": float})

    # 按time_len划分出不同的序列数据，每个时刻记录每个edge对应的车辆ID
    sequences_list = {}
    time_grouped_data = simulation_data.groupby("time")
    start_time = 1
    one_sequence = []
    count = 0
    for time, current_data in time_grouped_data:
        if time % 1000 == 0:
            print(time)
        if count == time_len:
            sequences_list[start_time] = one_sequence
            count = 0
            start_time = time
            one_sequence = []

        edge_vehicle_map = {}
        for index, one_vehicle in current_data.iterrows(): # 每个时刻的车辆
            edge_id = one_vehicle["edge_id"]
            vehicle_id = one_vehicle["vehicle_id"]
            if edge_id not in edge_vehicle_map:
                edge_vehicle_map[edge_id] = [vehicle_id]
            else:
                edge_vehicle_map[edge_id].append(vehicle_id)
        one_sequence.append(edge_vehicle_map)
        count += 1
    print(len(sequences_list))
    fw = open(save_file, "wb")
    pickle.dump(sequences_list, fw)

if __name__ == "__main__":
    # Simulate_scenario = "3-3-grid-10h"

    # Simulate_scenario = "Net4"
    # simulation_period = "24h"
    # simulation_period = "1h_1"
    # simulation_period = "1h_2"
    # simulation_period= "1h_3"

    Simulate_scenario = "bologna_pasubio"
    simulation_period = "24h"

    '''
        Step 1: 对每个时刻，分配车辆-sever 连接关系
    '''
    ##读取edge位置数据
    # edge_data_file = "data/{}/edge_position.csv".format(Simulate_scenario)
    # edge_data = pd.read_csv(edge_data_file)

    ## 读取轨迹数据
    # simulation_data_file = "data/{}/{}/vehicle_simulation.csv".format(Simulate_scenario, simulation_period)
    # simulation_data = pd.read_csv(simulation_data_file)
    # print("Total records:", len(simulation_data))
    # print("Total vehicles:",
    #       len(simulation_data.groupby("vehicle_id")))  # 3-3-grid-10h: 6条路由，每条每小时200个车，120w个记录，每个车平均1k个记录

    # connection_data_datafile = "data/{}/{}/vehicle_edge_distance.csv".format(Simulate_scenario,simulation_period)
    # connection_data = pd.DataFrame(columns=["time","vehicle_id", "edge_id", "distance"],data = [])
    # connection_data.to_csv(connection_data_datafile, index=False)
    #
    # time_grouped_data = simulation_data.groupby("time")
    # for time, current_data in time_grouped_data:
    #     if time % 1000 == 0:
    #         print(time)
    #     current_connection_data = []
    #     # print(current_data)
    #     for index, one_vehicle in current_data.iterrows():
    #         # print(one_vehicle)
    #         one_vehicle_pos = [one_vehicle["pos_x"],one_vehicle["pos_y"]]
    #         edge_connected, min_distance = closest_edge(edge_data, one_vehicle_pos)
    #         current_connection_data.append([time, one_vehicle["vehicle_id"], edge_connected, min_distance])
    #     pd.DataFrame(current_connection_data).to_csv(connection_data_datafile, mode='a', index=False, header=False)

    '''
            Step 1.1: 对于bologna_pasubio场景, edge_id 和vehicle_id 会自动存成 float， 需要改变dframe的dtypes
    '''
    # new_connection_data_datafile = "data/{}/{}/vehicle_edge_distance.csv".format(Simulate_scenario, simulation_period)
    # simulation_data_file = "data/{}/{}/vehicle_edge_distance.csv".format(Simulate_scenario, simulation_period)
    # simulation_data = pd.read_csv(simulation_data_file,dtype={"time":int,"vehicle_id": int, "edge_id":int, "distance": float})
    # simulation_data.astype({"time":int,"vehicle_id": str, "edge_id":str, "distance": float})
    # simulation_data.to_csv(new_connection_data_datafile, index=False)

    '''
        Step 2: 切分成固定时间长度的 sequence-form simulation数据
    '''
    sequence_len = 60
    simulation_data_file = "data/{}/{}/vehicle_edge_distance.csv".format(Simulate_scenario,simulation_period)
    saved_file = "data/{}/{}/simulation_sequences.pkl".format(Simulate_scenario,simulation_period)
    sequence_data = split_simulation_data(simulation_data_file, sequence_len, saved_file)

    '''
    
    '''