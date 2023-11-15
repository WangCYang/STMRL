import numpy as np
from utils import check_and_create_path, load_simulation_data
from draw_image import fig_hist, fig_scatter, fig_plot, fig_bar

def flow_statistics_per_server_per_time(input_file, edge_num, edge_index, saved_file):
    simulation_sequences = load_simulation_data(input_file)  # {start_time: [{edge_0:[v_1,v_2,...., ], edge_1:[]...edge_num:[]},{},...,time_len],}
    seq_keys = list(simulation_sequences.keys())
    seq_num = len(seq_keys)
    seq_len = len(list(simulation_sequences.values())[0])
    flow_statis = np.zeros(shape=(seq_num,seq_len,edge_num))

    time_index = 0
    for t, seq_t in simulation_sequences.items():
        for i, step_i in enumerate(seq_t):
            for edge_k, vehicle_dict in step_i.items():
                edge_id = edge_index[edge_k]
                flow_statis[time_index,i,edge_id] = len(vehicle_dict)

        if time_index % 100 == 0 and time_index != 0:
            print(time_index)
        time_index += 1

    np.savez(saved_file,
             start_time_keys=np.array(seq_keys),
             flow_statis=flow_statis
             )

def flow_per_eposide(input_file, saved_file):
    flow_statis = np.load(input_file, allow_pickle=True)["flow_statis"]
    eposide_flow = np.sum(flow_statis,axis=(1,2))
    print(len(eposide_flow), eposide_flow)
    np.save(saved_file,eposide_flow)

def flow_per_edge(input_file, saved_file):
    flow_statis = np.load(input_file, allow_pickle=True)["flow_statis"]
    edge_flow = np.sum(flow_statis, axis=(0, 1))
    print(len(edge_flow), edge_flow)
    np.save(saved_file, edge_flow)

Simulation_scenario = "bologna_pasubio"
Time_range = "24h"

'''
    Step1: 统计per_server_per_time 的流量信息
'''
# simulation_data_file = "../sumo/data/{}/{}/simulation_sequences.pkl".format(Simulation_scenario, Time_range)
# saved_flow_statis_file = "saved_data/{}/flow_statistic/{}.npz".format(Simulation_scenario, "Flow_server_time")
# check_and_create_path(saved_flow_statis_file)
# edge_num = 18
# edge_index_map = {'4': 0, '7': 1, '12': 2, '19': 3, '18': 4, '0': 5, '9': 6, '27': 7, '23': 8, '2': 9, '29': 10, '33': 11, '32': 12, '1': 13, '15': 14, '40': 15,
#                       '39': 16, '36': 17}  # 确保和其他预测一致
# flow_statistics_per_server_per_time(simulation_data_file,edge_num, edge_index_map, saved_flow_statis_file)

'''
    Step2: 统计scenario-level per-eposide的流量
'''
# flow_statis_file = "saved_data/{}/flow_statistic/{}.npz".format(Simulation_scenario, "Flow_server_time")
# saved_flow_episode_file = "saved_data/{}/flow_statistic/{}.npy".format(Simulation_scenario, "Flow_episode")
# flow_per_eposide(flow_statis_file, saved_flow_episode_file)
'''
    Step3: 绘制流量图，观察流量整体动态变化情况，筛选中高低流量区域
'''
# flow_eposide_file = "saved_data/{}/flow_statistic/{}.npy".format(Simulation_scenario, "Flow_episode")
# flow_eposide = np.load(flow_eposide_file)
# # fig_scatter(np.arange(len(flow_eposide)), flow_eposide)
# fig_plot(np.arange(len(flow_eposide)-5), flow_eposide[5:])

'''
    Step4: 按 edge 统计流量，找到流量差异比较大的局部，方便找 流量动态变化的案例分析
'''
# flow_statis_file = "saved_data/{}/flow_statistic/{}.npz".format(Simulation_scenario, "Flow_server_time")
# saved_flow_edge_file = "saved_data/{}/flow_statistic/{}.npy".format(Simulation_scenario, "Flow_edge")
# flow_per_edge(flow_statis_file, saved_flow_edge_file)

'''
    Step5： 找到对应的位置，绘制柱状图，观察总流量
'''
# edge_index_map = {'4': 0, '7': 1, '12': 2, '19': 3, '18': 4, '0': 5, '9': 6, '27': 7, '23': 8, '2': 9, '29': 10, '33': 11, '32': 12, '1': 13, '15': 14, '40': 15,
#                       '39': 16, '36': 17}
# flow_edge_file = "saved_data/{}/flow_statistic/{}.npy".format(Simulation_scenario, "Flow_edge")
# flow_edge = np.load(flow_edge_file)
# fig_bar(list(edge_index_map.keys()), flow_edge)

'''
    Step6: 通过折线图，观察各节点的流量变化，找到递增或者递减区间，从而筛选出具有这个特点的
'''