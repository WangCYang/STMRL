from multi_agent_env import Multi_Edge_Env  # 边缘计算环境
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd 
import pickle
from draw_image import fig2plot
from utils import check_and_create_path, load_edge_ids, load_neighbor_edges, load_simulation_data

def fig(episode, nums, flag):
    plt.figure()
    plt.plot(np.array(np.arange(0, episode)), nums)
    plt.xlabel('Step', fontsize=15)
    plt.ylabel(flag, fontsize=15)
    plt.grid(linestyle=':')
    plt.savefig('./figv0/'+flag+'.png')

if __name__ == "__main__":

    # environment
    Simulation_scenario = "bologna_pasubio"
    Time_range = "24h"

    # num_edge = 9  # 3*3
    # num_edge = 13  # Net4
    num_edge = 18  # bologna_pasubio

    # num_bs = 10    # 基站数量
    # one_hop = 0.7  # 基站间转移时延0.7s/hop
    time_slots = 60  # 车辆移动过程中进行60个时间片（s）的任务，每个时间片一个任务,
    num_car_max = 10 # 最多十个车

    # Edge_index_map = {"J0": 0, "J1": 1, "J2": 2, "J3": 3, "J4": 4, "J5": 5, "J6": 6, "J7": 7, "J8": 8}  # for for 3x3 grid
    # Edge_index_map = {'J11': 0, 'J12': 1, 'J0': 2, 'J2': 3, 'J3': 4, 'J17': 5, 'J6': 6, 'J8': 7, 'J9': 8, 'J14': 9,
    #                   'J7': 10, 'J10': 11, 'J15': 12}  # for Net4
    Edge_index_map = edge_index_map = {"4": 0, '7': 1, '12': 2, '19': 3, '18': 4, '0': 5, '9': 6, '27': 7, '23': 8,
                                       '2': 9, '29': 10, '33': 11, '32': 12, '1': 13, '15': 14, '40': 15,
                                       '39': 16, '36': 17}  # for bologna_pasubio

    # 多边缘服务器环境
    # Edge_IDs = load_edge_ids("../sumo/data/3-3-grid-1h/edge_position.csv")
    # Edge_neighbors = load_neighbor_edges("../sumo/data/3-3-grid-1h/neighbor_edges.pkl")
    Edge_IDs = load_edge_ids("../sumo/data/{}/edge_position.csv".format(Simulation_scenario))
    Edge_neighbors = load_neighbor_edges("../sumo/data/{}/neighbor_edges.pkl".format(Simulation_scenario))
    e = Multi_Edge_Env(Edge_IDs, Edge_index_map, Edge_neighbors, num_edge, num_car_max, time_slots)  # new environment

    # 加载真实数据
    simulation_data_file = "../sumo/data/{}/{}/simulation_sequences.pkl".format(Simulation_scenario, Time_range)
    simulation_sequences = load_simulation_data(simulation_data_file)
    sequence_keys = list(simulation_sequences.keys())
    print("ALL episodes：{}".format(len(sequence_keys)))

    # 通过wordload analysis，找到对应的流量水平的区间
    # start_epi = 300  # low
    # start_epi = 1000  # medium
    start_epi = 500  # high

    episode = 100  # 迭代次数: 前一个小时的性能
    avr_per_task = np.zeros(episode)  # 记录每次episode的结果
    avr_wait_per_task = np.zeros(episode)
    edge_util_per_time = np.zeros(episode)
    vehicle_util_per_time = np.zeros(episode)
    count_i = 0
    for epd in range(start_epi, start_epi + episode):
        print('||                      Episode:', epd, '                            ||')
        all_latency = 0              # record latency consumption
        all_wait_latency = 0
        edge_utils = []
        vehicle_utils = []

        # 加载一条模拟数据
        one_sequence = simulation_sequences[sequence_keys[epd]]
        current_step = 0
        all_task_num = 0
        future_flow = np.zeros(shape=num_edge)  # 为了统一系统建模
        task_num_now = e.reset(one_sequence[0],future_flow)          # init enevironment 每个episode 重置环境, 得到初始化任务状态
        while True:
            # 每个agent根据当前自身的state，选择action
            all_task_num += task_num_now
            states = {}
            actions = {}
            # for i in range(len(Multi_Agents)):
            for i in Edge_IDs:
                actions[i] = 0

            # 整体执行
            if current_step < time_slots-1:
                current_step += 1

            # print("....",current_step,"....")
            rewards,wait_latencys, done, task_num_now, edge_util, vehicle_util = e.step(actions, one_sequence[current_step],future_flow) #环境执行动作，得到新状态，以及reward
            edge_utils.append(edge_util)
            vehicle_utils.append(vehicle_util)

            all_latency += np.sum([x*(-1) for x in rewards.values()])
            all_wait_latency += np.sum([x for x in wait_latencys.values()])
            if done:
                aver_latency = all_latency/time_slots
                break

        print('total latency, %.4f, average latency per time,%.4f, average latency per task: %.4f ,average wait latency per task: %.4f' % (
            all_latency, aver_latency, all_latency / all_task_num, all_wait_latency / all_task_num))
        print('Edge util, {}, Vehicle util {}'.format(np.mean(edge_utils), np.mean(vehicle_utils)))
        avr_per_task[count_i] = all_latency / all_task_num
        avr_wait_per_task[count_i] = all_wait_latency / all_task_num
        edge_util_per_time[count_i] = np.mean(edge_utils)
        vehicle_util_per_time[count_i] = np.mean(vehicle_utils)
        count_i += 1

    print("Test average latency per task: {}, Final average wait latency per task:{} ".format(np.mean(avr_per_task),
                                                                                              np.mean(
                                                                                                  avr_wait_per_task)))
    print("Test average edge util per time: {}, Final average vehicle util per time:{} ".format(
        np.mean(edge_util_per_time), np.mean(vehicle_util_per_time)))

    saved_avr_laten_file = "saved_data/{}/Local_test_range_{}_{}.pkl".format(Simulation_scenario, start_epi,
                                                                                   start_epi + episode)
    check_and_create_path(saved_avr_laten_file)
    fw = open(saved_avr_laten_file, "wb")
    pickle.dump(avr_per_task, fw)

    # fig(episode, avr_per_task,'Test Average Latency Per Task') # 画出曲线
    flag = "Test Average Latency Per Task"
    saved_fig_file = "figv0/{}/{}_Local_range_{}_{}.png".format(Simulation_scenario, flag, start_epi,
                                                                 start_epi + episode)
    check_and_create_path(saved_fig_file)
    fig2plot(episode, avr_per_task, avr_wait_per_task, flag, saved_fig_file)  # 画出曲线