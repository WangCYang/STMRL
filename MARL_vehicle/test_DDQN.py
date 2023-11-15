import tensorflow.python.client.device_lib

from multi_agent_env import Multi_Edge_Env, Multi_Edge_Env_no_neighbor  # 边缘计算环境
from multi_rl_agent_DDQN import RL_agent_DDQN   # single_net DDQN
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import pickle
from Flow_predciton import GCRNN, Flow_Loader
from draw_image import fig2plot
from utils import check_and_create_path, load_edge_ids, load_neighbor_edges, load_simulation_data
import tensorflow as tf

def now(step, now_epsilon): #reduce the exploration
    e_epsilon = 0.05
    if step % 200 == 0 and e_epsilon < now_epsilon:
        now_epsilon -= 0.05          # 探索率每次下降0.01
        return round(now_epsilon, 2)
    else:
        return round(now_epsilon, 2)

if __name__ == "__main__":
    # GPU状态设置
    # print(tensorflow.python.client.device_lib.list_local_devices())
    # gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    # print(gpus)
    # tf.config.experimental.set_visible_devices(devices=gpus[0],device_type="GPU")
    # tf.config.experimental.set_memory_growth(gpus[0], True)

    # environment
    # Simulation_scenario = "Net4"
    Simulation_scenario = "bologna_pasubio"
    Time_range = "24h"

    # num_edge = 9  # 3*3
    # num_edge = 13  # Net4
    num_edge = 18  # bologna_pasubio

    # num_bs = 10    # 基站数量
    # one_hop = 0.7  # 基站间转移时延0.7s/hop
    time_slots = 60  # 车辆移动过程中进行60个时间片（s）的任务，每个时间片一个任务,
    num_car_max = 10 # 最多十个车

    # DDQN 参数设置
    # num_hidden_nodes1 = 128  # for Net4, 3x3
    # num_hidden_nodes2 = 128
    num_hidden_nodes1 = 128  # for bologna_pasubio
    num_hidden_nodes2 = 128  # for bologna_pasubio
    memory_size = 10000         # experience pool
    replace_target_iter = 200   # network parameters replace  # 每200个迭代，更新一次target net的参数
    discount_factor = 0.90       # discount factor  # 累计奖励时的折扣因子
    batch_size = 512            # batch size of sampling   # 每次采样512个经验样本
    learning_rate = 0.001       # learning rate            # 梯度更新学习率
    now_epsilon = 1             # exploration value of begining  # 贪心选择中，进行expoloration（随机）的比例，初始为1

    # 流量预测模型参数设置
    history_time = 15
    gc_layer_sizes = [16, 10]
    lstm_layer_sizes = [200, 200]
    # scale_max = 15  # for 3x3 grid
    # scale_max = 24  # for Net4
    scale_max = 64  # for bologna_pasubio
    scale_min = 0
    # Edge_index_map = {"J0": 0, "J1": 1, "J2": 2, "J3": 3, "J4": 4, "J5": 5, "J6": 6, "J7": 7, "J8": 8}  # for for 3x3 grid
    # Edge_index_map = {'J11': 0, 'J12': 1, 'J0': 2, 'J2': 3, 'J3': 4, 'J17': 5, 'J6': 6, 'J8': 7, 'J9': 8, 'J14': 9, 'J7': 10, 'J10': 11, 'J15': 12}  # for Net4
    Edge_index_map =  edge_index_map = {'4': 0, '7': 1, '12': 2, '19': 3, '18': 4, '0': 5, '9': 6, '27': 7, '23': 8, '2': 9, '29': 10, '33': 11, '32': 12, '1': 13, '15': 14, '40': 15,
                      '39': 16, '36': 17}  # for bologna_pasubio

    # 多边缘服务器环境
    Edge_IDs = load_edge_ids("../sumo/data/{}/edge_position.csv".format(Simulation_scenario))
    Edge_neighbors = load_neighbor_edges("../sumo/data/{}/neighbor_edges.pkl".format(Simulation_scenario))

    e = Multi_Edge_Env(Edge_IDs, Edge_index_map, Edge_neighbors, num_edge, num_car_max, time_slots)           # new environment
    # e = Multi_Edge_Env_no_neighbor(Edge_IDs, Edge_neighbors, num_edge, num_car_max, time_slots)           # 不考虑邻居状态

    # RL_based Agents
    Multi_Agents = {}
    restore_dir = "models_DDQN/bologna_pasubio/v1_hidden_128_bs_64_epoch_1"
    for i in Edge_IDs:
        edge_i = e.edges[i]
        ### 考虑自身的流量预测，以及neighboring edge 的 state（任务量，算力，流量预测）
        rl_agent = RL_agent_DDQN(num_car_max*3 + num_car_max* 3 + 2 + 1 + len(edge_i.neighboring_edges)*3,  # state： 自身状态（任务信息，车辆信息，边缘服务器信息，流量预测） + 邻居节点的状态(任务总量, 算力，预测)
                                 int(math.pow(2,num_car_max)),  # action: 自身范围内车辆任务卸载决策 （2^K）
                                 num_hidden_nodes1,
                                 num_hidden_nodes2,
                                 discount_factor, memory_size, replace_target_iter,
                                 learning_rate, batch_size, flag=i)          # new an agent
        rl_agent.restore_model(restore_dir+"/edge{}".format(i)) # File要输对，没有错误提示; 存储位置默认没有设置数据集名
        # rl_agent.restore_model('models_DDQN/{}/edge{}'.format(Simulation_scenario,i)) # File要输对，没有错误提示; 存储位置默认没有设置数据集名

        ### 消融实验1，不考虑自身流量预测，考虑neighboring edge 的 state（任务量，算力）
        # rl_agent = RL_agent_DDQN(num_car_max*3 + num_car_max* 3 + 2 + len(edge_i.neighboring_edges)*2,  # state： 自身状态（任务信息，车辆信息以及边缘服务器信息） + 邻居节点的状态(任务总量, 算力)
        #                          int(math.pow(2,num_car_max)),  # action: 自身范围内车辆任务卸载决策 （2^K）
        #                          num_hidden_nodes1,
        #                          num_hidden_nodes2,
        #                          discount_factor, memory_size, replace_target_iter,
        #                          learning_rate, batch_size, '10')          # new an agent

        ### 消融实验2：不考虑neighboring edge 的 state
        # rl_agent = RL_agent_DDQN(num_car_max * 3 + num_car_max * 3 + 2,  # state： 自身状态（任务信息，车辆信息以及边缘服务器信息）
        #                          int(math.pow(2, num_car_max)),  # action: 自身范围内车辆任务卸载决策 （2^K）
        #                          num_hidden_nodes1,
        #                          num_hidden_nodes2,
        #                          discount_factor, memory_size, replace_target_iter,
        #                          learning_rate, batch_size, '10')  # new an agent
        Multi_Agents[i]= rl_agent

    # 流量预测模型
    adj_matrix = np.load("../sumo/data/{}/adjacent_matrix.npy".format(Simulation_scenario))  # 虽然spatiotemporal_prediction生成的，但是统一复制到sumo数据下
    FlowPredictionModel = GCRNN(gc_layer_sizes,lstm_layer_sizes,adj_matrix,history_time,scale_max,scale_min)
    FlowPredictionModel.load_weights("../spatiotemporal_prediction/saved_models/GCN_LSTM/{}/checkpoints/".format(Simulation_scenario))

    # 加载轨迹数据 (初步使用模拟和预处理好的数据，后续可改为实时调用的API)
    simulation_data_file = "../sumo/data/{}/{}/simulation_sequences.pkl".format(Simulation_scenario, Time_range)
    simulation_sequences = load_simulation_data(simulation_data_file)  # {start_time: [{edge_0:[v_1,v_2,...., ], edge_1:[]...edge_num:[]},{},...,time_len],}
    sequence_keys = list(simulation_sequences.keys())
    print("ALL episodes：{}".format(len(sequence_keys)))

    # 加载流量数据
    flow_loader = Flow_Loader("../sumo/data/{}/{}/edge_flow.npy".format(Simulation_scenario,Time_range))

    # 模型测试
    step = 0
    # 通过wordload analysis，找到对应的流量水平的区间
    # start_epi = 300  # low
    # start_epi = 1000  # medium
    start_epi   = 500  # high

    # episode = len(sequence_keys)  # 迭代次数
    episode = 100  # 迭代次数: 一个半小时的性能
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

        # 加载模拟数据
        start_time =  sequence_keys[epd]
        one_sequence = simulation_sequences[sequence_keys[epd]]

        # 进行预测
        query_flag, flow_history = flow_loader.query_history(start_time, history_time)
        future_flow = np.zeros(shape = num_edge)
        if query_flag == 1:
            future_flow = FlowPredictionModel.predict(np.expand_dims(flow_history, axis=0))[0]  # 扩展 batch 维度

        # 重置系统，更新状态
        current_step = 0
        all_task_num = 0
        task_num_now = e.reset(one_sequence[0], future_flow)          # init enevironment 每个episode 重置环境, 得到初始化任务状态

        # 开始模拟
        while True:
            # 每个agent根据当前自身的state，选择action
            all_task_num += task_num_now
            # now_epsilon = now(step, now_epsilon) # training 时 根据step 逐步降低exploration的比例
            now_epsilon = 0.05 # 测试时exploration设为较小固定值0.05
            states = {}
            actions = {}

            # 生成决策
            for i in Edge_IDs:
                agent_i = Multi_Agents[i]
                edge_i = e.edges[i]
                states[i] = edge_i.state
                a = agent_i.choose_action(edge_i.state, edge_i.avaiable_action, now_epsilon) # 选择动作
                actions[i] = a

            # 执行决策，更新状态
            if current_step < time_slots-1:
                current_step += 1 # 下一步模拟数据
            # 进行预测
            query_flag, flow_history = flow_loader.query_history(start_time+current_step, history_time)
            future_flow = np.zeros(shape=num_edge)
            if query_flag == 1:
                future_flow = FlowPredictionModel.predict(np.expand_dims(flow_history, axis=0))[0]
            rewards, wait_latencys, done, task_num_now, edge_util, vehicle_util = e.step(actions, one_sequence[current_step], future_flow) #环境执行动作，得到新状态，以及reward

            edge_utils.append(edge_util)
            vehicle_utils.append(vehicle_util)

            all_latency += np.sum([x*(-1) for x in rewards.values()])
            all_wait_latency += np.sum([x for x in wait_latencys.values()])
            # e_con += (r2*-4.5)
            step += 1
            if done:
                aver_latency = all_latency/time_slots
                break

        print('total latency, %.4f, average latency per time, %.4f, average latency per task: %.4f ,average wait latency per task: %.4f' % (
                all_latency, aver_latency, all_latency / all_task_num, all_wait_latency / all_task_num))
        print('Edge util, {}, Vehicle util {}'.format(np.mean(edge_utils), np.mean(vehicle_utils)))
        avr_per_task[count_i] = all_latency/all_task_num
        avr_wait_per_task[count_i] = all_wait_latency / all_task_num
        edge_util_per_time[count_i] = np.mean(edge_utils)
        vehicle_util_per_time[count_i] = np.mean(vehicle_utils)
        count_i +=1

    print("Test average latency per task: {}, Final average wait latency per task:{} ".format(np.mean(avr_per_task),np.mean(avr_wait_per_task)))
    print("Test average edge util per time: {}, Final average vehicle util per time:{} ".format(np.mean(edge_util_per_time),np.mean(vehicle_util_per_time)))
    saved_avr_laten_file = "saved_data/{}/MARL_test_MADDQN_range_{}_{}.pkl".format(Simulation_scenario,start_epi,start_epi+episode)
    check_and_create_path(saved_avr_laten_file)
    fw = open(saved_avr_laten_file, "wb")
    pickle.dump(avr_per_task, fw)

    # fig(episode, avr_per_task,'Test Average Latency Per Task') # 画出曲线
    flag = "Test Average Latency Per Task"
    saved_fig_file = "figv0/{}/{}_STMARL_range_{}_{}.png".format(Simulation_scenario,flag, start_epi,start_epi+episode)
    check_and_create_path(saved_fig_file)
    fig2plot(episode, avr_per_task, avr_wait_per_task, flag, saved_fig_file) # 画出曲线