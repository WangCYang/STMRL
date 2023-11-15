import numpy as np
import random
import math
import pandas as pd
import queue

'''
多个Edge,互不重叠; Edge 算力固定;
每个时刻，每个基站(最多) K个car; 每个基站分布式地进行自身车辆的决策; 
Agent的学习，每个车辆独立地进行，维持自己的训练样本池；

Agent: 除了自身的任务状态，还包括邻居结点的状态（当前任务总量，当前邻居节点算力，上一时刻的决策） 

state: K个car的任务状态,edge当前的算力, 队列;
action: K个car的卸载决策; (0-1的卸载量？或者0-1的卸载决策) 
cost: K个车的时延总和; 
'''
class Edge:
    def __init__(self, edge_id, neighbors, num_edge, num_car_max=10, cpu=20, bw=10, initial_buff=0):

        self.edge_id = edge_id
        self.num_edge = num_edge
        self.car_max = num_car_max

        self.default_settings = {"cpu": cpu, "bw" : bw, "initial_buff": initial_buff}
        # edge 状态
        self.cpu = cpu   # edge server 算力
        self.bw = bw     # edge server 带宽
        self.buff_CPU = initial_buff  # 任务量队列(以所需CPU时间为单位)
        self.buff_pool = queue.Queue() # 任务队列池（记录任务缓存情况）

        # edge 计算及通信建模 （待调整）
        self.tran_grads = 2  # 数据卸载频率
        self.grads = 2
        self.k = 1 / 3
        self.noise = 2 / np.power(10, 13)

        # 邻居 edges
        self.neighboring_edges = [] # 根据实际路网连接情况而确定
        self.set_neighboring_edges(neighbors)

        # 连接 车辆
        self.connected_vehicles = []

        # 未来流量预测
        self.future_flow = 0

    def set_neighboring_edges(self, neighbors):
        # 暂定前后一个id的edge为neighbor
        # if self.edge_id - 1 >= 0:
        #     self.neighboring_edges.append(self.edge_id - 1)
        # if self.edge_id + 1 < self.num_edge:
        #     self.neighboring_edges.append(self.edge_id + 1)

        self.neighboring_edges = neighbors

    def reset(self):
        self.cpu = self.default_settings["cpu"]  # edge server 算力
        self.bw =  self.default_settings["bw"]  # edge server 带宽
        self.buff_CPU =  self.default_settings["initial_buff"]  # 任务量队列(以所需CPU时间为单位)
        self.buff_pool = queue.Queue()  # 重置队列

        self.connected_vehicles = [] # 清空连接车辆

        self.future_flow = 0 # 重置预测值

    def vehicle_connection(self, vehicles):
        self.connected_vehicles = vehicles

    def update_futureflow(self,future_flow):
        self.future_flow = future_flow

    def get_channel_gain(self, distance):
        # if distance == 50:
        #     channel = 60
        # elif distance == 100:
        #     channel = 50
        # elif distance == 150:
        #     channel = 40
        # return channel
        return 50 # 暂时固定，不考虑位置对信道状态的影响

    def record_future_flow(self,future_flow):
        self.future_flow = future_flow

    def ava_a(self):
        #
        num_current_vehicle = len(self.connected_vehicles)
        if num_current_vehicle >=  self.car_max:
            mask = np.ones(int(math.pow(2,self.car_max)))  # 超过K个车辆
        else:
            mask = np.zeros(int(math.pow(2,self.car_max)))
            mask[0:int(math.pow(2,num_current_vehicle))] = 1
        return mask

    def get_current_task_sum(self,tasks):
        sum = 0
        for task_i in tasks:
            sum += task_i[0]
        return sum

class Task:
    def __init__(self, data_size, CPU_cycle, ddl):
        # 任务属性
        self.data_size = data_size
        self.CPU_cycle = CPU_cycle
        self.ddl = ddl

        self.vehicle_id = -1

class Task_pool:
    def __init__(self):
        # (dara_size, CPU_cycle, DDl)
        self.pool = [[90, 3, 5], [60, 3, 3], [90, 5, 3], [60, 5, 5]] # 任务的固定模式, 暂不考虑任务模式的随机

    def random_sample_n(self, size):
        task_list = []
        sampled_index = np.random.choice([0,1,2,3], size=size, p = [0.25,0.25,0.25,0.25])
        for i in sampled_index:
            task_i = self.pool[i]
            one_task = Task(task_i[0], task_i[1], task_i[2])
            task_list.append(one_task)
        return task_list

class Vehicle:
    def __init__(self, vehicle_id, cpu=2):

        self.vehicle_id = vehicle_id

        # vehicle 状态
        self.cpu = cpu   # edge server 算力
        self.buff_CPU = 0  # 任务量队列(以所需CPU时间为单位)
        self.buff_pool = queue.Queue() # 任务队列池（记录任务缓存情况）
        self.channel_gain = 50

        # edge 连接
        self.connected_edge = -1
        self.distance = -1  # 暂时不考虑
        # vehicle 计算及通信建模 （待调整）

        # 当前任务信息
        self.current_task = -1

    def edge_connection(self, edge_id, distance = -1):
        self.connected_edge = edge_id
        self.distance = distance

    def generate_task(self, task_pool):
        one_task = task_pool.random_sample_n(size=1)[0]
        self.current_task = one_task

class Multi_Edge_Env:
    def __init__(self,
                 Edge_IDs,
                 Edge_index_map,
                 Edge_neighbors,
                 num_edge = 9,
                 num_car_max = 10,
                 time_slot = 60,
                 ):
        # 设置edge
        self.edges = {}
        for i in Edge_IDs:
            edge_i = Edge(i,Edge_neighbors[i],num_edge,num_car_max, cpu = 20) # ID, 邻居edgeID list， 总edge数，最大车辆数
            self.edges[i] = edge_i
        self.edge_index_map = Edge_index_map

        self.time_slots = time_slot  # 强化学习评估时长 （一分钟）

        # 任务建模
        self.task_pool = Task_pool()
        # self.distance_pool = [50,100,150]  # 距离另外随机采样

        # 车辆建模
        self.vehicle_pool = {}   # 系统内当前车辆，vehicle_ID 为 key

    def reset(self, current_vehicles, future_flow):
        # 每次reset都需要加载初始时刻的车辆，进行车辆的建模
        # 每个edge 都重置自身状态
        all_task_num = 0
        # vechicle_pool 重置
        self.vehicle_pool = {}  # 尚未考虑多个连续episode之间vehicle_pool的保留

        for id_i, edge_i in self.edges.items(): # 重置每个edges的建模
            # 计算，带宽，缓存 重置
            edge_i.reset()
            edge_i.update_futureflow(future_flow[self.edge_index_map[id_i]])

        for id_i, edge_i in self.edges.items():  # 根据当前模拟数据，生成每个edge的state
            # 车辆 根据 模拟数据 加入pool，并与Edges连接
            if id_i in current_vehicles:
                vehicle_ids = current_vehicles[id_i][:edge_i.car_max]  # 切片操作 截取 car-max个车参与控制，忽略其余车辆
                for v_id in vehicle_ids:
                    vehicle_i = Vehicle(v_id) # 新建一个车辆
                    # vehicle_i.edge_connection(id_i) # 连接 至 指定edge
                    self.vehicle_pool[v_id] = vehicle_i # 加入系统的车辆pool中
                all_task_num += len(vehicle_ids)
                edge_i.vehicle_connection(vehicle_ids) # ID 记录至对应edge

            # 每个车辆生成任务
            for v_id in edge_i.connected_vehicles:
                v_i = self.vehicle_pool[v_id]
                v_i.generate_task(self.task_pool)

            # Edge State
            # Vehicle information + Task information
            task_state = np.zeros([edge_i.car_max, 6])
            for i, v_id in enumerate(edge_i.connected_vehicles):
                if i >= edge_i.car_max:
                    break
                v_i = self.vehicle_pool[v_id]
                task_state[i] = [v_i.cpu, v_i.buff_CPU, v_i.channel_gain, v_i.current_task.data_size,v_i.current_task.CPU_cycle, v_i.current_task.ddl] # [车辆算力，车辆CPU，车辆信道增益；任务数据量，任务所需CPU，任务截止时间]
            task_state = task_state.flatten()

            # Edge Information
            edge_state = np.array([edge_i.cpu,edge_i.buff_CPU, edge_i.future_flow])

            # Neighboring Edges Information
            neigh_list = []
            for neigh_i in edge_i.neighboring_edges:
                neigh_list.append([self.edges[neigh_i].cpu, self.edges[neigh_i].buff_CPU, self.edges[neigh_i].future_flow])
            neighbor_edge_state = np.array(neigh_list).flatten()

            edge_i.state = np.hstack((task_state, edge_state, neighbor_edge_state)) # 当前state (6*K + 3 + 3*neigh_num)
            edge_i.avaiable_action = edge_i.ava_a()  # 根据当前车辆数，设置可执行action的mask

        self.done = False    # 模拟终止flag
        self.count_step = 0  #
        return all_task_num

    def get_resource(self, action):
        if action // (self.n + 1) == 0:
            return 0
        elif action // (self.n + 1) == 1:
            return 1

    def step(self, actions, next_simulation, next_flow):  # 根据action对各个edge下的每个车 执行任务
        latencys = {}
        wait_latencys = {}
        rewards = {}
        edge_resource_utilizations = []
        vecicle_resource_utilizations = []
        for id_i, action_i in actions.items():
            # action转换成决策数组 K个元素
            edge_i = self.edges[id_i]
            decisions = np.zeros(edge_i.car_max)
            res = action_i
            for k in range(edge_i.car_max):
                decisions[edge_i.car_max-k-1] = res//int(math.pow(2, edge_i.car_max-1-k))
                res = res % int(math.pow(2, edge_i.car_max-1-k))
            # print(id_i,len(edge_i.connected_vehicles))
            # print(decisions)

            # 执行决策
            all_wait_latency_i = 0
            all_latency_i = 0
            all_offload_CPU = 0 # 统一更新edge_buff

            edge_i_vehicle_util = []
            for i in range(len(edge_i.connected_vehicles)):
                vehicle_i = self.vehicle_pool[edge_i.connected_vehicles[i]]
                if i < edge_i.car_max and decisions[i] == 1: # 边缘卸载执行
                    data_rate = edge_i.bw * np.log2(1 + edge_i.tran_grads * 0.1 * vehicle_i.channel_gain / edge_i.noise)
                    tran_latency = vehicle_i.current_task.data_size / data_rate + 0.05  # 传输时延 与任务量有关
                    exe_latency = vehicle_i.current_task.CPU_cycle / edge_i.cpu  # 执行时延 与基站算力有关, 暂时只考虑基站提供稳定的计算能力

                    wait_latency = 0
                    if edge_i.buff_CPU > 0:  # 边缘计算是否等待
                        wait_latency = edge_i.buff_CPU / edge_i.cpu

                    latency = tran_latency + exe_latency + wait_latency
                    # latency = exe_latency + wait_latency
                    # latency = tran_latency

                    # 统计所有需要卸载到edge的任务量
                    all_offload_CPU += vehicle_i.current_task.CPU_cycle

                else: # 本地执行(超出决策范围，或者 决策为本地执行)
                    cpu_cycle = vehicle_i.cpu
                    task_cpu=  vehicle_i.current_task.CPU_cycle
                    calculate_latency = task_cpu / cpu_cycle

                    wait_latency = 0
                    if vehicle_i.buff_CPU > 0: # 本地计算是否等待
                        wait_latency = vehicle_i.buff_CPU / cpu_cycle
                    latency = calculate_latency + wait_latency

                    # 更新车辆buff
                    vehicle_i.buff_CPU += vehicle_i.current_task.CPU_cycle

                # 执行车辆的计算: 无论当前任务是否卸载到车辆执行，每个时刻车辆都要执行计算buff上的任务

                if vehicle_i.buff_CPU < vehicle_i.cpu:
                    vecicle_resource_utilizations.append(vehicle_i.buff_CPU/vehicle_i.cpu)
                    vehicle_i.buff_CPU = 0
                else:
                    vecicle_resource_utilizations.append(1)  # 如果执行之后，vehicle 的buff不为0, 计算资源为1
                    vehicle_i.buff_CPU -= vehicle_i.cpu


                all_latency_i += latency  # i edge 的全部车辆的计算时延
                all_wait_latency_i += wait_latency

            # 更新edge buff
            edge_i.buff_CPU += all_offload_CPU
            if edge_i.buff_CPU < edge_i.cpu:
                edge_i_edge_util = edge_i.buff_CPU/edge_i.cpu
                edge_i.buff_CPU = 0
            else:
                edge_i_edge_util = 1
                edge_i.buff_CPU -= edge_i.cpu

            latencys[id_i] = all_latency_i
            wait_latencys[id_i] = all_wait_latency_i
            rewards[id_i] = all_latency_i * (-1)

            edge_resource_utilizations.append(edge_i_edge_util)

        ### 执行完毕，进入下一时刻, 结合模拟数据调整任务
        # 更新预测结果
        for id_i, edge_i in self.edges.items():  # 更新每个edge的flow prediction
            edge_i.update_futureflow(next_flow[self.edge_index_map[id_i]])

        # 更新车辆和任务信息
        all_task_num = 0
        for id_i, edge_i in self.edges.items():
            # 车辆 根据 模拟数据 设置
            if id_i in next_simulation:
                vehicle_ids = next_simulation[id_i][:edge_i.car_max]  # # 切片操作 截取 car-max个车参与控制，忽略其余车辆
                for v_id in vehicle_ids:
                    if v_id not in self.vehicle_pool: # 首次出现的车辆
                        vehicle_i = Vehicle(v_id)  # 新建一个车辆
                        self.vehicle_pool[v_id] = vehicle_i  # 加入系统的车辆pool中  （尚未考虑清理旧车辆信息）
                all_task_num += len(vehicle_ids)
                edge_i.vehicle_connection(vehicle_ids)  # ID 记录至对应edge

            # 每个连接的车辆 生成任务
            for v_id in edge_i.connected_vehicles:
                v_i = self.vehicle_pool[v_id]
                v_i.generate_task(self.task_pool)

            # 更新下一时刻每个edge的状态
            # Vehicle information + Task information
            task_state = np.zeros([edge_i.car_max, 6])
            for i, v_id in enumerate(edge_i.connected_vehicles):
                if i >= edge_i.car_max:
                    break
                v_i = self.vehicle_pool[v_id]
                task_state[i] = [v_i.cpu, v_i.buff_CPU, v_i.channel_gain, v_i.current_task.data_size,
                                 v_i.current_task.CPU_cycle, v_i.current_task.ddl]  # 六维
            task_state = task_state.flatten()

            # Edge Information
            edge_state = np.array([edge_i.cpu, edge_i.buff_CPU,edge_i.future_flow])

            # Neighboring Edges Information
            neigh_list = []
            for neigh_i in edge_i.neighboring_edges:
                neigh_list.append([self.edges[neigh_i].cpu, self.edges[neigh_i].buff_CPU,self.edges[neigh_i].future_flow])
            neighbor_edge_state = np.array(neigh_list).flatten()

            edge_i.state = np.hstack((task_state, edge_state, neighbor_edge_state))  # 当前state
            edge_i.avaiable_action = edge_i.ava_a()  # 可执行action的mask

        # 返回reward
        self.count_step += 1
        if self.count_step>=self.time_slots:
            self.done = True
        return rewards, wait_latencys, self.done, all_task_num, np.mean(edge_resource_utilizations), np.mean(vecicle_resource_utilizations)

class Multi_Edge_Env_no_neighbor:
    def __init__(self,
                 Edge_IDs,
                 Edge_neighbors,
                 num_edge = 9,
                 num_car_max = 10,
                 time_slot = 60,
                 ):
        # 设置edge
        self.edges = {}
        for i in Edge_IDs:
            edge_i = Edge(i,Edge_neighbors[i],num_edge,num_car_max)
            self.edges[i] = edge_i

        self.time_slots = time_slot  # 强化学习评估时长 （一分钟）

        # 任务建模
        self.task_pool = Task_pool()
        # self.distance_pool = [50,100,150]  # 距离另外随机采样

        # 车辆建模
        self.vehicle_pool = {}

    def reset(self, current_vehicles):
        # 每次reset都需要加载一段时间的车辆轨迹，进行车辆的建模
        # 每个edge 都重置自身状态
        all_task_num = 0
        # vechicle_pool 重置
        self.vehicle_pool = {}

        for id_i, edge_i in self.edges.items():
            # 计算，带宽，缓存 重置
            edge_i.reset()

            # 车辆 根据 模拟数据 设置
            if id_i in current_vehicles:
                vehicle_ids = current_vehicles[id_i]
                for v_id in vehicle_ids:
                    vehicle_i = Vehicle(v_id) # 新建一个车辆
                    # vehicle_i.edge_connection(id_i) # 连接 至 指定edge
                    self.vehicle_pool[v_id] = vehicle_i # 加入系统的车辆pool中
                all_task_num += len(vehicle_ids)
                edge_i.vehicle_connection(vehicle_ids) # ID 记录至对应edge

            # 每个车辆生成任务
            for v_id in edge_i.connected_vehicles:
                v_i = self.vehicle_pool[v_id]
                v_i.generate_task(self.task_pool)

            # Edge State
            # Vehicle information + Task information
            task_state = np.zeros([edge_i.car_max, 6])
            for i, v_id in enumerate(edge_i.connected_vehicles):
                if i >= edge_i.car_max:
                    break
                v_i = self.vehicle_pool[v_id]
                task_state[i] = [v_i.cpu, v_i.buff_CPU, v_i.channel_gain, v_i.current_task.data_size,v_i.current_task.CPU_cycle, v_i.current_task.ddl] # 六维
            task_state = task_state.flatten()

            # Edge Information
            edge_state = np.array([edge_i.cpu,edge_i.buff_CPU])

            edge_i.state = np.hstack((task_state, edge_state)) # 当前state
            edge_i.avaiable_action = edge_i.ava_a()  # 可执行action的mask

        self.done = False  # 移动终止
        self.count_step = 0  #
        return all_task_num

    def get_resource(self, action):
        if action // (self.n + 1) == 0:
            return 0
        elif action // (self.n + 1) == 1:
            return 1

    def step(self, actions, next_simulation):  # 根据action对各个edge下的每个车 执行任务
        latencys = {}
        wait_latencys = {}
        rewards = {}
        for id_i, action_i in actions.items():
            # action转换成决策数组 K个元素
            edge_i = self.edges[id_i]
            decisions = np.zeros(edge_i.car_max)
            res = action_i
            for l in range(edge_i.car_max):
                decisions[edge_i.car_max-l-1] = res//int(math.pow(2, edge_i.car_max-1-l))
                res = res % int(math.pow(2, edge_i.car_max-1-l))

            # print(id_i,len(edge_i.connected_vehicles))
            # print(decisions)

            # 执行决策
            all_wait_latency_i = 0
            all_latency_i = 0
            all_offload_CPU = 0 # 统一更新edge_buff
            for i in range(len(edge_i.connected_vehicles)):
                vehicle_i =  self.vehicle_pool[edge_i.connected_vehicles[i]]
                if i < edge_i.car_max and decisions[i] == 1: # 边缘卸载执行
                    data_rate = edge_i.bw * np.log2(1 + edge_i.tran_grads * 0.1 * vehicle_i.channel_gain / edge_i.noise)
                    tran_latency = vehicle_i.current_task.data_size / data_rate + 0.05  # 传输时延 与任务量有关
                    exe_latency = vehicle_i.current_task.CPU_cycle / edge_i.cpu  # 执行时延 与基站算力有关, 暂时只考虑基站提供稳定的计算能力

                    wait_latency = 0
                    if edge_i.buff_CPU > 0:  # 边缘计算是否等待
                        wait_latency = edge_i.buff_CPU / edge_i.cpu

                    latency = tran_latency + exe_latency + wait_latency
                    # latency = exe_latency + wait_latency
                    # latency = tran_latency
                    all_offload_CPU += vehicle_i.current_task.CPU_cycle

                else: # 本地执行(超出决策范围，或者 决策为本地执行)
                    cpu_cycle = vehicle_i.cpu
                    task_cpu=  vehicle_i.current_task.CPU_cycle # 第三位是cpu频率
                    calculate_latency = task_cpu / cpu_cycle

                    wait_latency = 0
                    if vehicle_i.buff_CPU > 0: # 本地计算是否等待
                        wait_latency = vehicle_i.buff_CPU / cpu_cycle
                    latency = calculate_latency + wait_latency


                    # 更新车辆buff
                    vehicle_i.buff_CPU += task_cpu
                    vehicle_i.buff_CPU -= cpu_cycle
                    if vehicle_i.buff_CPU < 0:
                        vehicle_i.buff_CPU = 0

                all_latency_i += latency  # i edge 的全部车辆的计算时延
                all_wait_latency_i += wait_latency
            # 更新edge buff
            edge_i.buff_CPU += all_offload_CPU
            edge_i.buff_CPU -= edge_i.cpu
            if edge_i.buff_CPU  < 0:
                edge_i.buff_CPU = 0
            latencys[id_i] = all_latency_i
            wait_latencys[id_i] = all_wait_latency_i
            rewards[id_i] = all_latency_i * (-1)

        # 执行完毕，进入下一时刻, 结合模拟数据调整任务
        all_task_num = 0
        for id_i, edge_i in self.edges.items():
            # 车辆 根据 模拟数据 设置
            if id_i in next_simulation:
                vehicle_ids = next_simulation[id_i]
                for v_id in vehicle_ids:
                    if v_id not in self.vehicle_pool: # 首次出现的车辆
                        vehicle_i = Vehicle(v_id)  # 新建一个车辆

                        self.vehicle_pool[v_id] = vehicle_i  # 加入系统的车辆pool中  （尚未考虑清理旧车辆信息）
                all_task_num += len(vehicle_ids)
                edge_i.vehicle_connection(vehicle_ids)  # ID 记录至对应edge

            # 每个连接的车辆 生成任务
            for v_id in edge_i.connected_vehicles:
                v_i = self.vehicle_pool[v_id]
                v_i.generate_task(self.task_pool)

            # 更新下一时刻每个edge的状态
            # Edge State
            # Vehicle information + Task information
            task_state = np.zeros([edge_i.car_max, 6])
            for i, v_id in enumerate(edge_i.connected_vehicles):
                if i >= edge_i.car_max:
                    break
                v_i = self.vehicle_pool[v_id]
                task_state[i] = [v_i.cpu, v_i.buff_CPU, v_i.channel_gain, v_i.current_task.data_size,
                                 v_i.current_task.CPU_cycle, v_i.current_task.ddl]  # 六维
            task_state = task_state.flatten()

            # Edge Information
            edge_state = np.array([edge_i.cpu, edge_i.buff_CPU])

            edge_i.state = np.hstack((task_state, edge_state))  # 当前state
            edge_i.avaiable_action = edge_i.ava_a()  # 可执行action的mask

        # 返回reward
        self.count_step += 1
        if self.count_step>=self.time_slots:
            self.done = True
        return rewards,wait_latencys, self.done, all_task_num

