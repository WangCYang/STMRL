# we need to import python modules from the $SUMO_HOME/tools directory
import os
import sys
import time
import pandas as pd

def check_and_create_path(filename):
    file_dir = os.path.split(filename)[0]
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)

Simluation_name = "net4"
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa

show_gui = True
if not show_gui:
    sumoBinary = checkBinary("sumo")
else:
    sumoBinary = checkBinary("sumo-gui")

### 选择模拟场景
## Simulate_scenario = "Net4"
# sumocfgfile = "D:\\SUMO\\workspace\\Simulate_roadnet_v1\\net4.sumocfg"
# sumocfgfile = "D:\\SUMO\\workspace\\Simulate_roadnet_v1\\net4_test_3.sumocfg"
# sumocfgfile = "D:\\SUMO\\workspace\\Simulate_roadnet_v1\\net4_test_2.sumocfg"
# sumocfgfile = "D:\\SUMO\\workspace\\Simulate_roadnet_v1\\net4_test_3.sumocfg"
# scenario = "Net4"
## Simulate_scenario = "bologna_pasubio"
sumocfgfile = "D:\\SUMO\\workspace\\bologna_simulation\\pasubio\\pasubio_training_24h.sumocfg"
traci.start([sumoBinary, "-c", sumocfgfile])
scenario = "bologna_pasubio"

'''
### Step 1: 查询路网信息
'''
# num_all_junction = traci.junction.getIDCount()  # 交叉路口数量
# junction_list = traci.junction.getIDList() # 交叉路口ID
# print("all_junction_num:", num_all_junction) # 与图上找到的不一致
# print("all_junctions:", junction_list)

'''
### Step 2: 保存edge server位置数据
'''
### 根据路网筛选center junctions 作为server list
## Simulate_scenario = "Net4"
# center_junction_ids = ['J11','J12','J0', 'J2', 'J3','J17', 'J6', 'J8', 'J9', 'J14', 'J7','J10', 'J15']
# other_junction_ids = ['J22', 'J21', 'J20', 'J13', 'J4', 'J23', 'J26', 'J18', 'J19']
## Simulate_scenario = "bologna_pasubio"
# center_junction_ids = ['4','7','12', '19', '18','0', '9', '27', '23', '2', '29','33', '32', '1', '15','40','39','36']  # 18 servres
#
# saved_edge_position_file = 'data/{}/edge_position.csv'.format(scenario)
# check_and_create_path(saved_edge_position_file)
# junction_pos_list = [traci.junction.getPosition(i) for i in center_junction_ids]  # edge节点的位置
# junction_pos_x_list = [i[0] for i in junction_pos_list]
# junction_pos_y_list = [i[1] for i in junction_pos_list]
# data = pd.DataFrame(columns=["edge_id","edge_pos_x", "edge_pos_y"],data = [edge_i for edge_i in zip(center_junction_ids, junction_pos_x_list, junction_pos_y_list)])
# data.to_csv(saved_edge_position_file, index=False)
# print(data)

'''
### Step 3: 运行模拟，生成车辆轨迹数据
'''
saved_dir = "24h" # 24小时
simulation_steps = 86400 # 24小时

# saved_dir = "1h" # 24小时
# # simulation_steps = 3600

saved_vehicle_file = 'data/{}/{}/vehicle_simulation.csv'.format(scenario,saved_dir)
check_and_create_path(saved_vehicle_file)
vehicle_pos_data = pd.DataFrame(columns=["time","vehicle_id", "pos_x", "pos_y"],data = [])
vehicle_pos_data.to_csv(saved_vehicle_file, index=False)
for step in range(0, simulation_steps):
    # time.sleep() # 相当于延迟1s
    traci.simulationStep(step + 1)
    simulation_current_time = traci.simulation.getTime() # 以秒为单位
    # print("当前仿真时间", simulation_current_time)

    # 当前所有车辆ID 及 车辆位置
    current_vehicle_id_list = traci.vehicle.getIDList()
    current_vehicle_pos_list = [[simulation_current_time, i, traci.vehicle.getPosition(i)[0], traci.vehicle.getPosition(i)[1]] for i in current_vehicle_id_list]

    # 保存轨迹数据 （时间，ID, 位置_x, 位置_y）
    current_vehicle_pos_data = pd.DataFrame(current_vehicle_pos_list)
    current_vehicle_pos_data.to_csv(saved_vehicle_file,mode='a',index=False,header=False)
traci.close()

'''
### Step 4: 设置server的neighbors信息，转road_net.py文件
'''
'''
### Step 5: 根据每个车和每个server的距离，分配vehicle-server的connection关系，转traffic_analysis.py文件
'''