import os
import sys
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
from stellargraph.layer import GCN_LSTM
import stellargraph as sg

### GCN_LSTM 输入要求：N*T的特征矩阵，N*N的邻接矩阵

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
print("stellargraph version:",sg.__version__)
print("Numpy version:",np.__version__)
print("version:",tf.__version__)
#
# dataset = sg.datasets.METR_LA()
# speed_data, sensor_dist_adj = dataset.load()
# print(sensor_dist_adj.shape)
# print(sensor_dist_adj)
# num_nodes, time_len = speed_data.shape

# print("No. of sensors:", num_nodes, "\nNo of timesteps:", time_len)
# print(speed_data.head())
# print(tf.python.client.device_lib.list_local_devices())
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# print(gpus)
# tf.config.experimental.set_visible_devices(devices=gpus[0],device_type="GPU")
# tf.config.experimental.set_memory_growth(gpus[0], True)

def train_test_split(data, train_portion):
    time_len = data.shape[1]
    train_size = int(time_len * train_portion)
    train_data = np.array(data.iloc[:, :train_size])
    test_data = np.array(data.iloc[:, train_size:])
    return train_data, test_data

dataset_dir = "flow_data/"

'''
    Step 1：场景选择
'''
# Simulation_scenario = "3-3-grid-10h"

# Simulation_scenario = "Net4"
# Time_range = "24h"

Simulation_scenario = "bologna_pasubio"
Time_range = "24h"

train_data = np.load(os.path.join(dataset_dir, "{}/{}/flow_graph.npz").format(Simulation_scenario,Time_range), allow_pickle=True)
test_data = np.load(os.path.join(dataset_dir, "{}/{}/flow_graph.npz").format(Simulation_scenario,Time_range), allow_pickle=True)
adj_matrix = np.load(os.path.join(dataset_dir,"{}/adjacent_matrix.npy").format(Simulation_scenario))

time_len = train_data["input"].shape[0]
train_X = train_data["input"][:int(time_len*0.9)] # 90% 训练, 10%测试
train_Y = train_data["label"][:int(time_len*0.9)]
test_X = test_data["input"][int(time_len*0.9):]
test_Y= test_data["label"][int(time_len*0.9):]

print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)

def  scale_value(data):
    max = 0
    min = 100000
    for data_i in data:
        if data_i.max()>max:
            max = data_i.max()
        if data_i.min()<min:
            min = data_i.min()
    return max, min

scale_max, scale_min = scale_value([train_X,train_Y])
print(scale_max, scale_min)

def scale_data(data, scale_max, scale_min):
    scaled_data = (data - scale_min) / (scale_max - scale_min)
    return scaled_data

def rescale_data(data, scale_max, scale_min):
    rescaled_data = (data * (scale_max - scale_min)) + scale_min
    return rescaled_data

train_X_scaled= scale_data(train_X, scale_max,scale_min)
train_Y_scaled= scale_data(train_Y, scale_max,scale_min)
test_X_scaled= scale_data(test_X, scale_max,scale_min)
test_Y_scaled= scale_data(test_Y, scale_max,scale_min)

# seq_len = 10
# # pre_len = 12
# # def sequence_data_preparation(seq_len, pre_len, train_data, test_data):
# #     trainX, trainY, testX, testY = [], [], [], []
# #
# #     for i in range(train_data.shape[1] - int(seq_len + pre_len - 1)):
# #         a = train_data[:, i : i + seq_len + pre_len]
# #         trainX.append(a[:, :seq_len])
# #         trainY.append(a[:, -1])
# #
# #     for i in range(test_data.shape[1] - int(seq_len + pre_len - 1)):
# #         b = test_data[:, i : i + seq_len + pre_len]
# #         testX.append(b[:, :seq_len])
# #         testY.append(b[:, -1])
# #
# #     trainX = np.array(trainX)
# #     trainY = np.array(trainY)
# #     testX = np.array(testX)
# #     testY = np.array(testY)
# #
# #     return trainX, trainY, testX, testY
# #
# # trainX, trainY, testX, testY = sequence_data_preparation(
# #     seq_len, pre_len, train_scaled, test_scaled
# # )

# 加载模型
gcn_lstm = GCN_LSTM(
    seq_len=15,
    adj=adj_matrix,
    gc_layer_sizes=[16, 10],
    gc_activations=["relu", "relu"],
    lstm_layer_sizes=[200, 200],
    lstm_activations=("tanh", "tanh"),
)

x_input, x_output = gcn_lstm.in_out_tensors()
model = Model(inputs=x_input, outputs=x_output)
print(x_input.shape, x_output.shape)

# Train: 模型训练 并 保存参数
model.compile(optimizer="adam", loss="mse", metrics=["mae",tf.keras.metrics.RootMeanSquaredError()])
history = model.fit(
    train_X_scaled,
    train_Y_scaled,
    epochs=20,
    batch_size=60,
    shuffle=True,
    # verbose=0,  # 指定为0，不展示输出结果
    validation_data=(test_X_scaled, test_Y_scaled),
)
print(
    "Train loss: ",
    history.history["loss"][-1],
    "\nTest loss:",
    history.history["val_loss"][-1],
)
sg.utils.plot_history(history)
# model.save_weights("saved_models/GCN_LSTM/pasubio/checkpoints_1/".format(Simulation_scenario))  # 只保存参数
# 定义保存权重的路径
weights_path = 'saved_models/GCN_LSTM/{}/checkpoints/'.format(Simulation_scenario)
# 检查路径是否存在，如果不存在则创建路径
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
model.save_weights(weights_path+'model_weights.h5')  # 只保存参数

#  Test: 加载模型参数
# model.load_weights("saved_models/GCN_LSTM/{}/checkpoints/".format(Simulation_scenario)+'model_weights.h5')
model.summary()

ythat = model.predict(train_X_scaled)
yhat = model.predict(test_X_scaled)

## actual train and test values
train_rescref = rescale_data(train_Y_scaled, scale_max, scale_min)
test_rescref = rescale_data(test_Y_scaled, scale_max, scale_min)

## Rescale model predicted values
train_rescpred = rescale_data(ythat, scale_max, scale_min)
test_rescpred = rescale_data(yhat, scale_max, scale_min)

# metrics
test_MAE = np.sum(np.abs(test_rescref - test_rescpred)) / np.size(test_rescpred)
test_RMSE = np.sqrt(np.sum(np.square(test_rescref - test_rescpred)) / np.size(test_rescpred))

train_MAE = np.sum(np.abs(train_rescref - train_rescpred)) / np.size(train_rescpred)
train_RMSE = np.sqrt(np.sum(np.square(train_rescref - train_rescpred)) / np.size(train_rescpred))

print("----------------------result-----------------------")
print("test_MAE={:.3f},test_RMSE={:.3f},train_MAE={:.3f},train_RMSE={:.3f}".format(test_MAE, test_RMSE, train_MAE, train_RMSE))

##all test result visualization
fig1 = plt.figure(figsize=(15, 8))
#    ax1 = fig1.add_subplot(1,1,1)
a_pred = test_rescpred[:, 0]
a_true = test_rescref[:, 0]
plt.plot(a_pred, "r-", label="prediction")
plt.plot(a_true, "b-", label="true")
plt.xlabel("time")
plt.ylabel("speed")
plt.legend(loc="best", fontsize=20)
plt.show()