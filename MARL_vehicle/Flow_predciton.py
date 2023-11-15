import os
import numpy as np
from stellargraph.layer import GCN_LSTM
import stellargraph as sg

### GCN_LSTM 输入要求：N*T的特征矩阵，N*N的邻接矩阵
import tensorflow as tf
from tensorflow.keras import Model

# print("stellargraph version:",sg.__version__)
# print("Numpy version:",np.__version__)
# print("version:",tf.__version__)

class GCRNN:
    def __init__(self,gc_layer_sizes,lstm_layer_sizes,adj_matrix,history_time,scale_max,scale_min):
        self.gcn_lstm = GCN_LSTM(
            seq_len=history_time,
            adj=adj_matrix,
            gc_layer_sizes=gc_layer_sizes,
            gc_activations=["relu", "relu"],
            lstm_layer_sizes=lstm_layer_sizes,
            lstm_activations=("tanh", "tanh"),
        )
        x_input, x_output = self.gcn_lstm.in_out_tensors()
        self.model = Model(inputs=x_input, outputs=x_output)
        self.model.summary()

        self.scale_max = scale_max
        self.scale_min = scale_min

    def load_weights(self,weight_file):
        self.model.load_weights(weight_file+'model_weights.h5')  # Net4 和 3*3需要修改模型路径

    def scale(self, data):
        scaled_data = (data - self.scale_min) / (self.scale_max - self.scale_min)
        return scaled_data

    def rescale(self, data):
        rescaled_data = (data * (self.scale_max - self.scale_min)) + self.scale_min
        return rescaled_data

    def predict(self,input):
        scaled_input = self.scale(input)
        out_pre = self.model.predict(scaled_input)
        rescaled_output = self.rescale(out_pre)
        return rescaled_output

class Flow_Loader:
    def __init__(self,flow_data_file):
        self.history_data = np.load(flow_data_file, allow_pickle=True)

    def query_history(self, current_time, time_len):
        flag = 0
        if (current_time - time_len) < 0:
            data = np.zeros(self.history_data[:time_len].shape).transpose((1,0))  # 无法预测，flag返回0
        else:
            flag = 1 # 数据有效
            data = np.array(self.history_data[int(current_time-time_len):int(current_time)]).transpose((1,0))
        return flag, data
