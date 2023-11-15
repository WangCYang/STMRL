import numpy as np
import os
import pickle
import time
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

dataset_dir = "flow_data/"
train_data = np.load(os.path.join(dataset_dir, "flow_data/3-3-grid-10h/flow_sequence.npz"), allow_pickle=True)
test_data = np.load(os.path.join(dataset_dir,"3-3-grid-1h/flow_sequence.npz"),allow_pickle=True)

train_input = train_data["input"]
train_label = train_data["label"]
test_input = test_data["input"]
test_label = test_data["label"]

print(train_input.shape)
print(train_label.shape)
print(test_input.shape)
print(test_label.shape)

train_input_1 = np.transpose(train_input, [0, 2, 1])
test_input_1 = np.transpose(test_input, [0, 2, 1])

train_input_linear = np.reshape(train_input_1, [-1, train_input_1.shape[2]])
train_label_linear = np.reshape(train_label, [-1, 1])
test_input_linear = np.reshape(test_input_1, [-1, test_input_1.shape[2]])
test_label_linear = np.reshape(test_label, [-1, 1])

print("input", train_input_linear.shape)
print("label", train_label_linear.shape)
print("input", test_input_linear.shape)
print("label", test_label_linear.shape)

# HA
# test_input_pm=test_input_1[...,0]
# print(test_input_pm.shape)
# test_input_HA = np.reshape(test_input_pm, [-1, test_input_pm.shape[-1]])
# print(test_input_HA.shape)
# train_pre=np.mean(train_input_linear,axis=1)
# test_pre=np.mean(test_input_linear,axis=1)

# Linear Regression
# model = MultiOutputRegressor(LinearRegression())  # MultiOutputRegressor() 利用相同的feature进行独立的多目标回归
# model = LinearRegression()

# Ridge
# ridge=Ridge(alpha = 100)
# model = MultiOutputRegressor(ridge)
# model = ridge

# Lasso
#model = MultiOutputRegressor(Lasso(alpha = 0.001))

# SVM
# model = MultiOutputRegressor(SVR(kernel='rbf',gamma= "auto"))
model = SVR(kernel='rbf',gamma= "auto")

# MLP
# model = MLPRegressor(
#     hidden_layer_sizes=(64,64,16),  activation='relu', solver='adam', alpha=0.0001, batch_size=256,
#     learning_rate='adaptive', learning_rate_init=0.01, power_t=0.5, max_iter=100, shuffle=True,
#     tol=0.001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
#     early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08,n_iter_no_change=200)

# GBDT
# model=MultiOutputRegressor(GradientBoostingRegressor(
#     loss='ls',learning_rate=0.1, n_estimators=100, subsample=1, min_samples_split=2, min_samples_leaf=1, max_depth=5))

print("----------------------start traning----------------------")
start_time = time.time()  # 计算训练时间
model.fit(train_input_linear, train_label_linear)
end_time = time.time()
print("training_time: ", end_time - start_time)


train_pre = model.predict(train_input_linear)
test_pre = model.predict(test_input_linear)
print("predictions:",train_pre.shape)
print("predictions:",train_pre.shape)

train_scaler_f = open('flow_data/3-3-grid-10h/standard.pkl', 'rb')
train_scaler = pickle.load(train_scaler_f)
print("train_scaler:{}_{}".format(train_scaler.mean_, train_scaler.var_))

test_scaler_f = open('flow_data/3-3-grid-1h/standard.pkl', 'rb')
test_scaler = pickle.load(test_scaler_f)
print("test_scaler:{}_{}".format(test_scaler.mean_, test_scaler.var_))

test_pred_all = test_scaler.inverse_transform(test_pre.flatten())
test_truth_all = test_scaler.inverse_transform(test_label_linear.flatten())
test_MAE = np.sum(np.abs(test_truth_all - test_pred_all)) / np.size(test_truth_all)
test_RMSE = np.sqrt(np.sum(np.square(test_truth_all - test_pred_all)) / np.size(test_truth_all))

train_pred_all = train_scaler.inverse_transform(train_pre.flatten())
train_truth_all = train_scaler.inverse_transform(train_label_linear.flatten())
train_MAE = np.sum(np.abs(train_truth_all - train_pred_all)) / np.size(train_truth_all)
train_RMSE = np.sqrt(np.sum(np.square(train_truth_all - train_pred_all)) / np.size(train_truth_all))

print("----------------------result-----------------------")
print("test_MAE={:.3f},test_RMSE={:.3f},train_MAE={:.3f},train_RMSE={:.3f}".format(
    test_MAE, test_RMSE, train_MAE, train_RMSE))
# print("test_MAE={:.3f},test_RMSE={:.3f}".format(test_MAE, test_RMSE))
# output_len = test_pre.shape[1]
# for i in range(output_len):
#     test_pred_i = scaler.inverse_transform(test_pre[:, i].flatten())
#     test_truth_i = scaler.inverse_transform(test_label_linear[:, i].flatten())
#     test_MAE_i = np.sum(np.abs(test_truth_i - test_pred_i)) / np.size(test_truth_i)
#     test_RMSE_i = np.sqrt(np.sum(np.square(test_truth_i - test_pred_i)) / np.size(test_truth_i))
#     print("Horizon {:.3f}, MAE={:.3f},RMSE={:.3f}".format(i, test_MAE_i, test_RMSE_i))
