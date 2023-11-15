# -*- coding: utf-8 -*-
import numpy as np
# import random
import tensorflow as tf
from tensorflow.keras import Model
from utils import check_and_create_path

class MyNN(Model):
    def __init__(self, s_dim, a_dim, hidden_nodes1, hidden_nodes2):
        super().__init__()
        tf.random.set_seed(3)
        self.d1 = tf.keras.layers.Dense(hidden_nodes1, 'relu') # w:[s_dim,hidden_nodes1], b:[hidden_nodes1,]
        self.d2 = tf.keras.layers.Dense(hidden_nodes2, 'relu')
        self.d3 = tf.keras.layers.Dense(a_dim)  # Q值不需要激活函数， [-inf, inf]
        self(tf.keras.Input(shape=s_dim))   # 初始化参数 [B, s_dim]

    def call(self, s):
        x = self.d1(s)
        x = self.d2(x)
        q = self.d3(x)
        return q    # [B, A]

class RL_agent_DDQN:
    def __init__(self,
            state_dim,  #
            action_dim,
            num_hidden1,
            num_hidden2,
            discount_factor,
            memory_size,
            replace_target_iter,
            learning_rate,
            batch_size,
            saved_version="no_save",
            flag =''):

        #basic information（消融1）
        self.s_dim = state_dim  # 3*K+ 3 + N*3   K个车辆的状态（任务量，计算需求，计算能力，通信状态） + edge buffer + 邻居节点的状态（(任务总量, 算力，预测)）
        self.a_dim = action_dim # 2^K K个车辆的（0-1决策）
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2
        self.gamma = discount_factor

        #experience
        self.memory_size = memory_size  # buffer size
        self.memory_counter = 0
        self.memory_now_size = 0
        self.memory = np.zeros((self.memory_size, self.s_dim*2+2+self.a_dim))  # s:S, s':S, a:1, r:1, ava:A  （s, a, r, s+1, ）

        #neural network
        self.q_net = MyNN(self.s_dim, self.a_dim, self.num_hidden1, self.num_hidden2)  # Q-net: 输入state，输出action
        self.ck_point = tf.train.Checkpoint(policy=self.q_net)
        if flag!="no_save":
            saved_directory= 'models_DDQN/{}/edge{}'.format(saved_version,flag)   # 在本地无法正常保存，server上可以
            check_and_create_path(saved_directory)
            self.saver = tf.train.CheckpointManager(self.ck_point, directory=saved_directory, max_to_keep=5, checkpoint_name='wcy_model')

        self.q_target_net = MyNN(self.s_dim, self.a_dim, self.num_hidden1, self.num_hidden2)     # 同样结构的Q-net作为target net （DDQN的结构）
        tf.group([t.assign(s) for t, s in zip(self.q_target_net.weights,self.q_net.weights)])   #  组合多个操作

        self.replace_target_iter = replace_target_iter
        self.train_step_counter = 0
        self.batch_size = batch_size
        self.learning_rate = learning_rate  # 不算小
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    def save_model(self, episode):
        self.saver.save(checkpoint_number=episode)  # 记录Q-net 网络参数

    def restore_model(self, directory):
        print(directory)
        self.ck_point.restore(tf.train.latest_checkpoint(directory))
        tf.group([t.assign(s) for t, s in zip(self.q_target_net.weights, self.q_net.weights)]) 
    
    def get_action(self, action_value, ava_action):
        av_np = action_value.numpy()  # 神经网络输出的Q值 [B, A]
        av_np_min = av_np.min(axis = 1, keepdims = True)
        av_np -= av_np_min  # 按行减去相应每行的最小值，数据全部大于0 => [B, A] - [B, 1] = [B, A], 每行减去每行的最小值
        ava_action_value = np.multiply(av_np, ava_action)    # [B, A] * [B, A] = [B, A]
        action = np.argmax(ava_action_value, axis = 1)   # [B, A] => [B,]  # 取最大值所在的位置，即base_station
        return action

    def choose_action(self, state, ava_action, epsilon):
        '''
        state: List
        '''    
        s = np.array(state).reshape((1,len(state)))           # [B=1, S]
        action_value = self.q_net(s)                          # 动作价值 Q=[B=1, A]  输入状态，得到价值
        action = self.get_action(action_value, ava_action)[0]  # 贪心选择Q值最大的动作
        if np.random.uniform() < epsilon:                     # 探索
            # print(np.random.uniform())
            # input()
            a = list(range(self.a_dim))                       # 生成所有动作的索引值 a = [0~51]
            ava = list(map(lambda x, y:x*y, a, ava_action))   # 生成可选动作 len(ava)=52
            c = []                                            # 存储可以随机选择的动作
            c.append(0)                                       # 先把0加上,因为0为本地
            for i in ava:
                if i != 0:
                    c.append(i)                               # 把所有可选动作加上           
            action = np.random.choice(c) # 如果探索，则随机选择一个动作

        return action

    def store_transition(self, s, a, s_, ava_action, r):
        transition = np.hstack((s, a, r, ava_action, s_)) # S, 1, 1, 1, A
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_now_size = min(self.memory_size, self.memory_now_size+1)
        self.memory_counter += 1

    def learn(self):
        if self.train_step_counter % self.replace_target_iter == 0:  # 训练200次 online-net，替换一次target net
            tf.group([t.assign(s) for t, s in zip(self.q_target_net.weights,self.q_net.weights)])    # 替换target网络参数
        if self.memory_now_size <= self.batch_size:  # 这边有小错误 memory 不足
            sample_index = np.arange(self.memory_now_size)   # 则分配全部memory出来
        else:  #否则采样batch-size个样本
            sample_index = np.random.choice(self.memory_size, size = self.batch_size)   #随机采样几条经验的索引值
        batch_memory = self.memory[sample_index, :]#将采样到的经验放到batch_memory中

        mem_s = batch_memory[:, :self.s_dim]     #当前状态
        mem_s_= batch_memory[:, -self.s_dim:]    #下一个状态
        mem_action= batch_memory[:, self.s_dim].astype(int)  #获取到所有动作
        mem_reward = batch_memory[:, self.s_dim+1]    #获取到所有奖励
        mem_avaAction = batch_memory[:, self.s_dim+2:-self.s_dim]   #获得所有可执行动作
        q_loss = self.train(mem_s, mem_s_, mem_action, mem_reward, mem_avaAction)

    def train(self, s, s_, a, r, ava_a):
        '''
        s: [B, S]
        s_: [B, S]
        a: [B, ]
        r: [B, ]
        ava_a: [B, A]
        '''
        with tf.GradientTape() as tape:
            q_eval_current = self.q_net(s)              # [B, A]   # 真实Q值
            q_eval_next = self.q_net(s_)                # [B, A]   # 执行之后的Q值
            q_target_next = self.q_target_net(s_)        # [B, A]  # target net的执行后的Q值
       
            max_AevalNext = self.get_action(q_eval_next, ava_a) #获得q_eval中的下一个状态的最大动作的索引    # [B, ]   # 新状态下Q值最大的动作
            max_AevalNext_onehot = tf.one_hot(max_AevalNext, self.a_dim)   # [B, A]
            value_AtargeNext = tf.reduce_sum(tf.multiply(q_target_next, max_AevalNext_onehot), axis=1) #获得q_target中的根据q_net选出的最大动作在target网络相应的值    [B, A] => [B, ] target 网络中，该动作的Q值
            q_target = r + self.gamma * value_AtargeNext   #计算出target值q(s_,a)=r+gamma*q(s_,a_)  [B, ] # Bellman方程动态规划：加上即时奖励后，以及之后步动作的Q-值，就当前该动作的Q值

            mem_action_onehot = tf.one_hot(a, self.a_dim) # [B, ] => [B, A]
            q_eval = tf.reduce_sum(tf.multiply(q_eval_current,mem_action_onehot), axis=1)    #计算出q(s,a) [B, A] => [B, ]  # 当前动作的Q值
            td_error = q_eval - q_target  # 希望这两个Q值相等，即模型对Q值的拟合稳定
            q_loss=tf.reduce_mean(tf.square(td_error))  # [B, ] => 1,   # MSE计算loss
        grads=tape.gradient(q_loss, self.q_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))

        self.train_step_counter += 1
        return q_loss
    
    # nohup python run.py 2>&1 &
    # [2477] 
    # kill 2477
    # top
    # conda activate tf2
    # cd dragon
