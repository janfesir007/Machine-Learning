# -*-coding:gbk-*-
# -*-coding:utf-8-*-
# 该tensorflow的版本是0.10
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time
import config as conf  # 引入另一个config.py文件
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn as tflearn
from tensorflow.contrib import layers as tflayers
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


def Divide_data(data_file):
    '''
    Divide_data(): 将数据分割成“训练集/验证集/测试集”三部分数据,列表类型
    如果划分文件已存在则加载,否则进行文件的划分并将划分后的文件进行保存.
    Returns:
    - a list of arrays of close-to-close percentage returns, normalized by running
      stdev calculated over last c.normalize_std_len days
    '''
    def divide_data():
        # find date range for the split train, val, test (0.8, 0.1, 0.1 of total days)
        # split = [0.8, 0.1, 0.1]
        # 读入原始数据,保存在listing_data：DataFrame类型,索引是默认的“int”类型
        listing_data = pd.read_csv('data/flow_year.csv', usecols=['Hour', 'Flow (Veh/Hour)', "# Lane Points", "% Observed"])  # 读出后已去掉第一行字段
        seq = [[], [], []]
        lastAc = -1
        for i in range(int(len(listing_data)*0.8)):
            try:
                ac = listing_data.ix[i]['Flow (Veh/Hour)']  # listing_data:数据源,由.csv文件读入,DateFrame类型
                daily_return = (ac - lastAc)  # 一次一阶差分
                # daily_return = (ac - lastAc) / lastAc  # 一次一阶差分标准化;  还原公式：ac = lastAc*daily_return+lastAc
                # if len(daily_returns) == daily_returns.maxlen:
                #     seq[idx].append(daily_return / np.std(daily_returns))  # 一次一阶差分标准化后,再进行数据长度为50的标准差归一化
                # daily_returns.append(daily_return)
                lastAc = ac
                seq[0].append(daily_return)  # 一次一阶差分数据（原始不经过处理的数据效果很差）
                # seq[0].append(ac)  # 原始数据
            except KeyError:
                pass
        for i in range(int(len(listing_data)*0.1)):
            try:
                ac = listing_data.ix[i+4800]['Flow (Veh/Hour)']  # listing_data:数据源,由.csv文件读入,DateFrame类型
                daily_return = (ac - lastAc)  # 一次一阶差分
                # daily_return = (ac - lastAc) / lastAc  # 一次一阶差分标准化
                # if len(daily_returns) == daily_returns.maxlen:
                #     seq[idx].append(daily_return / np.std(daily_returns))  # 一次一阶差分标准化后,再进行数据长度为50的标准差归一化
                # daily_returns.append(daily_return)
                lastAc = ac
                seq[1].append(daily_return)  # 一次一阶差分数据
                # seq[1].append(ac)  # 原始数据
            except KeyError:
                pass
        for i in range(int(len(listing_data)*0.1)):
            try:
                ac = listing_data.ix[i+5400]['Flow (Veh/Hour)']  # listing_data:数据源,由.csv文件读入,DateFrame类型
                daily_return = (ac - lastAc)  # 一次一阶差分
                # daily_return = (ac - lastAc) / lastAc  # 一次一阶差分标准化
                # if len(daily_returns) == daily_returns.maxlen:
                #     seq[idx].append(daily_return / np.std(daily_returns))  # 一次一阶差分标准化后,再进行数据长度为50的标准差归一化
                # daily_returns.append(daily_return)
                lastAc = ac
                seq[2].append(daily_return)  # 一次一阶差分数据
                # seq[2].append(ac)  # 原始数据
            except KeyError:
                pass
        return [np.asarray(dat, dtype=np.float32) for dat in seq]

    if not os.path.exists(data_file):
        datasets = divide_data()
        print('Saving in {}'.format(file))
        np.savez(file, *datasets)
    else:
        with np.load(data_file) as file_load:
            datasets = [file_load['arr_%d' % i] for i in range(len(file_load.files))]
    return datasets


def rnn_data(data, time_steps, labels=False):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]]
        -> labels == True [3, 4, 5]
    """
    rnn_df = []
    for i in range(len(data) - time_steps):
        if labels:  # labels==True:表示在划分y,每一个序列(长度为time_steps)对应(映射/预测)一个y
            try:
                rnn_df.append(data[i + time_steps])
            except AttributeError:
                rnn_df.append(data[i + time_steps])
        else:  # labels==False:表示在划分X(序列),有重复的划分（隔一位继续划分）
            data_ = data[i: i + time_steps]
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

    return np.array(rnn_df, dtype=np.float32)



def prepare_data(time_steps, data_file, labels=False):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell.
    """
    df_train, df_val, df_test = Divide_data(data_file)  # 分割数据集:以一维数组(narray)形式存放
    return (rnn_data(df_train, time_steps, labels=labels),
            rnn_data(df_val, time_steps, labels=labels),
            rnn_data(df_test, time_steps, labels=labels))


def generate_data(time_steps, data_file):
    train_x, val_x, test_x = prepare_data(time_steps, data_file)
    train_y, val_y, test_y = prepare_data(time_steps, data_file, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)


def lstm_model(time_steps, rnn_layers, dense_layers=None, learning_rate=0.1, optimizer='Adagrad'):
    """
    Creates a deep model based on:
        * stacked lstm cells
        * an optional dense layers
    :param time_steps: the size of the cells.
    :param rnn_layers: list of int or dict
                         * list of int: the steps used to instantiate the `BasicLSTMCell` cell
                         * list of dict: [{steps: int, keep_prob: int}, ...]
    :param dense_layers: list of nodes for each layer
    :return: the model definition
    """

    def lstm_cells(layers):
        if isinstance(layers[0], dict):
            return [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(layer['num_units'],
                                                                               state_is_tuple=True),
                                                  layer['keep_prob'])
                    if layer.get('keep_prob') else tf.nn.rnn_cell.BasicLSTMCell(layer['num_units'],
                                                                                state_is_tuple=True)
                    for layer in layers]
        return [tf.nn.rnn_cell.BasicLSTMCell(steps, state_is_tuple=True) for steps in layers]

    def dnn_layers(input_layers, layers):
        if layers and isinstance(layers, dict):
            return tflayers.stack(input_layers, tflayers.fully_connected,
                                  layers['layers'],
                                  activation=layers.get('activation'),
                                  dropout=layers.get('dropout'))
        elif layers:
            return tflayers.stack(input_layers, tflayers.fully_connected, layers)
        else:
            return input_layers

    def _lstm_model(X, y):
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells(rnn_layers), state_is_tuple=True)
        x_ = tf.unpack(X, axis=1, num=time_steps)
        output, layers = tf.nn.rnn(stacked_lstm, x_, dtype=dtypes.float32)
        output = dnn_layers(output[-1], dense_layers)
        prediction, loss = tflearn.models.linear_regression(output, y)
        train_op = tf.contrib.layers.optimize_loss(
            loss, tf.contrib.framework.get_global_step(), optimizer=optimizer,
            learning_rate=learning_rate)
        return prediction, loss, train_op

    return _lstm_model


start_time = time.time()
LOG_DIR = './ops_logs/traffic_flow1'  # 保存训练好得到的模型参数和图表等
TIMESTEPS = 4
# RNN_LAYERS = [{'num_units': 600}, {'num_units': 500}]  # 表示：有两个隐藏层(LSTM),隐藏层内的隐藏神经元个数分别为600,500
RNN_LAYERS = [{'num_units': 600}, {'num_units': 500},  {'num_units': 500}]
DENSE_LAYERS = None
TRAINING_STEPS = 1000
PRINT_STEPS = TRAINING_STEPS / 10
BATCH_SIZE = 30


data_format = "diff"  # 输入控制处！ 参数控制：raw/diff/diff_normalize:三种处理原数据方式



# 构造模型估算器
# regressor = tflearn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS), model_dir=LOG_DIR)  # 得到训练后的模型实例：lstm_model(“预测值/损失值/优化操作”)
regressor = tflearn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS))

# 分割"输入(X)"和"输出(Y)"对应的数据集
if data_format == "raw":
    X, y = generate_data(TIMESTEPS, conf.save_file)  # 对原始数据(没有做差分处理)
elif data_format == "diff":
    X, y = generate_data(TIMESTEPS, conf.save_file_diff)
elif data_format == "diff_normalize":
    X, y = generate_data(TIMESTEPS, conf.save_file_diff_normalize)

# 发挥验证集的作用,create a lstm instance and validation monitor
validation_monitor = tflearn.monitors.ValidationMonitor(X['val'], y['val'],
                                                     every_n_steps=PRINT_STEPS,
                                                     early_stopping_rounds=1000)
print("X[train]:", X['train'][:5])
print("y[train]", y['train'][:5])

# 模型拟合
regressor.fit(X['train'], y['train'],  monitors=[validation_monitor],
              batch_size=BATCH_SIZE, steps=TRAINING_STEPS)
# 模型预测
predicted = regressor.predict(X['test'])
# 均方根误差RMSE
RMES = np.sqrt(((predicted - y['test']) ** 2).mean(axis=0))
MAPE = np.mean(np.abs((predicted - y['test'])/y['test']))
print ("RMES:%f - MAPE: %f" % (RMES, MAPE))
end_time = time.time()
print("cost time: %dminutes" % (int(end_time - start_time) / 60))

# 数据还原操作
with np.load(conf.save_file) as file_load:  # 加载原始数据
    orig_train_data, orig_valid_data, orig_test_data = [file_load['arr_%d' % i] for i in range(len(file_load.files))]
priTest_data = []  # 预测结果有"len(test_data)-TIMESTEPS"个数据,靠后的
for i, data in enumerate(predicted):  # enumerate:既能输出索引又能输出值
    if data_format == "raw":  # 原始数据,没有做差分处理,不必还原处理
        priTest_data.append(data)
    elif data_format == "diff":  # "差分"后的还原处理,还原公式：orig_x = pridict + orig_last_X
        priTest_data.append(data + orig_test_data[i + TIMESTEPS - 1])  # 还原: TIMESTEPS个数据序列预测下一个数据
    elif data_format == "diff_normalize":
        priTest_data.append((data + 1) * orig_test_data[i + TIMESTEPS - 1])
    else:
        print ("数据还原时数据源不存在！")
pri_tRMSE = np.sqrt(np.mean((np.array(priTest_data) - np.array(orig_test_data[TIMESTEPS:])) ** 2))  # 均方根误差RMSE:平均单个数据的实际值与预测值之间的偏差
pri_tMAPE = np.mean(np.abs(np.array(priTest_data) - np.array(orig_test_data[TIMESTEPS:])) / np.array(orig_test_data[TIMESTEPS:]))
print("TestData pri_tRMSE: %.3f - pri_tMAPE: %.3f" % (pri_tRMSE, pri_tMAPE))

# 可视化
fig = plt.figure(figsize=(12, 8))  # 画布
ax1 = fig.add_subplot(211)  # 坐标系
ax2 = fig.add_subplot(212)
plot_predicted, = ax1.plot(predicted[-100:], marker="*", ms=6.0, linewidth=0.5, label='predicted', color="r")
plot_test, = ax1.plot(y['test'][-100:], marker="o", ms=6.0, linewidth=0.5, label='test', color="b")
plot_orig_predicted, = ax2.plot(priTest_data[-100:], marker="*", ms=6.0, linewidth=0.5, label='orig_predicted', color="r")
plot_orig_test, = ax2.plot(orig_test_data[-100:], marker="o", ms=6.0, linewidth=0.5, label='orig_test', color="b")
ax1.legend(handles=[plot_predicted, plot_test])  # 在坐标右上方画出label
ax2.legend(handles=[plot_orig_predicted, plot_orig_test])
plt.show()
a = 1

