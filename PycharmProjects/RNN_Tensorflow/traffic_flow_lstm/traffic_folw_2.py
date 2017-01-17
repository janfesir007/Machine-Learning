# -*-coding:gbk-*-
# -*-coding:utf-8-*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
# from sklearn.grid_search import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import config as c
import tensorflow as tf

# 读入原始数据,保存在listing_data：DataFrame类型,索引是默认的“int”类型
listing_data = pd.read_csv('data/flow_year.csv', usecols=['Hour', 'Flow (Veh/Hour)', "# Lane Points", "% Observed"])  # 读出后已去掉第一行字段
# 如果将“Date”指定为索引index_col="Date",则“Date”这一列则不属于listing_data的其中一列值,即不属于表的列,成为表的索引,和其他列有本质的不同.
# listing_data = pd.read_csv('data/dow30.csv', index_col="Date", usecols=['Date', 'Open', "High", "Low", 'Close', 'Volume', "Adj Close"])
# print(listing_data.head())
# print(listing_data.head().index)

"""原始数据部分可视化"""
def Data_Visualization():
    # 第一种画法：传统plt
    show_data = listing_data[-100:]['Flow (Veh/Hour)']
    fig = plt.figure(figsize=(12, 8))  # 画布
    ax1 = fig.add_subplot(111)  # 坐标系
    plt.plot(np.arange(len(show_data)), show_data, marker="*", markersize=9.0, linewidth=.5, color="b")  # plt.plot(x,y);maker：标记,*(描点的形状)
    # plt.plot(np.arange(len(show_data)), show_data, linewidth=.5, linestyle='-', color="b")
    xticks_dates = []  # x轴的刻度(ticks)显示日期
    [xticks_dates.append(listing_data[5900 + i:5901 + i]["Hour"].tolist()[0]) for i in np.arange(20) * 5]

    # 设置x轴刻度上的标签:plt.xticks() 第1个参数：需要显示标签的横坐标的位置(int),每个5个位置显示标签,共显示20处;
    #  第2个参数：需要显示标签的20个位置所对应的具体内容;第3个参数rotation: x轴刻度标签旋转多少度.
    plt.xticks(np.arange(20) * 5, xticks_dates, rotation=90)
    plt.show()

    # # 第二种画法：seaborn画图,factorplot()参数x:x轴变量展示的是“Data”列所对应的值(y轴同理);参数data:数据源;x,y来自data数据源
    # sns.factorplot(x='Date', y='Adj Close', data=listing_data[-100:], size=6, aspect=2, color="b")
    # xticks_dates = []  # x轴的刻度(ticks)显示日期
    # [xticks_dates.append(listing_data[7927+i:7928+i]["Date"].tolist()[0]) for i in np.arange(20)*5]
    #
    # # plt.xticks() 第1个参数：需要显示标签的横坐标的位置(int),每个5个位置显示标签,共显示20处; # 第2个参数：需要显示标签的20个位置所对应的具体内容;第3个参数rotation: x轴刻度标签旋转多少度.
    # plt.xticks(np.arange(20)*5, xticks_dates, rotation=90)  # 设置x轴刻度上的标签
    # plt.show()


"""
    Divide_data(): 将数据分割成“训练集/验证集/测试集”三部分数据,列表类型
"""
def Divide_data(file):
    '''
    如果划分文件已存在则加载,否则进行文件的划分并将划分后的文件进行保存.
    Returns:
    - a list of arrays of close-to-close percentage returns, normalized by running
      stdev calculated over last c.normalize_std_len days
    '''
    def divide_data():
        # find date range for the split train, val, test (0.8, 0.1, 0.1 of total days)
        # split = [0.8, 0.1, 0.1]
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

    if not os.path.exists(file):
        datasets = divide_data()
        print('Saving in {}'.format(file))
        np.savez(file, *datasets)
    else:
        with np.load(file) as file_load:
            datasets = [file_load['arr_%d' % i] for i in range(len(file_load.files))]
    return datasets


"""seq_iterator():将列表形式的原始数据(raw_data)转换成矩阵形式表示,矩阵大小[batch_size, batch_nums],将整个矩阵的数据划分成“块状的‘X-Y’数据对”,便于扔进模型训练."""
def seq_iterator(raw_data, batch_size, num_steps):
    """
    Iterate on the raw return sequence data.
    Args:
    - raw_data: array
    - batch_size: int, the batch size.
    - num_steps: int, the number of unrolls.
    Yields:
    - Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
      The second element of the tuple is the same data time-shifted to the right by one.
    Raises:
    - ValueError: if batch_size or num_steps are too high.
    """
    raw_data = np.array(raw_data, dtype=np.float64)
    data_len = len(raw_data)
    batch_nums = data_len // batch_size  # //:商取整
    data = np.zeros([batch_size, batch_nums], dtype=np.float32)
    for i in range(batch_size):  # 将列表形式的原始数据转换成矩阵形式表示,矩阵大小[batch_size, batch_nums]
        data[i] = raw_data[batch_nums * i:batch_nums * (i + 1)]

    epoch_size = (batch_nums - 1) // num_steps  # 确保训练时数据不能只有一组
    if epoch_size == 0:  # 减少batch_size/num_steps可增大epoch_size
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    if epoch_size == 1:
        x = data[:, 0:num_steps]
        y = data[:, 1:num_steps + 1]  # n个序列对应n个预测结果
        # y = data[:, num_steps:num_steps + 1]  # n个序列对应1个预测结果
        yield (x, y)
    else:   # 数据切分,将整个矩阵的数据划分成“块状的‘X-Y’数据对”,便于扔进模型训练
        for i in range(batch_nums - num_steps):  # 后移一位继续切分
            x = data[:, i:(i + num_steps)]  # num_steps:代表数据块的宽度.
            y = data[:, i+1:(i + num_steps + 1)]  # 长度为num_steps的数据序列x,对应预测num_steps个结果
            # y = data[:, (i + num_steps):(i + num_steps + 1)]  # 长度为num_steps的数据序列x,对应预测的一个y
            yield (x, y)
        # yield 的作用就是把一个函数变成一个 generator,带有 yield 的函数不再是一个普通函数,Python解释器会将其视为一个 generator
        # 一个带有 yield 的函数就是一个 generator，它和普通函数不同，生成一个 generator 看起来像函数调用，但不会执行任何函数代码，
        # 虽然执行流程仍按函数的流程执行，但每执行到一个 yield 语句就会中断，并返回一个迭代值，下次执行时从 yield 的下一个语句继续执行。
        # 看起来就好像一个函数在正常执行的过程中被 yield 中断了数次,每次中断都会通过 yield 返回当前的迭代值。
        # 一句话：起迭代器作用,省内存.


class StockLSTM(object):  # StockLSTM(): LSTM模型训练封装成类
    """
    This model predicts a 1D sequence of real numbers (here representing daily stock adjusted
    returns normalized by running fixed-length standard deviation) using an LSTM.
    It is regularized using the method in [Zaremba et al 2015]
    http://arxiv.org/pdf/1409.2329v5.pdf
    """
    def __init__(self, is_training, config):  # 类似于C++的“构造函数”
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size  # 每个隐藏层的单元数（抽象的特征数）

        self._input_data = tf.placeholder(tf.float32, [batch_size, num_steps])  # 输入batch_size×num_steps个数据,输出batch_size×1个
        self._targets = tf.placeholder(tf.float32, [batch_size, num_steps])  # placeholder:训练时需要传进真实数据的参数,每一个长度为num_steps的序列预测出一个值（最后一个）
        # self._targets = tf.placeholder(tf.float32, [batch_size, 1])

        # lstm_cell = tf.nn.rnn_cell.BasicRNNCell(size)  # 封装好的普通RNN单元
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=1.0)  # 封装好的LSTM单元
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)  # 多个RNN单元

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        #  以下是RNN_LSTM算法核心：最简单的线性函数y=wx+b(太简单了,精度不够？)
        iw = tf.get_variable("input_w", [1, size])
        ib = tf.get_variable("input_b", [size])
        inputs = [tf.nn.xw_plus_b(i_, iw, ib) for i_ in tf.split(1, num_steps, self._input_data)] # split沿列均匀分割成"num_steps个"张量(矩阵)
        # inputs = [i_ for i_ in tf.split(1, num_steps, self._input_data)]  # 不进行wx+b操作,直接作为输入
        if is_training and config.keep_prob < 1:
            inputs = [tf.nn.dropout(input_, config.keep_prob) for input_ in inputs]

        outputs, states = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)  # outputs:[num_steps,batch_size,size],num_steps个[batch_size,size]矩阵：即每个step的输出结果
        # c_out = tf.concat(1, outputs)  # outputs:p个m×n矩阵; c_out:[m × n*p]:m行,n*p列矩阵
        # c_out = tf.concat(0, outputs)  # outputs:p个m×n; c_out:m*p × n
        # rnn_output = tf.reshape(tf.concat(1, outputs), [-1, size])  # 该操作的目的：将每一步step的输出结果都保留.[-1, size]:保持总元素个数不变,size=hidden_size
        # rnn_output:所得到的是n×size,代表有n个输出(对应n个输入),每个输出由1×size的一维向量表示.

        # output：神经元计算的最终输出结果(即我们需要的结果)
        self._output = output = tf.nn.xw_plus_b(outputs[-1],  # 有num_steps个输出结果,只取最后一步step的结果
                                                tf.get_variable("out_w", [size, 1]),
                                                tf.get_variable("out_b", [1]))

        self._cost_RMSE = cost_RMSE = tf.sqrt(tf.reduce_mean((output - self._targets[:, num_steps-1:]) ** 2))  # 均方根误差RMSE:平均单个数据的实际值与预测值之间的偏差
        self._cost_MAPE = cost_MAPE = tf.reduce_mean(tf.abs((output - self._targets[:, num_steps-1:])/(self._targets[:, num_steps-1:])))
        self._final_state = states

        if not is_training:  # 验证/测试或着真实预测时,则不更新权重(即不执行下面的语句)
            return
        # 训练网络,反向传播,更新权重
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost_RMSE, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        # optimizer = tf.train.AdamOptimizer(self.lr)  # 反向传播,更新权重
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost_RMSE(self):
        return self._cost_RMSE

    @property
    def cost_MAPE(self):
        return self._cost_MAPE

    @property
    def output(self):
        return self._output

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


def main(data_format):  # data_format:数据格式(原始数据/差分后数据/差分标准化后数据)
    def get_config(config_size):
        config_size = config_size.lower()
        if config_size == 'small':
            return c.SmallConfig()
        elif config_size == 'medium':
            return c.MediumConfig()
        elif config_size == 'large':
            return c.LargeConfig()
        else:
            raise ValueError('Unknown config size {} (small, medium, large, test)'.format(config_size))

    """ run_epoch(): Runs the model on the given data.用整个数据集（测试集）将整个网络完整训练一遍的过程 Return:均方误差 """
    def run_epoch(session, m, data, eval_op, verbose=False):
        epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
        # print("epoch_size数据块大小："+epoch_size)
        start_time = time.time()
        costs_RMSE = 0.0
        costs_MAPE = 0.0
        iters = 0
        iters1 = 0
        # state = m.initial_state.eval()
        state = session.run(m.initial_state)
        outputs = []  # 存储每一轮的计算(预测)结果y_
        for step, (x, y) in enumerate(seq_iterator(data, m.batch_size, m.num_steps)):
            # 完整数据集分成了m个“(X,Y)数据块”,连续训练m次,刚好把整个数据集遍历了一遍
            output, cost_RMSE, cost_MAPE, final_state, _ = session.run([m.output, m.cost_RMSE, m.cost_MAPE, m.final_state, eval_op],
                                                 {m.input_data: x, m.targets: y, m.initial_state: state})
            outputs.append(output)
            costs_RMSE += cost_RMSE
            if cost_MAPE!=float("inf"):  # 去除分母为0的情况
                costs_MAPE += cost_MAPE
                iters1 += 1
            iters += 1
            print_interval = 20
            if verbose and epoch_size > print_interval and step % (epoch_size // print_interval) == print_interval:
                # 每个epoch要输出10个perplexity值
                print("%.3f mse: %.8f speed: %.0f ips" % (step * 1.0 / epoch_size, costs_RMSE / iters, iters * m.batch_size / (time.time() - start_time)))
        return outputs, costs_RMSE / (iters if iters > 0 else 1), costs_MAPE / (iters1 if iters1 > 0 else 1)  # 返回计算结果（预测值）和均方根误差RMSE和相对误差MAPE

    with tf.Graph().as_default(), tf.Session() as session:
        config = get_config("Medium")  # 获得某一组配置好的全局参数
        pridict_config = get_config("Medium")  # 测试时一个一个数据的预测,而不是块
        pridict_config.batch_size = 1
        # pridict_config.num_steps = 1

        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)  # 返回生成具有随机均匀分布的张量的初始化器,初始化相关参数.
        with tf.variable_scope("model", reuse=None, initializer=initializer):  # variable_scope():返回变量作用域的上下文.
            m = StockLSTM(is_training=True, config=config)  # 训练模型,is_trainable=True
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            # mvalid = StockLSTM(is_training=False, config=config)  # 交叉检验和测试模型,is_training=False
            mpridict = StockLSTM(is_training=False, config=pridict_config)

        tf.initialize_all_variables().run()

        if data_format == "raw":  # 原始数据
            train_data, valid_data, test_data = Divide_data(c.save_file)  # 分割经过差分处理后的数据集:都以一维数组(narray)形式存放
        elif data_format == "diff":
            train_data, valid_data, test_data = Divide_data(c.save_file_diff)
        elif data_format == "diff_normalize":
            train_data, valid_data, test_data = Divide_data(c.save_file_diff_normalize)
        else:
            print ("数据源不存在！")

        valid_test_data = np.array(list(valid_data) + list(test_data))  # 将验证集那部分数据也作为测试集加入
        for epoch in xrange(config.max_max_epoch):  # max_max_epoch:迭代次数,即整个网络完整训练max_max_epoch遍
            lr_decay = config.lr_decay ** max(epoch - config.max_epoch, 0.0)  # 学习率衰变操作:遍数小于max epoch时,lr_decay=1;大于max_epoch时,lr_decay = lr_decay**(epoch-max_epoch)
            m.assign_lr(session, config.learning_rate * lr_decay)  # 改变学习率
            cur_lr = session.run(m.lr)
            num_steps = config.num_steps

            _train, RMSE, MAPE = run_epoch(session, m, train_data, m.train_op, verbose=True)  # m/m.train_op中有优化操作,每遍历一遍就优化一次！

            # 验证集对于模型训练本身不起作用,但对于经过训练后得到的模型好坏可以提供判断标准（同一个模型不同参数好坏的判断;多个不同模型好坏的判断）
            # _valid, vRMSE, vMAPE = run_epoch(session, mvalid, valid_data, tf.no_op())  # tf.no_op():Does nothing. Only useful as a placeholder for control edges.
            print("Epoch: %d - learning rate: %.3f - data: train RMSE: %.3f " % (epoch, cur_lr, RMSE))
            print("Epoch: %d - learning rate: %.3f - data: train MAPE: %.3f " % (epoch, cur_lr, MAPE))

        # 模型训练完成后,用测试集数据测试,但此时batch_size设置为1,则可预测test_data靠后的"len(test_data)-num_steps"个数据;
        # pridict_test_onebyone, tRMSE, tMAPE = run_epoch(session, mpridict, test_data, tf.no_op())
        pridict_test_onebyone, tRMSE, tMAPE = run_epoch(session, mpridict, valid_test_data, tf.no_op())  # 将验证集那部分数据也作为测试集加入
        print("Test_onebyone tRMSE: %.3f - tMAPE:%.3f " % (tRMSE, tMAPE))

    with np.load(c.save_file) as file_load:  # 加载原始数据
        orig_train_data, orig_valid_data, orig_test_data = [file_load['arr_%d' % i] for i in range(len(file_load.files))]
    orig_valid_test_data = np.array(list(orig_valid_data) + list(orig_test_data))  # 将验证集那部分数据也作为测试集加入
    priTest_data = []   # 还原后的预测值.预测结果有"len(test_data)-num_steps"个数据,靠后的
    priTest_data1 = []  # 未进行还原操作的预测数据
    for i, data in enumerate(pridict_test_onebyone):  # enumerate:既能输出索引又能输出值
        priTest_data1.append(data[0][0])
        if data_format == "raw":  # 原始数据,没有做差分处理
            priTest_data.append(data[0][0])
        elif data_format == "diff":  # "差分"后的还原处理,还原公式：orig_x = pridict + orig_last_X
            priTest_data.append(data[0][0] + orig_valid_test_data[i + num_steps - 1])  # 还原
        elif data_format == "diff_normalize":
            priTest_data.append((data[0][0]+1)*orig_valid_test_data[i + num_steps - 1])
        else:
            print ("数据还原时数据源不存在！")

    pri_tRMSE = np.sqrt(np.mean((np.array(priTest_data) - np.array(orig_valid_test_data[num_steps:])) ** 2))  # 均方根误差RMSE:平均单个数据的实际值与预测值之间的偏差
    pri_tMAPE = np.mean(np.abs(np.array(priTest_data) - np.array(orig_valid_test_data[num_steps:])) / np.array(orig_valid_test_data[num_steps:]))
    print("TestData pri_tRMSE: %.3f - pri_tMAPE: %.3f" % (pri_tRMSE, pri_tMAPE))

    # # "差分标准化"后的还原处理,还原公式：orig_x = orig_last_X*pridict + orig_last_X
    # with np.load(c.save_file) as file_load:  # 加载原始数据
    #     orig_train_data, orig_valid_data, orig_test_data = [file_load['arr_%d' % i] for i in range(len(file_load.files))]
    # # priTest_data = [(pridict_test_onebyone[0][0][0]*valid_data[-1]+valid_data[-1])]  # 首个预测值
    # priTest_data = []  # 预测结果有"len(test_data)-num_steps"个数据,靠后的
    # for i, data in enumerate(pridict_test_onebyone):  # enumerate:既能输出索引又能输出值
    #     priTest_data.append(data[0][0]*orig_test_data[i+num_steps-1] + orig_test_data[i+num_steps-1])  # 还原
    # pri_tRMSE = np.sqrt(np.mean((np.array(priTest_data) - np.array(orig_test_data[num_steps:])) ** 2))  # 均方根误差RMSE:平均单个数据的实际值与预测值之间的偏差
    # pri_tMAPE = np.mean(np.abs(np.array(priTest_data) - np.array(orig_test_data[num_steps:]))/np.array(orig_test_data[num_steps:]))
    # print("TestData pri_tRMSE: %.3f - pri_tMAPE: %.3f" % (pri_tRMSE, pri_tMAPE))
    return priTest_data1, priTest_data

if __name__ == '__main__':
    # Data_Visualization()
    start_time = time.time()
    # 画出对比图：传统plt
    show_lenth = 300
    data_format = "raw"  # 参数控制：raw/diff/diff_normalize:三种处理原数据方式
    process_data = []
    orig_data = listing_data[-show_lenth:]['Flow (Veh/Hour)']
    a, b = main(data_format)  # 调用函数：输入数据,得到预测结果,列表类型
    pric_process_data, pric_orig_data = a[-show_lenth:], b[-show_lenth:]
    if data_format=="raw":
        process_data = orig_data
    if data_format=="diff":
        with np.load(c.save_file_diff) as file_load:  # 加载原始数据
            orig_train_data, orig_valid_data, orig_test_data = [file_load['arr_%d' % i] for i in range(len(file_load.files))]
        orig_valid_test_data = np.array(list(orig_valid_data) + list(orig_test_data))  # 将验证集那部分数据也作为测试集加入
        process_data = orig_valid_test_data[-show_lenth:]
    if data_format=="diff_normalize":
        with np.load(c.save_file_diff_normalize) as file_load:  # 加载原始数据
            orig_train_data, orig_valid_data, orig_test_data = [file_load['arr_%d' % i] for i in range(len(file_load.files))]
        orig_valid_test_data = np.array(list(orig_valid_data) + list(orig_test_data))  # 将验证集那部分数据也作为测试集加入
        process_data = orig_valid_test_data[-show_lenth:]
    fig = plt.figure(figsize=(12, 8))  # 画布
    ax1 = fig.add_subplot(211)  # 坐标系
    ax2 = fig.add_subplot(212)
    ax1.plot(np.arange(show_lenth), process_data, marker="*", ms=6.0, linewidth=0.5, color="b", label=data_format+"_data")  # plt.plot(x,y);ms=markersize
    ax1.plot(np.arange(show_lenth), pric_process_data, marker="o", ms=6.0, linewidth=0.5, color="r", label=data_format+'_data_pridict')
    ax2.plot(np.arange(show_lenth), orig_data, marker="*", ms=6.0, linewidth=0.5, color="b", label='raw_data')  # plt.plot(x,y);ms=markersize
    ax2.plot(np.arange(show_lenth), pric_orig_data, marker="o", ms=6.0, linewidth=0.5, color="r", label='raw_data_pridict')
    xticks_dates = []  # x轴的刻度(ticks)显示日期
    [xticks_dates.append(listing_data[int(len(listing_data))-show_lenth + i:int(len(listing_data))-show_lenth+1+i]["Hour"].tolist()[0]) for i in np.arange(20) * int(show_lenth/20)]

    # 设置x轴刻度上的标签:plt.xticks() 第1个参数：需要显示标签的横坐标的位置(int),每个5个位置显示标签,共显示20处;
    #  第2个参数：需要显示标签的20个位置所对应的具体内容;第3个参数rotation: x轴刻度标签旋转多少度.
    plt.xticks(np.arange(20) * (show_lenth/20), xticks_dates, axes=ax1, rotation=45)
    plt.show()
    end_time = time.time()
    print("cost time: %dminutes" % (int(end_time - start_time) / 60))



    # make all arguments of main(...) command line arguments (with type inferred from
    # the default value) - this doesn't work on bools so those are strings when
    # passed into main.
    # import argparse, inspect
    #
    # parser = argparse.ArgumentParser(description='Command line options')
    # ma = inspect.getargspec(main)
    # for arg_name, arg_type in zip(ma.args[-len(ma.defaults):], [type(de) for de in ma.defaults]):
    #     parser.add_argument('--{}'.format(arg_name), type=arg_type, dest=arg_name)
    # args = parser.parse_args(sys.argv[1:])
    # a = {k: v for (k, v) in vars(args).items() if v is not None}
    # main(**a)