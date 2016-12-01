# -*-coding:gbk-*-
# -*-coding:utf-8-*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import sys
import os
from collections import deque
import config as c
import tensorflow as tf
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell

# 读入原始数据,保存在listing_data：DataFrame类型,索引是默认的“int”类型
listing_data = pd.read_csv('data/dow30.csv', usecols=['Date', 'Open', "High", "Low", 'Close', 'Volume', "Adj Close"])  # 读出后已去掉第一行字段
# 如果将“Date”指定为索引index_col="Date",则“Date”这一列则不属于listing_data的其中一列值,即不属于表的列,成为表的索引,和其他列有本质的不同.
# listing_data = pd.read_csv('data/dow30.csv', index_col="Date", usecols=['Date', 'Open', "High", "Low", 'Close', 'Volume', "Adj Close"])
# print(listing_data.head())
# print(listing_data.head().index)

"""原始数据部分可视化"""
def Data_Visualization():
    # seaborn画图,factorplot()参数x:x轴变量展示的是“Data”列所对应的值(y轴同理);参数data:数据源;x,y来自data数据源
    # sns.factorplot(x='Date', y='Adj Close', data=listing_data[:50], size=6, aspect=2)

    xticks_dates = []  # x轴的刻度(ticks)显示日期
    [xticks_dates.append(listing_data[i:i+1]["Date"].tolist()[0]) for i in np.arange(10)*5]

    # plt.xticks() 第1个参数：需要显示标签的横坐标的位置(int),每个5个位置显示标签,共显示10处; # 第2个参数：需要显示标签的10个位置所对应的具体内容;第3个参数rotation: x轴刻度标签旋转多少度.
    xt = plt.xticks(np.arange(10)*5, xticks_dates, rotation=90)
    plt.show()

# 只取“Adj Close”这一列数据做分析和预测,并转换成list类型
AdjClose_datalist = listing_data['Adj Close'].tolist()
# print(AdjClose_datalist[:50])

"""
    Divide_data(): 将数据分割成“训练集/验证集/测试集”三部分数据,列表类型
"""
def Divide_data():
    '''
    如果划分文件已存在则加载,否则进行文件的划分并将划分后的文件进行保存.
    Returns:
    - a list of arrays of close-to-close percentage returns, normalized by running
      stdev calculated over last c.normalize_std_len days
    '''
    def divide_data():
        from datetime import timedelta
        # find date range for the split train, val, test (0.8, 0.1, 0.1 of total days)
        split = [0.8, 0.1, 0.1]
        cumusplit = [np.sum(split[:i]) for i, s in enumerate(split)]
        segment_start_dates = [c.start + timedelta(
            days=int((c.end - c.start).days * interv)) for interv in cumusplit][::-1]
        seq = [[], [], []]
        # lastAc = -1
        # daily_returns = deque(maxlen=c.normalize_std_len)
        listing_data.index = listing_data["Date"].tolist()
        for rec_date in (c.start + timedelta(days=n) for n in xrange((c.end - c.start).days)):
            idx = next(i for i, d in enumerate(segment_start_dates) if rec_date >= d)
            try:
                d = rec_date.strftime("%Y-%m-%d")
                ac = listing_data.ix[d]['Adj Close']  # listing_data:数据源,由.csv文件读入,DateFrame类型
                # daily_return = (ac - lastAc) / lastAc  # 一次一阶差分标准化
                # if len(daily_returns) == daily_returns.maxlen:
                #     seq[idx].append(daily_return / np.std(daily_returns))  # 一次一阶差分标准化后,再进行数据长度为50的标准差归一化
                # daily_returns.append(daily_return)
                # lastAc = ac
                seq[idx].append(ac)  # 原始数据,不经过处理
            except KeyError:
                pass
        return [np.asarray(dat, dtype=np.float32) for dat in seq][::-1]

    if not os.path.exists(c.save_file):
        datasets = divide_data()
        print('Saving in {}'.format(c.save_file))
        np.savez(c.save_file, *datasets)
    else:
        with np.load(c.save_file) as file_load:
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
      The second element of the tuple is the same data time-shifted to the
      right by one.
    Raises:
    - ValueError: if batch_size or num_steps are too high.
    """
    raw_data = np.array(raw_data, dtype=np.float32)

    data_len = len(raw_data)
    batch_nums = data_len // batch_size  # //:商取整
    data = np.zeros([batch_size, batch_nums], dtype=np.float32)
    for i in range(batch_size):  # 将列表形式的原始数据转换成矩阵形式表示,矩阵大小[batch_size, batch_nums]
        data[i] = raw_data[batch_nums * i:batch_nums * (i + 1)]

    epoch_size = (batch_nums - 1) // num_steps

    if epoch_size == 0:  # 减少batch_size/num_steps可增大epoch_size
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    # for i in range(epoch_size):  # 将整个矩阵的数据划分成“块状的‘X-Y’数据对”,便于扔进模型训练(这种方法切分(x,y)数据对太少,不利于模型训练)
    #     x = data[:, i * num_steps:(i + 1) * num_steps]  # num_steps:代表数据块的宽度.决定了"(x,y)数据对"有几对(可用于训练的数据条数),影响了模型训练精度和速度
    #     y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
    for i in range(epoch_size*num_steps-10):  # 该方法较上一种切分(x,y)数据对方法更科学,切分出更多的数据对用于训练.(上一种方法是无重复的切分,本方法是有重复的切分)
        x = data[:, i:(i + num_steps)]
        y = data[:, (i + 1): (i + 1 + num_steps)]
        yield (x, y)
        # yield 的作用就是把一个函数变成一个 generator,带有 yield 的函数不再是一个普通函数,Python解释器会将其视为一个 generator
        # 一个带有 yield 的函数就是一个 generator，它和普通函数不同，生成一个 generator 看起来像函数调用，但不会执行任何函数代码，
        # 虽然执行流程仍按函数的流程执行，但每执行到一个 yield 语句就会中断，并返回一个迭代值，下次执行时从 yield 的下一个语句继续执行。
        # 看起来就好像一个函数在正常执行的过程中被 yield 中断了数次，每次中断都会通过 yield 返回当前的迭代值。
        # 一句话：起迭代器作用,省内存.

"""StockLSTM(): LSTM模型训练封装成类"""
class StockLSTM(object):
    """
    This model predicts a 1D sequence of real numbers (here representing daily stock adjusted
    returns normalized by running fixed-length standard deviation) using an LSTM.
    It is regularized using the method in [Zaremba et al 2015]
    http://arxiv.org/pdf/1409.2329v5.pdf
    """

    def __init__(self, is_training, config):  # 类似于C++的“构造函数”
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size

        self._input_data = tf.placeholder(tf.float32, [batch_size, num_steps])  # 输入batch_size×num_steps个数据,输出个数相同
        self._targets = tf.placeholder(tf.float32, [batch_size, num_steps])  # placeholder:训练时需要传进真实数据的参数

        lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
        if is_training and config.keep_prob < 1:
            lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
        cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        #  以下是RNN_LSTM算法核心：最简单的线性函数y=wx+b(太简单了,精度不够？)
        iw = tf.get_variable("input_w", [1, size])
        ib = tf.get_variable("input_b", [size])
        inputs = [tf.nn.xw_plus_b(i_, iw, ib) for i_ in tf.split(1, num_steps, self._input_data)] # split沿列均匀分割成num_steps个张量(矩阵)
        if is_training and config.keep_prob < 1:
            inputs = [tf.nn.dropout(input_, config.keep_prob) for input_ in inputs]

        outputs, states = rnn.rnn(cell, inputs, initial_state=self._initial_state)
        # c_out = tf.concat(1, outputs)  # outputs:p个m×n; c_out:m × n*p
        # c_out = tf.concat(0, outputs)  # outputs:p个m×n; c_out:m*p × n
        rnn_output = tf.reshape(tf.concat(1, outputs), [-1, size])  # [-1, size]:保持总元素个数不变,size=1×200表示一个数的权重
        # rnn_output:所得到的是n×200,代表有n个输出(对应n个输入),每个输出由1×200的一维向量表示

        # output：神经元计算的最终输出结果(即我们需要的结果),输出个数与输入个数相同
        self._output = output = tf.nn.xw_plus_b(rnn_output,
                                                tf.get_variable("out_w", [size, 1]),
                                                tf.get_variable("out_b", [1]))

        self._cost = cost = tf.reduce_mean(tf.abs(output - tf.reshape(self._targets, [-1])))  # 平均误差
        self._final_state = states

        if not is_training:  # 验证/测试或着真实预测时,则不更新权重(即不执行下面的语句)
            return
        # 训练网络,反向传播,更新权重
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        # optimizer = tf.train.GradientDescentOptimizer(self.lr)
        optimizer = tf.train.AdamOptimizer(self.lr)  # 反向传播,更新权重
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
    def cost(self):
        return self._cost

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

def main(config_size='small', num_epochs=15):

    def get_config(config_size):
        config_size = config_size.lower()
        if config_size == 'small':
            return c.SmallConfig()
        elif config_size == 'medium':
            return c.MediumConfig()
        elif config_size == 'large':
            return c.LargeConfig()
        else:
            raise ValueError('Unknown config size {} (small, medium, large)'.format(config_size))

    """ run_epoch(): Runs the model on the given data.用整个数据集（测试集）将整个网络完整训练一遍的过程 Return:均方误差 """
    def run_epoch(session, m, data, eval_op, verbose=False):
        epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
        # print("epoch_size数据块大小："+epoch_size)
        start_time = time.time()
        costs = 0.0
        iters = 0
        state = m.initial_state.eval()
        outputs = []  # 存储每一轮的计算(预测)结果y_
        for step, (x, y) in enumerate(seq_iterator(data, m.batch_size, m.num_steps)):
            # 完整数据集分成了m个“(X,Y)数据块”,连续训练m次,刚好把整个数据集遍历了一遍
            output, cost, state, _ = session.run([m.output, m.cost, m.final_state, eval_op],
                                         {m.input_data: x, m.targets: y, m.initial_state: state})
            outputs.append(output)
            costs += cost
            iters += m.num_steps

            print_interval = 20
            if verbose and epoch_size > print_interval \
                    and step % (epoch_size // print_interval) == print_interval:
                print("%.3f mse: %.8f speed: %.0f ips" % (step * 1.0 / epoch_size, costs / iters,
                                                          iters * m.batch_size / (time.time() - start_time)))
        return outputs, costs / (iters if iters > 0 else 1)  # 返回计算结果（预测值）和均方误差

    with tf.Graph().as_default(), tf.Session() as session:
        config = get_config(config_size)
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)  # 返回生成具有均匀分布的张量的初始化器。
        with tf.variable_scope("model", reuse=None, initializer=initializer):  # variable_scope():返回变量作用域的上下文.
            m = StockLSTM(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mtest = StockLSTM(is_training=False, config=config)

        tf.initialize_all_variables().run()

        train_data, valid_data, test_data = Divide_data()

        for epoch in xrange(num_epochs):  # num_epochs:迭代次数,即整个网络完整训练num_epochs遍
            lr_decay = config.lr_decay ** max(epoch - num_epochs, 0.0)  # 学习率衰变操作
            m.assign_lr(session, config.learning_rate * lr_decay)
            cur_lr = session.run(m.lr)

            _, mse = run_epoch(session, m, train_data, m.train_op, verbose=True)  # Mean square error,训练集均方误差
            _, vmse = run_epoch(session, mtest, valid_data, tf.no_op())  # tf.no_op():Does nothing. Only useful as a placeholder for control edges.验证集均方误差
            print("Epoch: %d - learning rate: %.3f - train mse: %.3f - valid mse: %.3f" %
                  (epoch, cur_lr, mse, vmse))

        pridict_train, tmse = run_epoch(session, mtest, test_data, tf.no_op())  # 模型训练完成后,用测试集数据测试,得出测试集数据的均方误差
        print("Test mse: %.3f" % tmse)
if __name__ == '__main__':
    # make all arguments of main(...) command line arguments (with type inferred from
    # the default value) - this doesn't work on bools so those are strings when
    # passed into main.
    import argparse, inspect

    parser = argparse.ArgumentParser(description='Command line options')
    ma = inspect.getargspec(main)
    for arg_name, arg_type in zip(ma.args[-len(ma.defaults):], [type(de) for de in ma.defaults]):
        parser.add_argument('--{}'.format(arg_name), type=arg_type, dest=arg_name)
    args = parser.parse_args(sys.argv[1:])
    a = {k: v for (k, v) in vars(args).items() if v is not None}
    main(**a)