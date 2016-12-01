# -*-coding:gbk-*-
# -*-coding:utf-8-*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas_datareader.data as pdr_data
import numpy as np
import time
import os
import sys
from collections import deque

import config as c
import tensorflow as tf
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell
# import matplotlib.pyplot as plt
import seaborn as sns  # seaborn是matplotlib的补充,可视化,效果更好
from tensorflow.models.rnn import seq2seq

"""
Adapted from Google's PTB word prediction TensorFlow tutorial.

Copyright 2016 Tencia Lee

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
    get_data(): 在线下载“dow30”的历史数据,经过处理(差分/标准差归一化等)后,分割成“训练集/验证集/测试集”三部分中间数据,数据为列表类型,
    并保存成名为“dow30_19900101_20151125.npz”的文件
"""
def get_data():
    '''
    If filename exists, loads data, otherwise downloads and saves data
    from Yahoo Finance
    Returns:
    - a list of arrays of close-to-close percentage returns, normalized by running
      stdev calculated over last c.normalize_std_len days
    '''

    def download_data():
        from datetime import timedelta, datetime
        # find date range for the split train, val, test (0.8, 0.1, 0.1 of total days)
        print('Downloading data for dates {} - {}'.format(
            datetime.strftime(c.start, "%Y-%m-%d"),
            datetime.strftime(c.end, "%Y-%m-%d")))
        split = [0.8, 0.1, 0.1]
        cumusplit = [np.sum(split[:i]) for i, s in enumerate(split)]
        segment_start_dates = [c.start + timedelta(
            days=int((c.end - c.start).days * interv)) for interv in cumusplit][::-1]
        stocks_list = map(lambda l: l.strip(), open(c.names_file, 'r').readlines())  # map(function, sequence) ：对sequence中的item依次执行function(item),执行结果组成一个List返回

        # by_stock1 = pdr_data.DataReader("dow", 'yahoo', c.start, c.end)  # DataReader():在线下载“dow30”的历史数据,下载得到的是DataFrame类型的数据,且索引是“Date”
        # print(by_stock1.head(), by_stock1.head().index)
        by_stock = dict((s, pdr_data.DataReader(s, 'yahoo', c.start, c.end)) for s in stocks_list) # DataReader():在线下载“dow30”的历史数据
        seq = [[], [], []]
        for stock in by_stock:
            lastAc = -1
            daily_returns = deque(maxlen=c.normalize_std_len)
            for rec_date in (c.start + timedelta(days=n) for n in xrange((c.end - c.start).days)):
                idx = next(i for i, d in enumerate(segment_start_dates) if rec_date >= d)
                try:
                    d = rec_date.strftime("%Y-%m-%d")
                    ac = by_stock[stock].ix[d]['Adj Close']  # 只选取“Adj Close”这一列数据
                    daily_return = (ac - lastAc) / lastAc  # 一次一阶差分标准化
                    if len(daily_returns) == daily_returns.maxlen:
                        seq[idx].append(daily_return / np.std(daily_returns))  # 一次一阶差分标准化后,再进行数据长度为50的标准差归一化
                    daily_returns.append(daily_return)
                    lastAc = ac
                except KeyError:
                    pass
        return [np.asarray(dat, dtype=np.float32) for dat in seq][::-1]

    if not os.path.exists(c.save_file):  # 若文件不存在则,下载文件并划分,最后存储成.npz文件
        datasets = download_data()
        print('Saving in {}'.format(c.save_file))
        np.savez(c.save_file, *datasets)
    else:
        with np.load(c.save_file) as file_load:
            datasets = [file_load['arr_%d' % i] for i in range(len(file_load.files))]
    return datasets

"""
    seq_iterator():将列表形式的中间数据(raw_data,原始数据经过处理)转换成矩阵形式表示,矩阵大小[batch_size, batch_nums],
    将整个矩阵的数据划分成“块状的‘X-Y’数据对”,便于扔进模型训练.
"""
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

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):  # 将整个矩阵的数据划分成“块状的‘X-Y’数据对”,便于扔进模型训练
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
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

        self._input_data = tf.placeholder(tf.float32, [batch_size, num_steps])  # placeholder:训练时需要传进真实数据的参数
        self._targets = tf.placeholder(tf.float32, [batch_size, num_steps])

        lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
        if is_training and config.keep_prob < 1:
            lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
        cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        iw = tf.get_variable("input_w", [1, size])
        ib = tf.get_variable("input_b", [size])
        inputs = [tf.nn.xw_plus_b(i_, iw, ib) for i_ in tf.split(1, num_steps, self._input_data)]
        if is_training and config.keep_prob < 1:
            inputs = [tf.nn.dropout(input_, config.keep_prob) for input_ in inputs]

        outputs, states = rnn.rnn(cell, inputs, initial_state=self._initial_state)
        rnn_output = tf.reshape(tf.concat(1, outputs), [-1, size])

        self._output = output = tf.nn.xw_plus_b(rnn_output,
                                                tf.get_variable("out_w", [size, 1]),
                                                tf.get_variable("out_b", [1]))

        self._cost = cost = tf.reduce_mean(tf.square(output - tf.reshape(self._targets, [-1])))  # 均方误差
        self._final_state = states

        if not is_training:
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
        for step, (x, y) in enumerate(seq_iterator(data, m.batch_size, m.num_steps)):
            # 完整数据集分成了17个“(X,Y)数据块”,连续训练17次,刚好把整个数据集遍历了一遍
            cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                         {m.input_data: x, m.targets: y, m.initial_state: state})
            costs += cost
            iters += m.num_steps

            print_interval = 20
            if verbose and epoch_size > print_interval \
                    and step % (epoch_size // print_interval) == print_interval:
                print("%.3f mse: %.8f speed: %.0f ips" % (step * 1.0 / epoch_size, costs / iters,
                                                          iters * m.batch_size / (time.time() - start_time)))
        return costs / (iters if iters > 0 else 1)  # 均方误差

    with tf.Graph().as_default(), tf.Session() as session:
        config = get_config(config_size)
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)  # 返回生成具有均匀分布的张量的初始化器。
        with tf.variable_scope("model", reuse=None, initializer=initializer):  # variable_scope():返回变量作用域的上下文.
            m = StockLSTM(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mtest = StockLSTM(is_training=False, config=config)

        tf.initialize_all_variables().run()

        train_data, valid_data, test_data = get_data()  # 三部分数据已不是原始数据,是经过了处理的中间数据

        for epoch in xrange(num_epochs):  # num_epochs:迭代次数,即整个网络完整训练num_epochs遍
            lr_decay = config.lr_decay ** max(epoch - num_epochs, 0.0)  # 学习率衰变操作
            m.assign_lr(session, config.learning_rate * lr_decay)
            cur_lr = session.run(m.lr)

            mse = run_epoch(session, m, train_data, m.train_op, verbose=True)  # Mean square error,训练集均方误差
            vmse = run_epoch(session, mtest, valid_data, tf.no_op())  # tf.no_op():Does nothing. Only useful as a placeholder for control edges.验证集均方误差
            print("Epoch: %d - learning rate: %.3f - train mse: %.3f - valid mse: %.3f" %
                  (epoch, cur_lr, mse, vmse))

        tmse = run_epoch(session, mtest, test_data, tf.no_op())  # 模型训练完成后,用测试集数据测试,得出测试集数据的均方误差
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
