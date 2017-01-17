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

# ����ԭʼ����,������listing_data��DataFrame����,������Ĭ�ϵġ�int������
listing_data = pd.read_csv('data/flow_year.csv', usecols=['Hour', 'Flow (Veh/Hour)', "# Lane Points", "% Observed"])  # ��������ȥ����һ���ֶ�
# listing_data = pd.read_csv('data/dow30.csv', usecols=['Date', 'Open', "High", "Low", 'Close', 'Volume', "Adj Close"])  # ��������ȥ����һ���ֶ�

# �������Date��ָ��Ϊ����index_col="Date",��Date����һ��������listing_data������һ��ֵ,�������ڱ����,��Ϊ�������,���������б��ʵĲ�ͬ.
# listing_data = pd.read_csv('data/dow30.csv', index_col="Date", usecols=['Date', 'Open', "High", "Low", 'Close', 'Volume', "Adj Close"])
# print(listing_data.head())
# print(listing_data.head().index)

"""ԭʼ���ݲ��ֿ��ӻ�"""
def Data_Visualization():
    # ��һ�ֻ�������ͳplt
    show_data = listing_data[-100:]['Adj Close']
    fig = plt.figure(figsize=(12, 8))  # ����
    ax1 = fig.add_subplot(111)  # ����ϵ
    plt.plot(np.arange(len(show_data)), show_data)  # plt.plot(x,y)
    xticks_dates = []  # x��Ŀ̶�(ticks)��ʾ����
    [xticks_dates.append(listing_data[7927 + i:7928 + i]["Date"].tolist()[0]) for i in np.arange(20) * 5]

    # ����x��̶��ϵı�ǩ:plt.xticks() ��1����������Ҫ��ʾ��ǩ�ĺ������λ��(int),ÿ��5��λ����ʾ��ǩ,����ʾ20��;
    #  ��2����������Ҫ��ʾ��ǩ��20��λ������Ӧ�ľ�������;��3������rotation: x��̶ȱ�ǩ��ת���ٶ�.
    plt.xticks(np.arange(20) * 5, xticks_dates, rotation=90)
    plt.show()

    # # �ڶ��ֻ�����seaborn��ͼ,factorplot()����x:x�����չʾ���ǡ�Data��������Ӧ��ֵ(y��ͬ��);����data:����Դ;x,y����data����Դ
    # sns.factorplot(x='Date', y='Adj Close', data=listing_data[-100:], size=6, aspect=2, color="b")
    # xticks_dates = []  # x��Ŀ̶�(ticks)��ʾ����
    # [xticks_dates.append(listing_data[7927+i:7928+i]["Date"].tolist()[0]) for i in np.arange(20)*5]
    #
    # # plt.xticks() ��1����������Ҫ��ʾ��ǩ�ĺ������λ��(int),ÿ��5��λ����ʾ��ǩ,����ʾ20��; # ��2����������Ҫ��ʾ��ǩ��20��λ������Ӧ�ľ�������;��3������rotation: x��̶ȱ�ǩ��ת���ٶ�.
    # plt.xticks(np.arange(20)*5, xticks_dates, rotation=90)  # ����x��̶��ϵı�ǩ
    # plt.show()

# ֻȡ��Adj Close����һ��������������Ԥ��,��ת����list����
AdjClose_datalist = listing_data['Flow (Veh/Hour)'].tolist()
# AdjClose_datalist = listing_data['Adj Close'].tolist()
# print(AdjClose_datalist[:50])

"""
    Divide_data(): �����ݷָ�ɡ�ѵ����/��֤��/���Լ�������������,�б�����
"""
def Divide_data():
    '''
    ��������ļ��Ѵ��������,��������ļ��Ļ��ֲ������ֺ���ļ����б���.
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
        lastAc = -1
        # daily_returns = deque(maxlen=c.normalize_std_len)
        listing_data.index = listing_data["Date"].tolist()
        for rec_date in (c.start + timedelta(days=n) for n in xrange((c.end - c.start).days)):
            idx = next(i for i, d in enumerate(segment_start_dates) if rec_date >= d)
            try:
                d = rec_date.strftime("%Y-%m-%d")
                ac = listing_data.ix[d]['Adj Close']  # listing_data:����Դ,��.csv�ļ�����,DateFrame����
                daily_return = (ac - lastAc)  # һ��һ�ײ��
                # daily_return = (ac - lastAc) / lastAc  # һ��һ�ײ�ֱ�׼��
                # if len(daily_returns) == daily_returns.maxlen:
                #     seq[idx].append(daily_return / np.std(daily_returns))  # һ��һ�ײ�ֱ�׼����,�ٽ������ݳ���Ϊ50�ı�׼���һ��
                # daily_returns.append(daily_return)
                lastAc = ac
                seq[idx].append(daily_return)  # һ��һ�ײ�����ݣ�ԭʼ���������������Ч���ܲ
            except KeyError:
                pass
        return [np.asarray(dat, dtype=np.float32) for dat in seq][::-1]

    if not os.path.exists(c.save_file_diff):
        datasets = divide_data()
        print('Saving in {}'.format(c.save_file_diff))
        np.savez(c.save_file_diff, *datasets)
    else:
        with np.load(c.save_file_diff) as file_load:
            datasets = [file_load['arr_%d' % i] for i in range(len(file_load.files))]
    return datasets


"""seq_iterator():���б���ʽ��ԭʼ����(raw_data)ת���ɾ�����ʽ��ʾ,�����С[batch_size, batch_nums],��������������ݻ��ֳɡ���״�ġ�X-Y�����ݶԡ�,�����ӽ�ģ��ѵ��."""
def seq_iterator(raw_data, batch_size, num_steps,istraining_for_raw_data):
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
    batch_nums = data_len // batch_size  # //:��ȡ��
    data = np.zeros([batch_size, batch_nums], dtype=np.float32)
    for i in range(batch_size):  # ���б���ʽ��ԭʼ����ת���ɾ�����ʽ��ʾ,�����С[batch_size, batch_nums]
        data[i] = raw_data[batch_nums * i:batch_nums * (i + 1)]

    if istraining_for_raw_data:
        epoch_size = (batch_nums - 1) // num_steps  # ȷ��ѵ��ʱ���ݲ���ֻ��һ��
        if epoch_size == 0:  # ����batch_size/num_steps������epoch_size
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    else:
        epoch_size = batch_nums//num_steps

    if epoch_size == 1:
        x = data[:, 0:num_steps]
        y = data[:, 1:num_steps + 1]
        yield (x, y)
    else:  # ���ظ��зֺͲ��ظ��з�
        if istraining_for_raw_data:  # �ظ��з�,��������������ݻ��ֳɡ���״�ġ�X-Y�����ݶԡ�,�����ӽ�ģ��ѵ��
            # for i in range(epoch_size*num_steps-num_steps):  # �÷����������ظ����з�(x,y)���ݶ�,���зֳ�����ġ����ݶԡ�����ģ��ѵ��.��Ԥ����Ľ��Ҳ�����ظ���.
            for i in range(batch_nums - num_steps):
                # if i % 2 == 0:  # ��һλ�з�һ��
                x = data[:, i:(i + num_steps)]  # num_steps:�������ݿ�Ŀ��.
                y = data[:, (i + 1): (i + 1 + num_steps)]
                yield (x, y)
        # yield �����þ��ǰ�һ���������һ�� generator,���� yield �ĺ���������һ����ͨ����,Python�������Ὣ����Ϊһ�� generator
        # һ������ yield �ĺ�������һ�� generator��������ͨ������ͬ������һ�� generator �������������ã�������ִ���κκ������룬
        # ��Ȼִ�������԰�����������ִ�У���ÿִ�е�һ�� yield ���ͻ��жϣ�������һ������ֵ���´�ִ��ʱ�� yield ����һ��������ִ�С�
        # �������ͺ���һ������������ִ�еĹ����б� yield �ж������Σ�ÿ���ж϶���ͨ�� yield ���ص�ǰ�ĵ���ֵ��
        # һ�仰�������������,ʡ�ڴ�.
        else:  # ��ѵ��ʱ�����ظ����зַ���,Ԥ��Ľ�����ظ�.
            for i in range(epoch_size-1):  # �÷��������ظ����з�(x,y)���ݶ�
                x = data[:, i * num_steps:(i + 1) * num_steps]  # num_steps:�������ݿ�Ŀ��.������"(x,y)���ݶ�"�м���
                y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
                yield (x, y)


"""StockLSTM(): LSTMģ��ѵ����װ����"""
class StockLSTM(object):
    """
    This model predicts a 1D sequence of real numbers (here representing daily stock adjusted
    returns normalized by running fixed-length standard deviation) using an LSTM.
    It is regularized using the method in [Zaremba et al 2015]
    http://arxiv.org/pdf/1409.2329v5.pdf
    """

    def __init__(self, is_training, config):  # ������C++�ġ����캯����
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size

        self._input_data = tf.placeholder(tf.float32, [batch_size, num_steps])  # ����batch_size��num_steps������,���������ͬ
        self._targets = tf.placeholder(tf.float32, [batch_size, num_steps])  # placeholder:ѵ��ʱ��Ҫ������ʵ���ݵĲ���

        # lstm_cell = rnn_cell.BasicRNNCell(size)  # ��װ�õ���ͨRNN��Ԫ
        lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=0.0)  # ��װ�õ�LSTM��Ԫ
        if is_training and config.keep_prob < 1:
            lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
        cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)  # ���RNN��Ԫ

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        #  ������RNN_LSTM�㷨���ģ���򵥵����Ժ���y=wx+b(̫����,���Ȳ�����)
        iw = tf.get_variable("input_w", [1, size])
        ib = tf.get_variable("input_b", [size])
        inputs = [tf.nn.xw_plus_b(i_, iw, ib) for i_ in tf.split(1, num_steps, self._input_data)] # split���о��ȷָ��num_steps������(����)
        if is_training and config.keep_prob < 1:
            inputs = [tf.nn.dropout(input_, config.keep_prob) for input_ in inputs]

        outputs, states = rnn.rnn(cell, inputs, initial_state=self._initial_state)
        # c_out = tf.concat(1, outputs)  # outputs:p��m��n; c_out:m �� n*p
        # c_out = tf.concat(0, outputs)  # outputs:p��m��n; c_out:m*p �� n
        rnn_output = tf.reshape(tf.concat(1, outputs), [-1, size])  # [-1, size]:������Ԫ�ظ�������,size=1��200��ʾһ������Ȩ��
        # rnn_output:���õ�����n��200,������n�����(��Ӧn������),ÿ�������1��200��һά������ʾ

        # output����Ԫ���������������(��������Ҫ�Ľ��),������������������ͬ
        self._output = output = tf.nn.xw_plus_b(rnn_output,
                                                tf.get_variable("out_w", [size, 1]),
                                                tf.get_variable("out_b", [1]))

        self._cost = cost = tf.sqrt(tf.reduce_mean((output - tf.reshape(self._targets, [-1]))**2))  # ���������RMSE:ƽ���������ݵ�ʵ��ֵ��Ԥ��ֵ֮���ƫ��
        self._cost_MAPE = cost_MAPE = tf.reduce_mean(tf.abs(output - tf.reshape(self._targets, [-1])) / tf.reshape(self._targets, [-1]))
        self._final_state = states

        if not is_training:  # ��֤/���Ի�����ʵԤ��ʱ,�򲻸���Ȩ��(����ִ����������)
            return
        # ѵ������,���򴫲�,����Ȩ��
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        # optimizer = tf.train.AdamOptimizer(self.lr)  # ���򴫲�,����Ȩ��
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


def main():
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

    """ run_epoch(): Runs the model on the given data.���������ݼ������Լ�����������������ѵ��һ��Ĺ��� Return:������� """
    def run_epoch(session, m, data, eval_op, verbose=False, istraining_for_raw_data=True):
        epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
        # print("epoch_size���ݿ��С��"+epoch_size)
        start_time = time.time()
        costs = 0.0
        costs_MAPE = 0.0
        iters = 0
        state = m.initial_state.eval()
        outputs = []  # �洢ÿһ�ֵļ���(Ԥ��)���y_
        for step, (x, y) in enumerate(seq_iterator(data, m.batch_size, m.num_steps, istraining_for_raw_data)):
            # �������ݼ��ֳ���m����(X,Y)���ݿ顱,����ѵ��m��,�պð��������ݼ�������һ��
            output, cost, cost_MAPE, state, _ = session.run([m.output, m.cost, m.cost_MAPE, m.final_state, eval_op],
                                                 {m.input_data: x, m.targets: y, m.initial_state: state})
            outputs.append(output)
            costs += cost
            costs_MAPE += cost_MAPE
            iters += 1

            print_interval = 20
            if verbose and epoch_size > print_interval and step % (epoch_size // print_interval) == print_interval:
                # ÿ��epochҪ���10��perplexityֵ
                print("%.3f mse: %.8f speed: %.0f ips" % (step * 1.0 / epoch_size, costs / iters, iters * m.batch_size / (time.time() - start_time)))
        return outputs, costs / (iters if iters > 0 else 1), costs_MAPE / (iters if iters > 0 else 1)  # ���ؼ�������Ԥ��ֵ���;��������RMSE

    with tf.Graph().as_default(), tf.Session() as session:
        config = get_config("Medium")  # ���ĳһ�����úõ�ȫ�ֲ���
        pridict_config = get_config("Medium")  # ����ʱһ��һ�����ݵ�Ԥ��,�����ǿ�
        pridict_config.batch_size = 1
        pridict_config.num_steps = 1

        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)  # �������ɾ���������ȷֲ��������ĳ�ʼ����,��ʼ����ز���.
        with tf.variable_scope("model", reuse=None, initializer=initializer):  # variable_scope():���ر����������������.
            m = StockLSTM(is_training=True, config=config)  # ѵ��ģ��,is_trainable=True
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = StockLSTM(is_training=False, config=config)  # �������Ͳ���ģ��,is_training=False
            mtest = StockLSTM(is_training=False, config=config)
            mpridict = StockLSTM(is_training=False, config=pridict_config)

        tf.initialize_all_variables().run()

        diff_train_data, diff_valid_data, diff_test_data = Divide_data()  # ����һά������ʽ���

        for epoch in xrange(config.max_max_epoch):  # max_max_epoch:��������,��������������ѵ��max_max_epoch��
            lr_decay = config.lr_decay ** max(epoch - config.max_epoch, 0.0)  # ѧϰ��˥�����:����С��max epochʱ,lr_decay=1;����max_epochʱ,lr_decay = lr_decay**(epoch-max_epoch)
            m.assign_lr(session, config.learning_rate * lr_decay)  # �ı�ѧϰ��
            cur_lr = session.run(m.lr)

            _tr, rmse, tMAPE = run_epoch(session, m, diff_train_data, m.train_op, verbose=True, istraining_for_raw_data=True)  # ���������RMSE
            _v, vrmse, vMAPE = run_epoch(session, mvalid, diff_valid_data, tf.no_op(), istraining_for_raw_data=True)
            pridict_test, trmse, test_MAPE = run_epoch(session, mtest, diff_test_data, tf.no_op(), istraining_for_raw_data=True)
            print("Epoch: %d - learning rate: %.3f - diffdata: train rmse: %.3f - valid rmse: %.3f - test rmse: %.3f" % (epoch, cur_lr, rmse, vrmse, trmse))
            print("Epoch: %d - learning rate: %.3f - diffdata: train tMAPE: %.3f - valid vMAPE: %.3f - test test_MAPE: %.3f" % (epoch, cur_lr, tMAPE, vMAPE, test_MAPE))

        # ��֤������ģ��ѵ������������,�����ھ���ѵ����õ���ģ�ͺû������ṩ�жϱ�׼��ͬһ��ģ�Ͳ�ͬ�����û����ж�;�����ͬģ�ͺû����жϣ�
        # _v, vrmse = run_epoch(session, mvalid, diff_valid_data, tf.no_op(), istraining_for_raw_data=True)  # tf.no_op():Does nothing. Only useful as a placeholder for control edges.��֤�����������
        # print("Valid rmse: %.3f " % vrmse)

        # ģ��ѵ����ɺ�,�ò��Լ����ݲ���,�ó����Լ����ݵľ������;pridict_train:ÿһ������(x,y)�鶼�ж�Ӧ���������,��seq_iterator()����������(x,y)��Ƭ���ظ�,
        # ��Ԥ���������ظ���,����Ҫɾȥ�ظ���Ԥ����;���seq_iterator()������(x,y)���з������ظ���,��Ԥ����Ҳ�����ظ�,����ɾ��.
        # pridict_test1, trmse1 = run_epoch(session, m, diff_test_data, tf.no_op(), istraining_for_raw_data=True)
        # pridict_test2, trmse2 = run_epoch(session, mtest, diff_test_data, tf.no_op(), istraining_for_raw_data=True)
        # print("Test trmse1: %.3f - trmse: %.3f" % (trmse1, trmse2))

        pridict_test, trmse, teMAPE = run_epoch(session, mpridict, diff_test_data, tf.no_op(),istraining_for_raw_data=False)  # ����ֵ���ظ�Ԥ��
        print("diffdata: Test trmse: %.3f - teMAPE:%.3f " % (trmse, teMAPE))
        # ��ֻ�ԭ����
        with np.load(c.save_file) as file_load:
            train_data, valid_data, test_data = [file_load['arr_%d' % i] for i in range(len(file_load.files))]
        realTest_data = []
        for i, data in enumerate(pridict_test):  # enumerate:������������������ֵ
            realTest_data.append(data[0][0] + test_data[i])
        real_trmse = np.sqrt(np.mean((realTest_data - test_data[1:]) ** 2))  # ���������RMSE:ƽ���������ݵ�ʵ��ֵ��Ԥ��ֵ֮���ƫ��
        real_tMAPE = np.mean(np.abs(realTest_data - test_data[1:])/ test_data[1:])
        print("Test real_trmse: %.3f - real_tMAPE:%.3f " % (real_trmse, real_tMAPE))
        return realTest_data
        # pridict_next1, error = run_epoch(session, mtest, test_data[-100:], tf.no_op(), istraining_for_raw_data=False)
        # pridict_next2, error = run_epoch(session, m, test_data[-100:], tf.no_op(), istraining_for_raw_data=False)
        # print("Next data:", pridict_next1, " Error: %.3f" % error)
if __name__ == '__main__':
    # Data_Visualization()
    start_time = time.time()
    # �����Ա�ͼ����ͳplt
    show_orig_data = listing_data[-100:]['Flow (Veh/Hour)']
    # show_orig_data = listing_data[-100:]['Adj Close']
    show_pric_data = main()
    fig = plt.figure(figsize=(12, 8))  # ����
    ax1 = fig.add_subplot(111)  # ����ϵ
    plt.plot(np.arange(len(show_orig_data)), show_orig_data, color="r")  # plt.plot(x,y)
    plt.plot(np.arange(len(show_pric_data[-100:])), show_pric_data[-100:], color="b")
    xticks_dates = []  # x��Ŀ̶�(ticks)��ʾ����
    [xticks_dates.append(listing_data[7927 + i:7928 + i]["Date"].tolist()[0]) for i in np.arange(20) * 5]

    # ����x��̶��ϵı�ǩ:plt.xticks() ��1����������Ҫ��ʾ��ǩ�ĺ������λ��(int),ÿ��5��λ����ʾ��ǩ,����ʾ20��;
    #  ��2����������Ҫ��ʾ��ǩ��20��λ������Ӧ�ľ�������;��3������rotation: x��̶ȱ�ǩ��ת���ٶ�.
    plt.xticks(np.arange(20) * 5, xticks_dates, rotation=90)
    plt.show()
    end_time = time.time()
    print ("cost time: %.1fs" % (end_time-start_time))
