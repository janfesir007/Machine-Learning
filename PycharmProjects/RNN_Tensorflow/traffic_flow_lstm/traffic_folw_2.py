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

# ����ԭʼ����,������listing_data��DataFrame����,������Ĭ�ϵġ�int������
listing_data = pd.read_csv('data/flow_year.csv', usecols=['Hour', 'Flow (Veh/Hour)', "# Lane Points", "% Observed"])  # ��������ȥ����һ���ֶ�
# �������Date��ָ��Ϊ����index_col="Date",��Date����һ��������listing_data������һ��ֵ,�������ڱ����,��Ϊ�������,���������б��ʵĲ�ͬ.
# listing_data = pd.read_csv('data/dow30.csv', index_col="Date", usecols=['Date', 'Open', "High", "Low", 'Close', 'Volume', "Adj Close"])
# print(listing_data.head())
# print(listing_data.head().index)

"""ԭʼ���ݲ��ֿ��ӻ�"""
def Data_Visualization():
    # ��һ�ֻ�������ͳplt
    show_data = listing_data[-100:]['Flow (Veh/Hour)']
    fig = plt.figure(figsize=(12, 8))  # ����
    ax1 = fig.add_subplot(111)  # ����ϵ
    plt.plot(np.arange(len(show_data)), show_data, marker="*", markersize=9.0, linewidth=.5, color="b")  # plt.plot(x,y);maker�����,*(������״)
    # plt.plot(np.arange(len(show_data)), show_data, linewidth=.5, linestyle='-', color="b")
    xticks_dates = []  # x��Ŀ̶�(ticks)��ʾ����
    [xticks_dates.append(listing_data[5900 + i:5901 + i]["Hour"].tolist()[0]) for i in np.arange(20) * 5]

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


"""
    Divide_data(): �����ݷָ�ɡ�ѵ����/��֤��/���Լ�������������,�б�����
"""
def Divide_data(file):
    '''
    ��������ļ��Ѵ��������,��������ļ��Ļ��ֲ������ֺ���ļ����б���.
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
                ac = listing_data.ix[i]['Flow (Veh/Hour)']  # listing_data:����Դ,��.csv�ļ�����,DateFrame����
                daily_return = (ac - lastAc)  # һ��һ�ײ��
                # daily_return = (ac - lastAc) / lastAc  # һ��һ�ײ�ֱ�׼��;  ��ԭ��ʽ��ac = lastAc*daily_return+lastAc
                # if len(daily_returns) == daily_returns.maxlen:
                #     seq[idx].append(daily_return / np.std(daily_returns))  # һ��һ�ײ�ֱ�׼����,�ٽ������ݳ���Ϊ50�ı�׼���һ��
                # daily_returns.append(daily_return)
                lastAc = ac
                seq[0].append(daily_return)  # һ��һ�ײ�����ݣ�ԭʼ���������������Ч���ܲ
                # seq[0].append(ac)  # ԭʼ����
            except KeyError:
                pass
        for i in range(int(len(listing_data)*0.1)):
            try:
                ac = listing_data.ix[i+4800]['Flow (Veh/Hour)']  # listing_data:����Դ,��.csv�ļ�����,DateFrame����
                daily_return = (ac - lastAc)  # һ��һ�ײ��
                # daily_return = (ac - lastAc) / lastAc  # һ��һ�ײ�ֱ�׼��
                # if len(daily_returns) == daily_returns.maxlen:
                #     seq[idx].append(daily_return / np.std(daily_returns))  # һ��һ�ײ�ֱ�׼����,�ٽ������ݳ���Ϊ50�ı�׼���һ��
                # daily_returns.append(daily_return)
                lastAc = ac
                seq[1].append(daily_return)  # һ��һ�ײ������
                # seq[1].append(ac)  # ԭʼ����
            except KeyError:
                pass
        for i in range(int(len(listing_data)*0.1)):
            try:
                ac = listing_data.ix[i+5400]['Flow (Veh/Hour)']  # listing_data:����Դ,��.csv�ļ�����,DateFrame����
                daily_return = (ac - lastAc)  # һ��һ�ײ��
                # daily_return = (ac - lastAc) / lastAc  # һ��һ�ײ�ֱ�׼��
                # if len(daily_returns) == daily_returns.maxlen:
                #     seq[idx].append(daily_return / np.std(daily_returns))  # һ��һ�ײ�ֱ�׼����,�ٽ������ݳ���Ϊ50�ı�׼���һ��
                # daily_returns.append(daily_return)
                lastAc = ac
                seq[2].append(daily_return)  # һ��һ�ײ������
                # seq[2].append(ac)  # ԭʼ����
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


"""seq_iterator():���б���ʽ��ԭʼ����(raw_data)ת���ɾ�����ʽ��ʾ,�����С[batch_size, batch_nums],��������������ݻ��ֳɡ���״�ġ�X-Y�����ݶԡ�,�����ӽ�ģ��ѵ��."""
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
    batch_nums = data_len // batch_size  # //:��ȡ��
    data = np.zeros([batch_size, batch_nums], dtype=np.float32)
    for i in range(batch_size):  # ���б���ʽ��ԭʼ����ת���ɾ�����ʽ��ʾ,�����С[batch_size, batch_nums]
        data[i] = raw_data[batch_nums * i:batch_nums * (i + 1)]

    epoch_size = (batch_nums - 1) // num_steps  # ȷ��ѵ��ʱ���ݲ���ֻ��һ��
    if epoch_size == 0:  # ����batch_size/num_steps������epoch_size
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    if epoch_size == 1:
        x = data[:, 0:num_steps]
        y = data[:, 1:num_steps + 1]  # n�����ж�Ӧn��Ԥ����
        # y = data[:, num_steps:num_steps + 1]  # n�����ж�Ӧ1��Ԥ����
        yield (x, y)
    else:   # �����з�,��������������ݻ��ֳɡ���״�ġ�X-Y�����ݶԡ�,�����ӽ�ģ��ѵ��
        for i in range(batch_nums - num_steps):  # ����һλ�����з�
            x = data[:, i:(i + num_steps)]  # num_steps:�������ݿ�Ŀ��.
            y = data[:, i+1:(i + num_steps + 1)]  # ����Ϊnum_steps����������x,��ӦԤ��num_steps�����
            # y = data[:, (i + num_steps):(i + num_steps + 1)]  # ����Ϊnum_steps����������x,��ӦԤ���һ��y
            yield (x, y)
        # yield �����þ��ǰ�һ���������һ�� generator,���� yield �ĺ���������һ����ͨ����,Python�������Ὣ����Ϊһ�� generator
        # һ������ yield �ĺ�������һ�� generator��������ͨ������ͬ������һ�� generator �������������ã�������ִ���κκ������룬
        # ��Ȼִ�������԰�����������ִ�У���ÿִ�е�һ�� yield ���ͻ��жϣ�������һ������ֵ���´�ִ��ʱ�� yield ����һ��������ִ�С�
        # �������ͺ���һ������������ִ�еĹ����б� yield �ж�������,ÿ���ж϶���ͨ�� yield ���ص�ǰ�ĵ���ֵ��
        # һ�仰�������������,ʡ�ڴ�.


class StockLSTM(object):  # StockLSTM(): LSTMģ��ѵ����װ����
    """
    This model predicts a 1D sequence of real numbers (here representing daily stock adjusted
    returns normalized by running fixed-length standard deviation) using an LSTM.
    It is regularized using the method in [Zaremba et al 2015]
    http://arxiv.org/pdf/1409.2329v5.pdf
    """
    def __init__(self, is_training, config):  # ������C++�ġ����캯����
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size  # ÿ�����ز�ĵ�Ԫ�����������������

        self._input_data = tf.placeholder(tf.float32, [batch_size, num_steps])  # ����batch_size��num_steps������,���batch_size��1��
        self._targets = tf.placeholder(tf.float32, [batch_size, num_steps])  # placeholder:ѵ��ʱ��Ҫ������ʵ���ݵĲ���,ÿһ������Ϊnum_steps������Ԥ���һ��ֵ�����һ����
        # self._targets = tf.placeholder(tf.float32, [batch_size, 1])

        # lstm_cell = tf.nn.rnn_cell.BasicRNNCell(size)  # ��װ�õ���ͨRNN��Ԫ
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=1.0)  # ��װ�õ�LSTM��Ԫ
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)  # ���RNN��Ԫ

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        #  ������RNN_LSTM�㷨���ģ���򵥵����Ժ���y=wx+b(̫����,���Ȳ�����)
        iw = tf.get_variable("input_w", [1, size])
        ib = tf.get_variable("input_b", [size])
        inputs = [tf.nn.xw_plus_b(i_, iw, ib) for i_ in tf.split(1, num_steps, self._input_data)] # split���о��ȷָ��"num_steps��"����(����)
        # inputs = [i_ for i_ in tf.split(1, num_steps, self._input_data)]  # ������wx+b����,ֱ����Ϊ����
        if is_training and config.keep_prob < 1:
            inputs = [tf.nn.dropout(input_, config.keep_prob) for input_ in inputs]

        outputs, states = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)  # outputs:[num_steps,batch_size,size],num_steps��[batch_size,size]���󣺼�ÿ��step��������
        # c_out = tf.concat(1, outputs)  # outputs:p��m��n����; c_out:[m �� n*p]:m��,n*p�о���
        # c_out = tf.concat(0, outputs)  # outputs:p��m��n; c_out:m*p �� n
        # rnn_output = tf.reshape(tf.concat(1, outputs), [-1, size])  # �ò�����Ŀ�ģ���ÿһ��step��������������.[-1, size]:������Ԫ�ظ�������,size=hidden_size
        # rnn_output:���õ�����n��size,������n�����(��Ӧn������),ÿ�������1��size��һά������ʾ.

        # output����Ԫ���������������(��������Ҫ�Ľ��)
        self._output = output = tf.nn.xw_plus_b(outputs[-1],  # ��num_steps��������,ֻȡ���һ��step�Ľ��
                                                tf.get_variable("out_w", [size, 1]),
                                                tf.get_variable("out_b", [1]))

        self._cost_RMSE = cost_RMSE = tf.sqrt(tf.reduce_mean((output - self._targets[:, num_steps-1:]) ** 2))  # ���������RMSE:ƽ���������ݵ�ʵ��ֵ��Ԥ��ֵ֮���ƫ��
        self._cost_MAPE = cost_MAPE = tf.reduce_mean(tf.abs((output - self._targets[:, num_steps-1:])/(self._targets[:, num_steps-1:])))
        self._final_state = states

        if not is_training:  # ��֤/���Ի�����ʵԤ��ʱ,�򲻸���Ȩ��(����ִ����������)
            return
        # ѵ������,���򴫲�,����Ȩ��
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost_RMSE, tvars), config.max_grad_norm)
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


def main(data_format):  # data_format:���ݸ�ʽ(ԭʼ����/��ֺ�����/��ֱ�׼��������)
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
    def run_epoch(session, m, data, eval_op, verbose=False):
        epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
        # print("epoch_size���ݿ��С��"+epoch_size)
        start_time = time.time()
        costs_RMSE = 0.0
        costs_MAPE = 0.0
        iters = 0
        iters1 = 0
        # state = m.initial_state.eval()
        state = session.run(m.initial_state)
        outputs = []  # �洢ÿһ�ֵļ���(Ԥ��)���y_
        for step, (x, y) in enumerate(seq_iterator(data, m.batch_size, m.num_steps)):
            # �������ݼ��ֳ���m����(X,Y)���ݿ顱,����ѵ��m��,�պð��������ݼ�������һ��
            output, cost_RMSE, cost_MAPE, final_state, _ = session.run([m.output, m.cost_RMSE, m.cost_MAPE, m.final_state, eval_op],
                                                 {m.input_data: x, m.targets: y, m.initial_state: state})
            outputs.append(output)
            costs_RMSE += cost_RMSE
            if cost_MAPE!=float("inf"):  # ȥ����ĸΪ0�����
                costs_MAPE += cost_MAPE
                iters1 += 1
            iters += 1
            print_interval = 20
            if verbose and epoch_size > print_interval and step % (epoch_size // print_interval) == print_interval:
                # ÿ��epochҪ���10��perplexityֵ
                print("%.3f mse: %.8f speed: %.0f ips" % (step * 1.0 / epoch_size, costs_RMSE / iters, iters * m.batch_size / (time.time() - start_time)))
        return outputs, costs_RMSE / (iters if iters > 0 else 1), costs_MAPE / (iters1 if iters1 > 0 else 1)  # ���ؼ�������Ԥ��ֵ���;��������RMSE��������MAPE

    with tf.Graph().as_default(), tf.Session() as session:
        config = get_config("Medium")  # ���ĳһ�����úõ�ȫ�ֲ���
        pridict_config = get_config("Medium")  # ����ʱһ��һ�����ݵ�Ԥ��,�����ǿ�
        pridict_config.batch_size = 1
        # pridict_config.num_steps = 1

        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)  # �������ɾ���������ȷֲ��������ĳ�ʼ����,��ʼ����ز���.
        with tf.variable_scope("model", reuse=None, initializer=initializer):  # variable_scope():���ر����������������.
            m = StockLSTM(is_training=True, config=config)  # ѵ��ģ��,is_trainable=True
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            # mvalid = StockLSTM(is_training=False, config=config)  # �������Ͳ���ģ��,is_training=False
            mpridict = StockLSTM(is_training=False, config=pridict_config)

        tf.initialize_all_variables().run()

        if data_format == "raw":  # ԭʼ����
            train_data, valid_data, test_data = Divide_data(c.save_file)  # �ָ����ִ��������ݼ�:����һά����(narray)��ʽ���
        elif data_format == "diff":
            train_data, valid_data, test_data = Divide_data(c.save_file_diff)
        elif data_format == "diff_normalize":
            train_data, valid_data, test_data = Divide_data(c.save_file_diff_normalize)
        else:
            print ("����Դ�����ڣ�")

        valid_test_data = np.array(list(valid_data) + list(test_data))  # ����֤���ǲ�������Ҳ��Ϊ���Լ�����
        for epoch in xrange(config.max_max_epoch):  # max_max_epoch:��������,��������������ѵ��max_max_epoch��
            lr_decay = config.lr_decay ** max(epoch - config.max_epoch, 0.0)  # ѧϰ��˥�����:����С��max epochʱ,lr_decay=1;����max_epochʱ,lr_decay = lr_decay**(epoch-max_epoch)
            m.assign_lr(session, config.learning_rate * lr_decay)  # �ı�ѧϰ��
            cur_lr = session.run(m.lr)
            num_steps = config.num_steps

            _train, RMSE, MAPE = run_epoch(session, m, train_data, m.train_op, verbose=True)  # m/m.train_op�����Ż�����,ÿ����һ����Ż�һ�Σ�

            # ��֤������ģ��ѵ������������,�����ھ���ѵ����õ���ģ�ͺû������ṩ�жϱ�׼��ͬһ��ģ�Ͳ�ͬ�����û����ж�;�����ͬģ�ͺû����жϣ�
            # _valid, vRMSE, vMAPE = run_epoch(session, mvalid, valid_data, tf.no_op())  # tf.no_op():Does nothing. Only useful as a placeholder for control edges.
            print("Epoch: %d - learning rate: %.3f - data: train RMSE: %.3f " % (epoch, cur_lr, RMSE))
            print("Epoch: %d - learning rate: %.3f - data: train MAPE: %.3f " % (epoch, cur_lr, MAPE))

        # ģ��ѵ����ɺ�,�ò��Լ����ݲ���,����ʱbatch_size����Ϊ1,���Ԥ��test_data�����"len(test_data)-num_steps"������;
        # pridict_test_onebyone, tRMSE, tMAPE = run_epoch(session, mpridict, test_data, tf.no_op())
        pridict_test_onebyone, tRMSE, tMAPE = run_epoch(session, mpridict, valid_test_data, tf.no_op())  # ����֤���ǲ�������Ҳ��Ϊ���Լ�����
        print("Test_onebyone tRMSE: %.3f - tMAPE:%.3f " % (tRMSE, tMAPE))

    with np.load(c.save_file) as file_load:  # ����ԭʼ����
        orig_train_data, orig_valid_data, orig_test_data = [file_load['arr_%d' % i] for i in range(len(file_load.files))]
    orig_valid_test_data = np.array(list(orig_valid_data) + list(orig_test_data))  # ����֤���ǲ�������Ҳ��Ϊ���Լ�����
    priTest_data = []   # ��ԭ���Ԥ��ֵ.Ԥ������"len(test_data)-num_steps"������,�����
    priTest_data1 = []  # δ���л�ԭ������Ԥ������
    for i, data in enumerate(pridict_test_onebyone):  # enumerate:������������������ֵ
        priTest_data1.append(data[0][0])
        if data_format == "raw":  # ԭʼ����,û������ִ���
            priTest_data.append(data[0][0])
        elif data_format == "diff":  # "���"��Ļ�ԭ����,��ԭ��ʽ��orig_x = pridict + orig_last_X
            priTest_data.append(data[0][0] + orig_valid_test_data[i + num_steps - 1])  # ��ԭ
        elif data_format == "diff_normalize":
            priTest_data.append((data[0][0]+1)*orig_valid_test_data[i + num_steps - 1])
        else:
            print ("���ݻ�ԭʱ����Դ�����ڣ�")

    pri_tRMSE = np.sqrt(np.mean((np.array(priTest_data) - np.array(orig_valid_test_data[num_steps:])) ** 2))  # ���������RMSE:ƽ���������ݵ�ʵ��ֵ��Ԥ��ֵ֮���ƫ��
    pri_tMAPE = np.mean(np.abs(np.array(priTest_data) - np.array(orig_valid_test_data[num_steps:])) / np.array(orig_valid_test_data[num_steps:]))
    print("TestData pri_tRMSE: %.3f - pri_tMAPE: %.3f" % (pri_tRMSE, pri_tMAPE))

    # # "��ֱ�׼��"��Ļ�ԭ����,��ԭ��ʽ��orig_x = orig_last_X*pridict + orig_last_X
    # with np.load(c.save_file) as file_load:  # ����ԭʼ����
    #     orig_train_data, orig_valid_data, orig_test_data = [file_load['arr_%d' % i] for i in range(len(file_load.files))]
    # # priTest_data = [(pridict_test_onebyone[0][0][0]*valid_data[-1]+valid_data[-1])]  # �׸�Ԥ��ֵ
    # priTest_data = []  # Ԥ������"len(test_data)-num_steps"������,�����
    # for i, data in enumerate(pridict_test_onebyone):  # enumerate:������������������ֵ
    #     priTest_data.append(data[0][0]*orig_test_data[i+num_steps-1] + orig_test_data[i+num_steps-1])  # ��ԭ
    # pri_tRMSE = np.sqrt(np.mean((np.array(priTest_data) - np.array(orig_test_data[num_steps:])) ** 2))  # ���������RMSE:ƽ���������ݵ�ʵ��ֵ��Ԥ��ֵ֮���ƫ��
    # pri_tMAPE = np.mean(np.abs(np.array(priTest_data) - np.array(orig_test_data[num_steps:]))/np.array(orig_test_data[num_steps:]))
    # print("TestData pri_tRMSE: %.3f - pri_tMAPE: %.3f" % (pri_tRMSE, pri_tMAPE))
    return priTest_data1, priTest_data

if __name__ == '__main__':
    # Data_Visualization()
    start_time = time.time()
    # �����Ա�ͼ����ͳplt
    show_lenth = 300
    data_format = "raw"  # �������ƣ�raw/diff/diff_normalize:���ִ���ԭ���ݷ�ʽ
    process_data = []
    orig_data = listing_data[-show_lenth:]['Flow (Veh/Hour)']
    a, b = main(data_format)  # ���ú�������������,�õ�Ԥ����,�б�����
    pric_process_data, pric_orig_data = a[-show_lenth:], b[-show_lenth:]
    if data_format=="raw":
        process_data = orig_data
    if data_format=="diff":
        with np.load(c.save_file_diff) as file_load:  # ����ԭʼ����
            orig_train_data, orig_valid_data, orig_test_data = [file_load['arr_%d' % i] for i in range(len(file_load.files))]
        orig_valid_test_data = np.array(list(orig_valid_data) + list(orig_test_data))  # ����֤���ǲ�������Ҳ��Ϊ���Լ�����
        process_data = orig_valid_test_data[-show_lenth:]
    if data_format=="diff_normalize":
        with np.load(c.save_file_diff_normalize) as file_load:  # ����ԭʼ����
            orig_train_data, orig_valid_data, orig_test_data = [file_load['arr_%d' % i] for i in range(len(file_load.files))]
        orig_valid_test_data = np.array(list(orig_valid_data) + list(orig_test_data))  # ����֤���ǲ�������Ҳ��Ϊ���Լ�����
        process_data = orig_valid_test_data[-show_lenth:]
    fig = plt.figure(figsize=(12, 8))  # ����
    ax1 = fig.add_subplot(211)  # ����ϵ
    ax2 = fig.add_subplot(212)
    ax1.plot(np.arange(show_lenth), process_data, marker="*", ms=6.0, linewidth=0.5, color="b", label=data_format+"_data")  # plt.plot(x,y);ms=markersize
    ax1.plot(np.arange(show_lenth), pric_process_data, marker="o", ms=6.0, linewidth=0.5, color="r", label=data_format+'_data_pridict')
    ax2.plot(np.arange(show_lenth), orig_data, marker="*", ms=6.0, linewidth=0.5, color="b", label='raw_data')  # plt.plot(x,y);ms=markersize
    ax2.plot(np.arange(show_lenth), pric_orig_data, marker="o", ms=6.0, linewidth=0.5, color="r", label='raw_data_pridict')
    xticks_dates = []  # x��Ŀ̶�(ticks)��ʾ����
    [xticks_dates.append(listing_data[int(len(listing_data))-show_lenth + i:int(len(listing_data))-show_lenth+1+i]["Hour"].tolist()[0]) for i in np.arange(20) * int(show_lenth/20)]

    # ����x��̶��ϵı�ǩ:plt.xticks() ��1����������Ҫ��ʾ��ǩ�ĺ������λ��(int),ÿ��5��λ����ʾ��ǩ,����ʾ20��;
    #  ��2����������Ҫ��ʾ��ǩ��20��λ������Ӧ�ľ�������;��3������rotation: x��̶ȱ�ǩ��ת���ٶ�.
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