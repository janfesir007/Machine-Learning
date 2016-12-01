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
listing_data = pd.read_csv('data/dow30.csv', usecols=['Date', 'Open', "High", "Low", 'Close', 'Volume', "Adj Close"])  # ��������ȥ����һ���ֶ�
# �������Date��ָ��Ϊ����index_col="Date",��Date����һ��������listing_data������һ��ֵ,�������ڱ����,��Ϊ�������,���������б��ʵĲ�ͬ.
# listing_data = pd.read_csv('data/dow30.csv', index_col="Date", usecols=['Date', 'Open', "High", "Low", 'Close', 'Volume', "Adj Close"])
# print(listing_data.head())
# print(listing_data.head().index)

"""ԭʼ���ݲ��ֿ��ӻ�"""
def Data_Visualization():
    # seaborn��ͼ,factorplot()����x:x�����չʾ���ǡ�Data��������Ӧ��ֵ(y��ͬ��);����data:����Դ;x,y����data����Դ
    # sns.factorplot(x='Date', y='Adj Close', data=listing_data[:50], size=6, aspect=2)

    xticks_dates = []  # x��Ŀ̶�(ticks)��ʾ����
    [xticks_dates.append(listing_data[i:i+1]["Date"].tolist()[0]) for i in np.arange(10)*5]

    # plt.xticks() ��1����������Ҫ��ʾ��ǩ�ĺ������λ��(int),ÿ��5��λ����ʾ��ǩ,����ʾ10��; # ��2����������Ҫ��ʾ��ǩ��10��λ������Ӧ�ľ�������;��3������rotation: x��̶ȱ�ǩ��ת���ٶ�.
    xt = plt.xticks(np.arange(10)*5, xticks_dates, rotation=90)
    plt.show()

# ֻȡ��Adj Close����һ��������������Ԥ��,��ת����list����
AdjClose_datalist = listing_data['Adj Close'].tolist()
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
        # lastAc = -1
        # daily_returns = deque(maxlen=c.normalize_std_len)
        listing_data.index = listing_data["Date"].tolist()
        for rec_date in (c.start + timedelta(days=n) for n in xrange((c.end - c.start).days)):
            idx = next(i for i, d in enumerate(segment_start_dates) if rec_date >= d)
            try:
                d = rec_date.strftime("%Y-%m-%d")
                ac = listing_data.ix[d]['Adj Close']  # listing_data:����Դ,��.csv�ļ�����,DateFrame����
                # daily_return = (ac - lastAc) / lastAc  # һ��һ�ײ�ֱ�׼��
                # if len(daily_returns) == daily_returns.maxlen:
                #     seq[idx].append(daily_return / np.std(daily_returns))  # һ��һ�ײ�ֱ�׼����,�ٽ������ݳ���Ϊ50�ı�׼���һ��
                # daily_returns.append(daily_return)
                # lastAc = ac
                seq[idx].append(ac)  # ԭʼ����,����������
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

    epoch_size = (batch_nums - 1) // num_steps

    if epoch_size == 0:  # ����batch_size/num_steps������epoch_size
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    # for i in range(epoch_size):  # ��������������ݻ��ֳɡ���״�ġ�X-Y�����ݶԡ�,�����ӽ�ģ��ѵ��(���ַ����з�(x,y)���ݶ�̫��,������ģ��ѵ��)
    #     x = data[:, i * num_steps:(i + 1) * num_steps]  # num_steps:�������ݿ�Ŀ��.������"(x,y)���ݶ�"�м���(������ѵ������������),Ӱ����ģ��ѵ�����Ⱥ��ٶ�
    #     y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
    for i in range(epoch_size*num_steps-10):  # �÷�������һ���з�(x,y)���ݶԷ�������ѧ,�зֳ���������ݶ�����ѵ��.(��һ�ַ��������ظ����з�,�����������ظ����з�)
        x = data[:, i:(i + num_steps)]
        y = data[:, (i + 1): (i + 1 + num_steps)]
        yield (x, y)
        # yield �����þ��ǰ�һ���������һ�� generator,���� yield �ĺ���������һ����ͨ����,Python�������Ὣ����Ϊһ�� generator
        # һ������ yield �ĺ�������һ�� generator��������ͨ������ͬ������һ�� generator �������������ã�������ִ���κκ������룬
        # ��Ȼִ�������԰�����������ִ�У���ÿִ�е�һ�� yield ���ͻ��жϣ�������һ������ֵ���´�ִ��ʱ�� yield ����һ��������ִ�С�
        # �������ͺ���һ������������ִ�еĹ����б� yield �ж������Σ�ÿ���ж϶���ͨ�� yield ���ص�ǰ�ĵ���ֵ��
        # һ�仰�������������,ʡ�ڴ�.

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

        lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
        if is_training and config.keep_prob < 1:
            lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
        cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

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

        self._cost = cost = tf.reduce_mean(tf.abs(output - tf.reshape(self._targets, [-1])))  # ƽ�����
        self._final_state = states

        if not is_training:  # ��֤/���Ի�����ʵԤ��ʱ,�򲻸���Ȩ��(����ִ����������)
            return
        # ѵ������,���򴫲�,����Ȩ��
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        # optimizer = tf.train.GradientDescentOptimizer(self.lr)
        optimizer = tf.train.AdamOptimizer(self.lr)  # ���򴫲�,����Ȩ��
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

    """ run_epoch(): Runs the model on the given data.���������ݼ������Լ�����������������ѵ��һ��Ĺ��� Return:������� """
    def run_epoch(session, m, data, eval_op, verbose=False):
        epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
        # print("epoch_size���ݿ��С��"+epoch_size)
        start_time = time.time()
        costs = 0.0
        iters = 0
        state = m.initial_state.eval()
        outputs = []  # �洢ÿһ�ֵļ���(Ԥ��)���y_
        for step, (x, y) in enumerate(seq_iterator(data, m.batch_size, m.num_steps)):
            # �������ݼ��ֳ���m����(X,Y)���ݿ顱,����ѵ��m��,�պð��������ݼ�������һ��
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
        return outputs, costs / (iters if iters > 0 else 1)  # ���ؼ�������Ԥ��ֵ���;������

    with tf.Graph().as_default(), tf.Session() as session:
        config = get_config(config_size)
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)  # �������ɾ��о��ȷֲ��������ĳ�ʼ������
        with tf.variable_scope("model", reuse=None, initializer=initializer):  # variable_scope():���ر����������������.
            m = StockLSTM(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mtest = StockLSTM(is_training=False, config=config)

        tf.initialize_all_variables().run()

        train_data, valid_data, test_data = Divide_data()

        for epoch in xrange(num_epochs):  # num_epochs:��������,��������������ѵ��num_epochs��
            lr_decay = config.lr_decay ** max(epoch - num_epochs, 0.0)  # ѧϰ��˥�����
            m.assign_lr(session, config.learning_rate * lr_decay)
            cur_lr = session.run(m.lr)

            _, mse = run_epoch(session, m, train_data, m.train_op, verbose=True)  # Mean square error,ѵ�����������
            _, vmse = run_epoch(session, mtest, valid_data, tf.no_op())  # tf.no_op():Does nothing. Only useful as a placeholder for control edges.��֤���������
            print("Epoch: %d - learning rate: %.3f - train mse: %.3f - valid mse: %.3f" %
                  (epoch, cur_lr, mse, vmse))

        pridict_train, tmse = run_epoch(session, mtest, test_data, tf.no_op())  # ģ��ѵ����ɺ�,�ò��Լ����ݲ���,�ó����Լ����ݵľ������
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