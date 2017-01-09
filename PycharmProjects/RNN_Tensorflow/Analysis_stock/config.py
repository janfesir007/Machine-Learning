# -*-coding:gbk-*-
# -*-coding:utf-8-*-
import datetime

names_file = 'data/flow_year.csv'
start = datetime.datetime(2014, 1, 2)
end = datetime.datetime(2014, 12, 31)
save_file = 'data/flow_year_{}_{}.npz'.format(datetime.datetime.strftime(start, "%Y%m%d"),
        datetime.datetime.strftime(end, "%Y%m%d"))
save_file_diff = 'data/diff_flow_year_{}_{}.npz'.format(datetime.datetime.strftime(start, "%Y%m%d"),
        datetime.datetime.strftime(end, "%Y%m%d"))
save_file_diff_normalize = 'data/diff_normalize_flow_year_{}_{}.npz'.format(datetime.datetime.strftime(start, "%Y%m%d"),
        datetime.datetime.strftime(end, "%Y%m%d"))
normalize_std_len = 50

# names_file = 'data/dow30.csv'
# start = datetime.datetime(1985, 1, 29)
# end = datetime.datetime(2016, 11, 29)
# save_file = 'data/dow30_{}_{}.npz'.format(datetime.datetime.strftime(start, "%Y%m%d"),
#         datetime.datetime.strftime(end, "%Y%m%d"))
# save_file_diff = 'data/diff_dow30_{}_{}.npz'.format(datetime.datetime.strftime(start, "%Y%m%d"),
#         datetime.datetime.strftime(end, "%Y%m%d"))
# normalize_std_len = 50

class SmallConfig(object):
    """Small config."""
    init_scale = 0.1  # 相关参数的初始值为随机均匀分布，范围是[-init_scale,+init_scale]
    learning_rate = 1.0  # 学习速率,值越大,更快地达到最优值,但要防止过头.在循环次数超过max_epoch以后会逐渐降低
    lr_decay = 0.5  # 学习率衰变
    max_grad_norm = 5  # 用于控制梯度膨胀,如果梯度向量的L2模超过max_grad_norm，则等比例缩小
    max_epoch = 5  # epoch<max_epoch时,lr_decay值=1,epoch>max_epoch时,lr_decay逐渐减小
    max_max_epoch = 10  # 整个大循环次数
    num_layers = 2  # LSTM层数
    num_steps = 10  # 单个数据中,序列的长度;时间点的个数(时间序列的链长)
    batch_size = 30  # 数据块厚度
    hidden_size = 100  # 隐藏层中单元数目
    keep_prob = 1.0  # 用于dropout.每批数据输入时神经网络中的每个单元会以1-keep_prob的概率不工作,可以防止过拟合


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    lr_decay = 0.8
    max_grad_norm = 5
    max_epoch = 0  # epoch<max_epoch时,lr_decay值=1,epoch>max_epoch时,lr_decay逐渐减小
    max_max_epoch = 1
    num_layers = 2
    num_steps = 10
    batch_size = 30
    hidden_size = 20
    keep_prob = 0.8


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 0.01
    lr_decay = 1 / 1.15
    max_grad_norm = 10
    max_epoch = 5  # epoch<max_epoch时,lr_decay值=1,epoch>max_epoch时,lr_decay逐渐减小
    max_max_epoch = 10
    num_layers = 2
    num_steps = 10
    batch_size = 30
    hidden_size = 1500
    keep_prob = 0.35

