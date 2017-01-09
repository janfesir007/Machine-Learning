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
    init_scale = 0.1  # ��ز����ĳ�ʼֵΪ������ȷֲ�����Χ��[-init_scale,+init_scale]
    learning_rate = 1.0  # ѧϰ����,ֵԽ��,����شﵽ����ֵ,��Ҫ��ֹ��ͷ.��ѭ����������max_epoch�Ժ���𽥽���
    lr_decay = 0.5  # ѧϰ��˥��
    max_grad_norm = 5  # ���ڿ����ݶ�����,����ݶ�������L2ģ����max_grad_norm����ȱ�����С
    max_epoch = 5  # epoch<max_epochʱ,lr_decayֵ=1,epoch>max_epochʱ,lr_decay�𽥼�С
    max_max_epoch = 10  # ������ѭ������
    num_layers = 2  # LSTM����
    num_steps = 10  # ����������,���еĳ���;ʱ���ĸ���(ʱ�����е�����)
    batch_size = 30  # ���ݿ���
    hidden_size = 100  # ���ز��е�Ԫ��Ŀ
    keep_prob = 1.0  # ����dropout.ÿ����������ʱ�������е�ÿ����Ԫ����1-keep_prob�ĸ��ʲ�����,���Է�ֹ�����


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    lr_decay = 0.8
    max_grad_norm = 5
    max_epoch = 0  # epoch<max_epochʱ,lr_decayֵ=1,epoch>max_epochʱ,lr_decay�𽥼�С
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
    max_epoch = 5  # epoch<max_epochʱ,lr_decayֵ=1,epoch>max_epochʱ,lr_decay�𽥼�С
    max_max_epoch = 10
    num_layers = 2
    num_steps = 10
    batch_size = 30
    hidden_size = 1500
    keep_prob = 0.35

