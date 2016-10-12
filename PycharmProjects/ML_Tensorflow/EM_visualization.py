# -*-coding:gbk-*-
# -*-coding:UTF-8-*-
"""
Tensorflow�汾��EM�㷨
�ص㣺����ͼ�е��������������� ������Tensor�������������飬�������Ԫ��ʵ�ֲ��м��㣩
"""
from __future__ import division  # python�С���ȷ���������������ǻ᷵����ʵ���̣����ܲ����������λ��Ǹ�����,���ᶪ�������С���㲿�֡�
from numpy import *
import tensorflow as tf

"""��������һЩ���ڲ��Ե�����"""
# ָ��������˹�ֲ��Ĳ�������ֵ�ͱ�׼��
sigma_1 = 2  # sigma���Ƿ���,�Ǳ�׼��ص㣩
sigma_2 = 4
miu_1 = 25
miu_2 = 30
# ������ȹ���������˹�ֲ���������������ֵ
k = 2
N = 10000
X_change = zeros((1, N))
sum_x_1 = 0
sum_x_2 = 0
sum_fc_1 = 0
sum_fc_2 = 0
count = 0
for i in xrange(N):
    if random.random() > 0.3:  # ʹ�õ���numpyģ���е�random():����0-1֮���ĳ��ʵ��
        X_change[0, i] = random.randn() * sigma_1 + miu_1  # randn():�ӡ���׼��̫�ֲ����Ϸ���һ��ʵ��
        sum_fc_1 = sum_fc_1+((X_change[0, i] - miu_1) ** 2)
        sum_x_1 = sum_x_1+X_change[0, i]
        count += 1
    else:
        X_change[0, i] = random.randn() * sigma_2 + miu_2
        sum_fc_2 = sum_fc_2 + ((X_change[0, i] - miu_2) ** 2)
        sum_x_2 = sum_x_2 + X_change[0, i]
print "����ľ�ֵ1=%.2f,����1=%.2f"%(sum_x_1/count, sum_fc_1/count)
print "����ľ�ֵ2=%.2f,����2=%.2f"%(sum_x_2/(N-count), sum_fc_2/(N-count))
a = zeros([N, 2])
X = array(a+X_change.reshape([N, 1]))  # ��������X��nx2��ʽ�����飬���е�����һ����Ϊ�˺��沢�м���ķ��㣩
# plt.hist(X_change[0, :], 500)
# plt.show()


"""
���������Ѿ���������(���ݼ�)
# �����ɵ�������ʹ��EM�㷨�������ֵmiu�ͱ�׼��sigma
"""
logs_path = 'ML_tensorflow_logs'  # �۲����仯ͼ���·��
# �������ݣ�ռλ��
x = tf.placeholder(tf.float64, [N, 2], name='InputData')  # �������ݣ�ռλ����
# ����
pro_pi = tf.Variable(array([[0.5, 0.5]]), name='pro_pi')
sigma = tf.Variable(array([[2.0, 4.0]]), name='sigma')
miu = tf.Variable(array([[35.0, 5.0]]), name='miu')
Expectations = tf.Variable(tf.zeros((N, k)))  # ��¼ÿ���������Ը�����˹�ֲ��ĸ���

with tf.name_scope('E-step'):
    # ����1����������������ÿ���������Ը�����˹�ֲ��ĸ���
    # �����ĸdenominator
    denominator = (pro_pi * (1 / (sqrt(2 * pi) * sigma))) * tf.exp((- (x - miu) ** 2) / (2 * sigma ** 2))
    denominator = tf.reshape(tf.reduce_sum(denominator, 1), (N, 1))
    # �������numerator
    numerator = pro_pi * (1 / (sqrt(2 * pi) * sigma)) * tf.exp(-((x - miu) ** 2) / (2 * sigma ** 2))
    Expectations = numerator / denominator
with tf.name_scope('M_step'):
    # ����2�������������
    Nk = tf.reduce_sum(Expectations, 0)  # ������������ĳһ��˹�ֲ��ĸ���֮��(�����)
    gama = tf.reduce_sum(Expectations * x, 0)  # ������������ĳһ��˹�ֲ�������ֵ
    sum_fc = tf.reduce_sum(Expectations * ((x - miu) ** 2), 0)  # �ܷ���
    # pro_pi = Nk / N  # �ǳ��ص㣺������ֵ,����pro_pi��ֵ��ѭ��ִ��run(pro_pi)��ֻ���һ�μ�����ֵһ��,������2��֮��ֵ�Ͳ��ᷢ���ı�
    # miu = gama / Nk  # ��ֵ
    # sigma = tf.sqrt(sum_fc / Nk)  # ��׼��=������ţ��ص㣩
    pro_pi = tf.assign(pro_pi, tf.reshape(Nk/N, (1, 2)))  # assign():���ڶ���������ֵ����һ������,�����������ܱ�֤ÿ��ѭ��ִ��run(pro_pi)��,ֵ�ᱻ������ֵ
    miu = tf.assign(miu, tf.reshape(gama/Nk, (1, 2)))  # gama/Nk:��ֵ
    sigma = tf.assign(sigma, tf.reshape(tf.sqrt(sum_fc/Nk), (1, 2)))   # sqrt(sum_fc/Nk):��׼��=������ţ��ص㣩

""" ���ӻ��� ����һ��summary���tensor """
tf.scalar_summary("0-miu", miu[0, 0])  # Create a summary to monitor miu tensor
tf.scalar_summary("1_miu", miu[0, 1])
tf.scalar_summary("0-sigma", sigma[0, 0])
tf.scalar_summary("1-sigma", sigma[0, 1])
merged_summary_op = tf.merge_all_summaries()  # Merge all summaries into a single op

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # op to write logs to Tensorboard
    summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
    for step in xrange(1000):  # ���õ�������
        end_pi, end_miu, end_sigma, summary = sess.run([pro_pi, miu, sigma, merged_summary_op], feed_dict={x: X})
        # ��ÿһ����������loss/accuracyд���¼���־���Ա���Tensorboardչʾ
        summary_writer.add_summary(summary, global_step=step)
        print "pi=", end_pi, "miu=", end_miu, "sigma", end_sigma
