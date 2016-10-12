# -*-coding:gbk-*-
# -*-coding:UTF-8-*-
"""
Tensorflow版本的EM算法
特点：计算图中的所有数据类型是 “张量Tensor”（类似于数组，数组里的元素实现并行计算）
"""
from __future__ import division  # python中“精确除法”：除法总是会返回真实的商，不管操作数是整形还是浮点型,不会丢掉结果的小数点部分。
from numpy import *
import tensorflow as tf

"""首先生成一些用于测试的样本"""
# 指定两个高斯分布的参数：均值和标准差
sigma_1 = 2  # sigma不是方差,是标准差（重点）
sigma_2 = 4
miu_1 = 25
miu_2 = 30
# 随机均匀构造两个高斯分布，用于生成样本值
k = 2
N = 10000
X_change = zeros((1, N))
sum_x_1 = 0
sum_x_2 = 0
sum_fc_1 = 0
sum_fc_2 = 0
count = 0
for i in xrange(N):
    if random.random() > 0.3:  # 使用的是numpy模块中的random():返回0-1之间的某个实数
        X_change[0, i] = random.randn() * sigma_1 + miu_1  # randn():从“标准正太分布”上返回一个实数
        sum_fc_1 = sum_fc_1+((X_change[0, i] - miu_1) ** 2)
        sum_x_1 = sum_x_1+X_change[0, i]
        count += 1
    else:
        X_change[0, i] = random.randn() * sigma_2 + miu_2
        sum_fc_2 = sum_fc_2 + ((X_change[0, i] - miu_2) ** 2)
        sum_x_2 = sum_x_2 + X_change[0, i]
print "构造的均值1=%.2f,方差1=%.2f"%(sum_x_1/count, sum_fc_1/count)
print "构造的均值2=%.2f,方差2=%.2f"%(sum_x_2/(N-count), sum_fc_2/(N-count))
a = zeros([N, 2])
X = array(a+X_change.reshape([N, 1]))  # 输入数据X是nx2形式的数组，两列的数据一样（为了后面并行计算的方便）
# plt.hist(X_change[0, :], 500)
# plt.show()


"""
上述步骤已经生成样本(数据集)
# 对生成的样本，使用EM算法计算其均值miu和标准差sigma
"""
logs_path = 'ML_tensorflow_logs'  # 观测量变化图存放路径
# 输入数据，占位符
x = tf.placeholder(tf.float64, [N, 2], name='InputData')  # 输入数据（占位符）
# 变量
pro_pi = tf.Variable(array([[0.5, 0.5]]), name='pro_pi')
sigma = tf.Variable(array([[2.0, 4.0]]), name='sigma')
miu = tf.Variable(array([[35.0, 5.0]]), name='miu')
Expectations = tf.Variable(tf.zeros((N, k)))  # 记录每个样本来自各个高斯分布的概率

with tf.name_scope('E-step'):
    # 步骤1，计算期望，计算每个样本来自各个高斯分布的概率
    # 计算分母denominator
    denominator = (pro_pi * (1 / (sqrt(2 * pi) * sigma))) * tf.exp((- (x - miu) ** 2) / (2 * sigma ** 2))
    denominator = tf.reshape(tf.reduce_sum(denominator, 1), (N, 1))
    # 计算分子numerator
    numerator = pro_pi * (1 / (sqrt(2 * pi) * sigma)) * tf.exp(-((x - miu) ** 2) / (2 * sigma ** 2))
    Expectations = numerator / denominator
with tf.name_scope('M_step'):
    # 步骤2，求期望的最大
    Nk = tf.reduce_sum(Expectations, 0)  # 所有样本来自某一高斯分布的概率之和(列相加)
    gama = tf.reduce_sum(Expectations * x, 0)  # 所有样本来自某一高斯分布的期望值
    sum_fc = tf.reduce_sum(Expectations * ((x - miu) ** 2), 0)  # 总方差
    # pro_pi = Nk / N  # 非常重点：这样赋值,变量pro_pi的值在循环执行run(pro_pi)后只与第一次计算后的值一致,即：第2次之后值就不会发生改变
    # miu = gama / Nk  # 均值
    # sigma = tf.sqrt(sum_fc / Nk)  # 标准差=方差开根号（重点）
    pro_pi = tf.assign(pro_pi, tf.reshape(Nk/N, (1, 2)))  # assign():将第二个参数赋值给第一个参数,这样操作才能保证每次循环执行run(pro_pi)后,值会被赋予新值
    miu = tf.assign(miu, tf.reshape(gama/Nk, (1, 2)))  # gama/Nk:均值
    sigma = tf.assign(sigma, tf.reshape(tf.sqrt(sum_fc/Nk), (1, 2)))   # sqrt(sum_fc/Nk):标准差=方差开根号（重点）

""" 可视化： 创建一个summary监控tensor """
tf.scalar_summary("0-miu", miu[0, 0])  # Create a summary to monitor miu tensor
tf.scalar_summary("1_miu", miu[0, 1])
tf.scalar_summary("0-sigma", sigma[0, 0])
tf.scalar_summary("1-sigma", sigma[0, 1])
merged_summary_op = tf.merge_all_summaries()  # Merge all summaries into a single op

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # op to write logs to Tensorboard
    summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
    for step in xrange(1000):  # 设置迭代次数
        end_pi, end_miu, end_sigma, summary = sess.run([pro_pi, miu, sigma, merged_summary_op], feed_dict={x: X})
        # 将每一迭代产生的loss/accuracy写入事件日志，以便在Tensorboard展示
        summary_writer.add_summary(summary, global_step=step)
        print "pi=", end_pi, "miu=", end_miu, "sigma", end_sigma
