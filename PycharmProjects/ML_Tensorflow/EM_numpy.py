# -*-coding:gbk-*-
# -*-coding:UTF-8-*-
"""
“数组并行计算”的EM算法版本
数组，数组里的元素实现并行计算（同时计算）
"""
from __future__ import division  # python中“精确除法”：除法总是会返回真实的商，不管操作数是整形还是浮点型,不会丢掉结果的小数点部分。
from numpy import *
import time
import matplotlib.pyplot as plt
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

#初始化
pro_pi = array([[0.5, 0.5]])
sigma = array([[2.0, 4.0]])
miu = array([[35.0, 5.0]])
Expectations = zeros((N, k))  # 记录每个样本来自各个高斯分布的概率

time_str = time.time()
for step in xrange(1000):  # 设置迭代次数
    # 步骤1，计算期望，计算每个样本来自各个高斯分布的概率
    # 计算分母denominator
    denominator = (pro_pi * (1 / (sqrt(2 * pi) * sigma))) * exp((- (X - miu) ** 2) / (2 * sigma ** 2))
    denominator = reshape(sum(denominator, 1), (N, 1))
    # 计算分子numerator
    numerator = pro_pi * (1 / (sqrt(2 * pi) * sigma)) * exp(-((X - miu) ** 2) / (2 * sigma ** 2))
    Expectations = numerator / denominator

    # 步骤2，求期望的最大
    Nk = sum(Expectations, 0)  # 所有样本来自某一高斯分布的概率之和(列相加)
    gama = sum(Expectations * X, 0)  # 所有样本来自某一高斯分布的期望值
    sum_fc = sum(Expectations * ((X - miu) ** 2), 0)  # 总方差
    pro_pi = Nk / N
    miu = gama / Nk  # 均值
    sigma = sqrt(sum_fc / Nk)  # 标准差=方差开根号（重点）
    print "pi=", pro_pi, "miu=", miu, "sigma", sigma
time_end = time.time()
print "循环1000次，用时：%d秒"% (time_end-time_str)  # 循环1000次，用时：5秒

