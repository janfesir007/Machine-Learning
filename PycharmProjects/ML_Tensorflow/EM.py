# -*-coding:gbk-*-
# """     EM算法：
# EM算法拟解决的问题（参数估计问题）：假设随机变量 X 是由 K 个"高斯分布"（理论上也可以是其他分布）混合而成,
# 取各个高斯分布的概率为π1, π2 ... πk ,第i个高斯分布的均值为μi ,方差为σi .
# 若观测到随机变量X的一系列样本x1 ,x2 ,...,xn ,试估计参数π,μ,σ的值.
#
# 举例：随机挑选10000位志愿者,测量他们的身高:若样本中存在男性和女性,身高分别服从N(μ1,σ1 )和N(μ2,σ2)
# 的高斯分布,试估计μ1,σ1和μ2,σ2
# 解读：随机变量X（身高）由2个（男/女）”高斯分布“混合而成,样本X以π1的概率来自男性高斯分布,以π2（1-π1）
# 的概率来自女性高斯分布。（换言之,如有一样本 x=180cm,它属于男性的身高概率假定为0.7,则它属于女性身高的
# 概率为0.3）我们需要估计的参数是来自两个高斯分布的概率分别为多少（π1, π2=？）以及两个高斯分布的均值和
# 方差多少（μ1,σ1和μ2,σ2=？）
#
# 在实际问题当中，上面的身高例子，只给出样本的具体身高值，却没有给出其所属性别，所以性别是属于隐变量，而我们
# 是根据经验推测是性别是影响分布的隐变量。从而假定原样本整体是由男/女高斯分布混合而成，接着我们要把这两个高斯
# 分布都分别求出来（求出其两个重要参数：均值和方差）。EM算法可以看做是寻找隐变量问题。
#
# EM算法是一种迭代型的算法，在每一次的迭代过程中，主要分为两步：即求期望(Expectation)步骤和最大化(Maximization)步骤。
#     1.首先初始化（人为给定）需要估计的参数值 π,μ,σ
#     2.E步骤：根据高斯分布的“概率公式”计算出每一个样本数据Xi是由第j个高斯分布产生的概率：
# """
#
from __future__ import division  # python中“精确除法”：除法总是会返回真实的商，不管操作数是整形还是浮点型,不会丢掉结果的小数点部分。
from numpy import *
import math as mt
import time
import matplotlib.pyplot as plt

"""首先生成一些用于测试的样本"""
# 指定两个高斯分布的参数：均值和标准差
sigma_1 = 2  # sigma不是方差,是标准差（重点）
sigma_2 = 4
miu_1 = 25
miu_2 = 30
# 随机均匀构造两个高斯分布，用于生成样本值
N = 10000
X = zeros((1, N))
sum_x_1 = 0
sum_x_2 = 0
sum_fc_1 = 0
sum_fc_2 = 0
count = 0
for i in xrange(N):
    if random.random() > 0.3:  # 使用的是numpy模块中的random():返回0-1之间的某个实数
        X[0, i] = random.randn() * sigma_1 + miu_1  # randn():从“标准正太分布”上返回一个实数
        sum_fc_1 = sum_fc_1+((X[0, i] - miu_1) ** 2)
        sum_x_1 = sum_x_1+X[0, i]
        count += 1
    else:
        X[0, i] = random.randn() * sigma_2 + miu_2
        sum_fc_2 = sum_fc_2 + ((X[0, i] - miu_2) ** 2)
        sum_x_2 = sum_x_2 + X[0, i]
print "构造的均值1=%.2f,方差1=%.2f"%(sum_x_1/count, sum_fc_1/count)
print "构造的均值2=%.2f,方差2=%.2f"%(sum_x_2/(N-count), sum_fc_2/(N-count))
# plt.hist(X[0, :], 500)
# plt.show()


"""
上述步骤已经生成样本(数据集)
# 对生成的样本，使用EM算法计算其均值miu和标准差sigma
采用普通for循环处理数据：循环1000次，用时：1690秒
采用numpy数组并行处理：循环1000次，用时：5秒
"""

# 取_pi,sigma,miu的初始值
# 矩阵（numpy里的）里的数值要设为浮点型，否则默认为整型数值，后面的矩阵计算会取整处理。
_pi = mat([0.5, 0.5])  # 来自两个高斯分布的样本数量占样本总数量的比例
sigma = mat([2.0, 4.0])
miu = mat([35.0, 5.0])
k = 2
Expectations = zeros((N, k))  # 记录每个样本来自各个高斯分布的概率

time_str = time.time()
for step in xrange(1000):  # 设置迭代次数
    # 步骤1，计算期望，计算每个样本来自各个高斯分布的概率
    for i in xrange(N):  # 样本总数N
        # 计算分母denominator
        denominator = 0
        for j in xrange(k):
            denominator = denominator + _pi[0, j] * (1 / (sqrt(2 * pi) * sigma[0, j])) * mt.exp((- (X[0, i] - miu[0, j]) ** 2) / (2 * sigma[0, j] ** 2))

        # 计算分子numerator
        for j in xrange(k):
            numerator = (_pi[0, j])*(1 / (sqrt(2 * pi) * sigma[0, j]))*mt.exp(-((X[0, i] - miu[0, j]) ** 2) / (2 * sigma[0, j] ** 2))
            Expectations[i, j] = numerator / denominator
        # print Expectations[i, 0], Expectations[i, 1]
        # print Expectations[i, 0]+Expectations[i, 1]

    # 步骤2，求期望的最大
    oldpi = zeros((1, k))
    oldMiu = zeros((1, k))
    oldSigma = zeros((1, k))
    for j in xrange(k):
        oldpi[0, j] = _pi[0, j]
        oldMiu[0, j] = miu[0, j]
        oldSigma[0, j] = sigma[0, j]
        gama = 0
        Nk = 0
        sum_fc = 0  # 方差
        for i in xrange(N):
            Nk = Nk + Expectations[i, j]  # 所有样本来自某一高斯分布的概率之和
            gama = gama + Expectations[i, j] * X[0, i]  # 所有样本来自某一高斯分布的期望值
            sum_fc = sum_fc + Expectations[i, j]*((X[0, i]-miu[0, j]) ** 2)  # 总方差
        _pi[0, j] = Nk / N
        miu[0, j] = gama / Nk  # 均值
        sigma[0, j] = sqrt(sum_fc/Nk)  # 标准差=方差开根号（重点）
    print miu, sigma, _pi
time_end = time.time()
print "循环1000次，用时：%d秒"%(time_end-time_str)  # 循环1000次，用时：1690秒

    # 判断是否满足要求
#     epsilon = 0.0001
#     if (sum(abs(miu - oldMiu)) < epsilon):  # and (sum(abs(sigma - oldSigma)) < epsilon)
#         break
#     print step, miu,  "\n"  # sigma,
# print miu

