# -*-coding:gbk-*-
# """     EM�㷨��
# EM�㷨���������⣨�����������⣩������������� X ���� K ��"��˹�ֲ�"��������Ҳ�����������ֲ�����϶���,
# ȡ������˹�ֲ��ĸ���Ϊ��1, ��2 ... ��k ,��i����˹�ֲ��ľ�ֵΪ��i ,����Ϊ��i .
# ���۲⵽�������X��һϵ������x1 ,x2 ,...,xn ,�Թ��Ʋ�����,��,�ҵ�ֵ.
#
# �����������ѡ10000λ־Ը��,�������ǵ����:�������д������Ժ�Ů��,��߷ֱ����N(��1,��1 )��N(��2,��2)
# �ĸ�˹�ֲ�,�Թ��Ʀ�1,��1�ͦ�2,��2
# ������������X����ߣ���2������/Ů������˹�ֲ�����϶���,����X�Ԧ�1�ĸ����������Ը�˹�ֲ�,�Ԧ�2��1-��1��
# �ĸ�������Ů�Ը�˹�ֲ���������֮,����һ���� x=180cm,���������Ե���߸��ʼٶ�Ϊ0.7,��������Ů����ߵ�
# ����Ϊ0.3��������Ҫ���ƵĲ���������������˹�ֲ��ĸ��ʷֱ�Ϊ���٣���1, ��2=�����Լ�������˹�ֲ��ľ�ֵ��
# ������٣���1,��1�ͦ�2,��2=����
#
# ��ʵ�����⵱�У������������ӣ�ֻ���������ľ������ֵ��ȴû�и����������Ա������Ա���������������������
# �Ǹ��ݾ����Ʋ����Ա���Ӱ��ֲ������������Ӷ��ٶ�ԭ��������������/Ů��˹�ֲ���϶��ɣ���������Ҫ����������˹
# �ֲ����ֱ�������������������Ҫ��������ֵ�ͷ����EM�㷨���Կ�����Ѱ�����������⡣
#
# EM�㷨��һ�ֵ����͵��㷨����ÿһ�εĵ��������У���Ҫ��Ϊ��������������(Expectation)��������(Maximization)���衣
#     1.���ȳ�ʼ������Ϊ��������Ҫ���ƵĲ���ֵ ��,��,��
#     2.E���裺���ݸ�˹�ֲ��ġ����ʹ�ʽ�������ÿһ����������Xi���ɵ�j����˹�ֲ������ĸ��ʣ�
# """
#
from __future__ import division  # python�С���ȷ���������������ǻ᷵����ʵ���̣����ܲ����������λ��Ǹ�����,���ᶪ�������С���㲿�֡�
from numpy import *
import math as mt
import time
import matplotlib.pyplot as plt

"""��������һЩ���ڲ��Ե�����"""
# ָ��������˹�ֲ��Ĳ�������ֵ�ͱ�׼��
sigma_1 = 2  # sigma���Ƿ���,�Ǳ�׼��ص㣩
sigma_2 = 4
miu_1 = 25
miu_2 = 30
# ������ȹ���������˹�ֲ���������������ֵ
N = 10000
X = zeros((1, N))
sum_x_1 = 0
sum_x_2 = 0
sum_fc_1 = 0
sum_fc_2 = 0
count = 0
for i in xrange(N):
    if random.random() > 0.3:  # ʹ�õ���numpyģ���е�random():����0-1֮���ĳ��ʵ��
        X[0, i] = random.randn() * sigma_1 + miu_1  # randn():�ӡ���׼��̫�ֲ����Ϸ���һ��ʵ��
        sum_fc_1 = sum_fc_1+((X[0, i] - miu_1) ** 2)
        sum_x_1 = sum_x_1+X[0, i]
        count += 1
    else:
        X[0, i] = random.randn() * sigma_2 + miu_2
        sum_fc_2 = sum_fc_2 + ((X[0, i] - miu_2) ** 2)
        sum_x_2 = sum_x_2 + X[0, i]
print "����ľ�ֵ1=%.2f,����1=%.2f"%(sum_x_1/count, sum_fc_1/count)
print "����ľ�ֵ2=%.2f,����2=%.2f"%(sum_x_2/(N-count), sum_fc_2/(N-count))
# plt.hist(X[0, :], 500)
# plt.show()


"""
���������Ѿ���������(���ݼ�)
# �����ɵ�������ʹ��EM�㷨�������ֵmiu�ͱ�׼��sigma
������ͨforѭ���������ݣ�ѭ��1000�Σ���ʱ��1690��
����numpy���鲢�д���ѭ��1000�Σ���ʱ��5��
"""

# ȡ_pi,sigma,miu�ĳ�ʼֵ
# ����numpy��ģ������ֵҪ��Ϊ�����ͣ�����Ĭ��Ϊ������ֵ������ľ�������ȡ������
_pi = mat([0.5, 0.5])  # ����������˹�ֲ�����������ռ�����������ı���
sigma = mat([2.0, 4.0])
miu = mat([35.0, 5.0])
k = 2
Expectations = zeros((N, k))  # ��¼ÿ���������Ը�����˹�ֲ��ĸ���

time_str = time.time()
for step in xrange(1000):  # ���õ�������
    # ����1����������������ÿ���������Ը�����˹�ֲ��ĸ���
    for i in xrange(N):  # ��������N
        # �����ĸdenominator
        denominator = 0
        for j in xrange(k):
            denominator = denominator + _pi[0, j] * (1 / (sqrt(2 * pi) * sigma[0, j])) * mt.exp((- (X[0, i] - miu[0, j]) ** 2) / (2 * sigma[0, j] ** 2))

        # �������numerator
        for j in xrange(k):
            numerator = (_pi[0, j])*(1 / (sqrt(2 * pi) * sigma[0, j]))*mt.exp(-((X[0, i] - miu[0, j]) ** 2) / (2 * sigma[0, j] ** 2))
            Expectations[i, j] = numerator / denominator
        # print Expectations[i, 0], Expectations[i, 1]
        # print Expectations[i, 0]+Expectations[i, 1]

    # ����2�������������
    oldpi = zeros((1, k))
    oldMiu = zeros((1, k))
    oldSigma = zeros((1, k))
    for j in xrange(k):
        oldpi[0, j] = _pi[0, j]
        oldMiu[0, j] = miu[0, j]
        oldSigma[0, j] = sigma[0, j]
        gama = 0
        Nk = 0
        sum_fc = 0  # ����
        for i in xrange(N):
            Nk = Nk + Expectations[i, j]  # ������������ĳһ��˹�ֲ��ĸ���֮��
            gama = gama + Expectations[i, j] * X[0, i]  # ������������ĳһ��˹�ֲ�������ֵ
            sum_fc = sum_fc + Expectations[i, j]*((X[0, i]-miu[0, j]) ** 2)  # �ܷ���
        _pi[0, j] = Nk / N
        miu[0, j] = gama / Nk  # ��ֵ
        sigma[0, j] = sqrt(sum_fc/Nk)  # ��׼��=������ţ��ص㣩
    print miu, sigma, _pi
time_end = time.time()
print "ѭ��1000�Σ���ʱ��%d��"%(time_end-time_str)  # ѭ��1000�Σ���ʱ��1690��

    # �ж��Ƿ�����Ҫ��
#     epsilon = 0.0001
#     if (sum(abs(miu - oldMiu)) < epsilon):  # and (sum(abs(sigma - oldSigma)) < epsilon)
#         break
#     print step, miu,  "\n"  # sigma,
# print miu

