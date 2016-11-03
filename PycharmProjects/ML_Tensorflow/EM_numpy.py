# -*-coding:gbk-*-
# -*-coding:UTF-8-*-
"""
�����鲢�м��㡱��EM�㷨�汾
���飬�������Ԫ��ʵ�ֲ��м��㣨ͬʱ���㣩
"""
from __future__ import division  # python�С���ȷ���������������ǻ᷵����ʵ���̣����ܲ����������λ��Ǹ�����,���ᶪ�������С���㲿�֡�
from numpy import *
import time
import matplotlib.pyplot as plt
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

#��ʼ��
pro_pi = array([[0.5, 0.5]])
sigma = array([[2.0, 4.0]])
miu = array([[35.0, 5.0]])
Expectations = zeros((N, k))  # ��¼ÿ���������Ը�����˹�ֲ��ĸ���

time_str = time.time()
for step in xrange(1000):  # ���õ�������
    # ����1����������������ÿ���������Ը�����˹�ֲ��ĸ���
    # �����ĸdenominator
    denominator = (pro_pi * (1 / (sqrt(2 * pi) * sigma))) * exp((- (X - miu) ** 2) / (2 * sigma ** 2))
    denominator = reshape(sum(denominator, 1), (N, 1))
    # �������numerator
    numerator = pro_pi * (1 / (sqrt(2 * pi) * sigma)) * exp(-((X - miu) ** 2) / (2 * sigma ** 2))
    Expectations = numerator / denominator

    # ����2�������������
    Nk = sum(Expectations, 0)  # ������������ĳһ��˹�ֲ��ĸ���֮��(�����)
    gama = sum(Expectations * X, 0)  # ������������ĳһ��˹�ֲ�������ֵ
    sum_fc = sum(Expectations * ((X - miu) ** 2), 0)  # �ܷ���
    pro_pi = Nk / N
    miu = gama / Nk  # ��ֵ
    sigma = sqrt(sum_fc / Nk)  # ��׼��=������ţ��ص㣩
    print "pi=", pro_pi, "miu=", miu, "sigma", sigma
time_end = time.time()
print "ѭ��1000�Σ���ʱ��%d��"% (time_end-time_str)  # ѭ��1000�Σ���ʱ��5��

