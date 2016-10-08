# -*-coding:gbk-*-
"""     EM�㷨��
EM�㷨���������⣨�����������⣩������������� X ���� K ��"��˹�ֲ�"��������Ҳ�����������ֲ�����϶���,
ȡ������˹�ֲ��ĸ���Ϊ��1, ��2 ... ��k ,��i����˹�ֲ��ľ�ֵΪ��i ,����Ϊ��i .
���۲⵽�������X��һϵ������x1 ,x2 ,...,xn ,�Թ��Ʋ�����,��,�ҵ�ֵ.

�����������ѡ10000λ־Ը��,�������ǵ����:�������д������Ժ�Ů��,��߷ֱ����N(��1,��1 )��N(��2,��2)
�ĸ�˹�ֲ�,�Թ��Ʀ�1,��1�ͦ�2,��2
������������X����ߣ���2������/Ů������˹�ֲ�����϶���,����X�Ԧ�1�ĸ����������Ը�˹�ֲ�,�Ԧ�2��1-��1��
�ĸ�������Ů�Ը�˹�ֲ���������֮,����һ���� x=180cm,���������Ե���߸��ʼٶ�Ϊ0.7,��������Ů����ߵ�
����Ϊ0.3��������Ҫ���ƵĲ���������������˹�ֲ��ĸ��ʷֱ�Ϊ���٣���1, ��2=�����Լ�������˹�ֲ��ľ�ֵ��
������٣���1,��1�ͦ�2,��2=����

��ʵ�����⵱�У������������ӣ�ֻ���������ľ������ֵ��ȴû�и����������Ա������Ա���������������������
�Ǹ��ݾ����Ʋ����Ա���Ӱ��ֲ������������Ӷ��ٶ�ԭ��������������/Ů��˹�ֲ���϶��ɣ���������Ҫ����������˹
�ֲ����ֱ�������������������Ҫ��������ֵ�ͷ����EM�㷨���Կ�����Ѱ�����������⡣

EM�㷨��һ�ֵ����͵��㷨����ÿһ�εĵ��������У���Ҫ��Ϊ��������������(Expectation)��������(Maximization)���衣
    1.���ȳ�ʼ������Ϊ��������Ҫ���ƵĲ���ֵ ��,��,��
    2.E���裺���ݸ�˹�ֲ��ġ����ʹ�ʽ�������ÿһ����������Xi���ɵ�j����˹�ֲ������ĸ��ʣ�
"""

from __future__ import division
from numpy import *
import math as mt

# ��������һЩ���ڲ��Ե�����
# ָ��������˹�ֲ��Ĳ���
sigma_1 = 6  # ����
sigma_2 = 4
miu_1 = 40
miu_2 = 20

# ������ȹ���������˹�ֲ���������������ֵ
N = 1000
X = zeros((1, N))
for i in xrange(N):
    if random.random() > 0.5:  # ʹ�õ���numpyģ���е�random():����0-1֮���ĳ��ʵ��
        X[0, i] = random.randn() * sigma_1 + miu_1  # randn():�ӡ���׼��̫�ֲ����Ϸ���һ��ʵ��
    else:
        X[0, i] = random.randn() * sigma_2 + miu_2

# ���������Ѿ���������
# �����ɵ�������ʹ��EM�㷨�������ֵmiu

# ȡsigma,miu�ĳ�ʼֵ
# miu = random.random((1, k))
sigma = mat([6, 4])
miu = mat([40.0, 20.0])
k = 2
Expectations = zeros((N, k))  # ��¼ÿ���������Ը�����˹�ֲ��ĸ���

for step in xrange(2000):  # ���õ�������
    # ����1����������������ÿ���������Ը�����˹�ֲ��ĸ���
    for i in xrange(N):  # ��������N
        # �����ĸdenominator
        denominator = 0
        for j in xrange(k):
            denominator = denominator + (1/sqrt(2*pi)*sigma[0, j])*mt.exp((- (X[0, i] - miu[0, j]) ** 2) / (2 * sigma[0,j] ** 2))

        # �������numerator
        for j in xrange(k):
            numerator = (1/sqrt(2*pi)*sigma[0, j])*mt.exp(-1 / (2 * sigma[0, j] ** 2) * (X[0, i] - miu[0, j]) ** 2)
            Expectations[i, j] = numerator / denominator

    # ����2�������������
    oldMiu = miu
    oldSigma = sigma
    for j in xrange(k):
        oldMiu[0, j] = miu[0, j]
        oldSigma[0, j] = sigma[0, j]
        gama = 0
        Nk = 0
        sum_sigma = 0
        for i in xrange(N):
            Nk = Nk + Expectations[i, j]  # ������������ĳһ��˹�ֲ��ĸ���֮��
            gama = gama + Expectations[i, j] * X[0, i]  # ������������ĳһ��˹�ֲ�������ֵ
            sum_sigma = sum_sigma + Expectations[i, j]*((X[0, i]-miu[0, j]) ** 2)
        miu[0, j] = gama / Nk  # ��ֵ
        sigma[0, j] = sum_sigma / Nk  # ����
    # �ж��Ƿ�����Ҫ��
    epsilon = 0.0001
    if (sum(abs(miu - oldMiu)) < epsilon) and (sum(abs(sigma - oldSigma)) < epsilon):
        break
    print step, miu, sigma, "\n"
print miu, sigma
