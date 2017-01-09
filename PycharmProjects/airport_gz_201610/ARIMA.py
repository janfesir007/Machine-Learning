# -*-coding:gbk-*-
# -*-coding:utf-8-*-
"""
时间序列模型："自回归/差分/移动平均"时间序列混合模型(ARIMA)
自回归：AR
差分：I  消除趋势和季节性
移动平均：MA
ARMA(dta,(p,q)).fit()   # ARMA模型拟合函数：dta：时间序列数据;参数p(自回归函数（AR）的条件),q是移动平均数(MA)的条件
ARIMA(dta,(p,d,q)).fit() # ARIMA模型拟合函数：参数p,q同上,d:差分的次数(取0,1或2),不是阶数！而且ARIMA中差分只是1阶！即：ARIMA模型只有d次1阶差分操作.
ARIMA模型:第一步：先找出使得数据平稳的差分次数d;第二步：利用d,根据“BIC准则”再找出参数p,q及其最优模型; 第三步：预测（无需差分还原操作）
ARMA和ARIMA模型的不同：
    如果数据不平稳,需差分处理,ARMA模型训练时使用的是差分后的数据,故预测后还需要差分还原操作.
    ARIMA模型训练时使用的是原始数据,预测后无需差分还原操作.该库对于ARIMA模型只提供了最高两次的差分ARIMA(p,d,q),即d最大只能取2.
    若某数据需要经过3次以上差分才平稳则使用ARMA模型.
"""
from __future__ import print_function  # 使得输出格式为：print()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller  # ADF单位根检验
import statsmodels.api as sm

def example(datas):  # “时间序列预测”进行数据预测
    d = 0  # 差分次数
    ts = pd.Series(datas, dtype=float)
    ts.index = pd.date_range(start="2001", periods=len(ts), freq="12M")  # periods：索引长度 ;freq：时间索引的间隔
    # dta.plot(figsize=(12,8))
    # plt.show()
    # 一.检测时间序列数据是否平稳
    dfoutput = testStationarity(ts)  # 平稳性检验结果dfoutput,返回的数据类型是时间序列
    print("原始数据p-value:", dfoutput["p-value"], "\n", dfoutput)  # p-value值越小越平稳

    # 二.对于不平稳的数据,找出其经过几次差分后才平稳（d=0,1,2）
    if (dfoutput['p-value'] >= 0.05):
        dta_diff1 = ts.diff(1)  # shift(m)：移动操作,将数据移动m个位置; diff(n)”一阶差分“处理:隔开n个位置的数据相减(先移动n个位置,再数据相减)即有：dta.diff(n) = dta - dta.shift(n)
        dta_diff1.dropna(inplace=True)  # 这句不能少:应该是diff后有无效数据(前两个数无效),直接去除
        dfoutput = testStationarity(dta_diff1)  # 再次检测数据是否平稳.平稳性检验结果dfoutput,返回的数据类型是时间序列
        if dfoutput["p-value"] < 0.05:
            d = 1
            print("1次1阶差分处理后p-value:", dfoutput["p-value"], "\n", dfoutput)
        else:
            dta_diff2 = dta_diff1.diff(1)
            dta_diff2.dropna(inplace=True)
            dfoutput_end = testStationarity(dta_diff2)
            if dfoutput_end["p-value"] < 0.05:
                d = 2
                print("2次1阶差分处理后p-value:", dfoutput["p-value"], "\n", dfoutput)
            else:
                raise Exception("警告：经2次1阶差分后数据仍不平稳！ARIMA模型不适用！")  # 异常被抛出,后面的语句无法执行

    # 三.模型训练
    p, q, arima_model = proper_model_pq(ts, len(ts)/10, d)  # 找出最优模型arima_model,模型训练使用原始数据ts
    # arima_model = ARIMA(ts, (q, d, q)).fit(disp=-1)
    # arima_model_fit = pd.Series(arima_model.fittedvalues, copy=True)  # 模型拟合得到的拟合值(2002-2080)
    # arima_model_fit['2001'] = ts['2001']
    # arima_model_fit['2002'] = ts['2002']

    #  四.使用模型来预测未来的估值
    # arima_model.summary2()
    # ARMA/ARIMA模型的predict函数都是针对差分后平稳序列的预测,故：使用该函数最终都要进行一步“还原差分”的操作
    # ARMA中forecast()与predict()类似.但ARIMA中forecast()可以直接预测,不需要差分还原操作.
    # 总结1：ARMA模型:数据经差分处理后,使用predict()预测平稳序列,后进行差分还原操作.
    # 总结2：ARIMA模型：使用forecast()直接预测,不需要差分还原操作.
    # predict_values = arima_model.predict(str(len(datas) + 2001), str(len(datas) + 2003), dynamic=True)  # # 预测下1个值,返回类型是Serise
    # forecast_values = arima_model.forecast(5)# 预测多个值

    forecast_values = arima_model.forecast(1)[0][0] # 预测下1个值

    # 新的预测值加入原始数据集,和原始数据集一起作为下一次预测的输入集
    new_data = []
    new_data.append(forecast_values)
    new_datas = datas + new_data
    return new_datas  # 返回值是list类型

"""
     对于个数不多的时序数据,我们可以通过观察自相关图和偏相关图来进行模型识别(确定p,q值),
     倘若我们要分析的时序数据量较多,例如要预测每只股票的走势,我们就不可能逐个去调参了.
     这时我们可以依据BIC准则识别模型的p, q值,通常认为BIC值越小的模型相对更优.
     BIC准则综合考虑了残差大小和自变量的个数,残差越小BIC值越小,自变量个数越少BIC值越小.
     下面的proper_model(data_ts, maxLag)函数：根据提供的平稳序列数据,求出其最优模型的p,q值.
"""
def proper_model_pq(data_ts, maxLag, d):
    init_bic = sys.maxint
    init_p = 0
    init_q = 0
    init_properModel = None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            model = ARIMA(data_ts, order=(p, d, q))  # (经上第二步骤验证1次差分后数据平稳,故d=1)
            try:
                results_ARIMA = model.fit(disp=-1)
            except:
                continue
            bic = results_ARIMA.bic
            if bic < init_bic:
                init_p = p
                init_q = q
                init_properModel = results_ARIMA
                init_bic = bic
    return init_p, init_q, init_properModel

"""ADF单位根检验,验证数据的平稳性"""
def testStationarity(ts):
    dftest = adfuller(ts)  # ADF单位根检验
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return dfoutput

"""主函数"""
train_data = [10930, 10318, 10595, 10972, 7706, 6756, 9092, 10551, 9722, 10913,
              11151, 8186, 6422, 6337, 11649, 11652, 10310, 12043, 7937, 6476,
              9662, 9570, 9981, 9331, 9449, 6773, 6304, 9355, 10477, 10148,
              10395, 11261, 8713, 7299, 10424, 10795, 11069, 11602, 11427, 9095,
              7707, 10767, 12136, 12812, 12006, 12528, 10329, 7818, 11719, 11683,
              12603, 11495, 13670, 11337, 10232, 13261, 13230, 15535, 16837, 19598,
              14823, 11622, 19391, 18177, 19994, 14723, 15694, 13248, 9543, 12872,
              13101, 15053, 12619, 13749, 10228, 9725, 14729, 12518, 14564, 15085]  # 80个数据
test_data = [14722, 11999, 9390, 13481, 14795, 15845, 15271, 14686, 11054, 10395]  # 10个数据作测试集

pridict_data = train_data
for i in range(len(test_data)):  # 迭代式循环预测多个后续值,而不采用在arma_model.pridict函数中一次预测多个值,出于对准确度的考虑
    pridict_data = example(pridict_data)  # pridict_data：预测后得到的完整数据（原始数据+预测数据）,list类型

# 作图展示
# 将list类型转化成Serise类型,periods表示数据个数,即对应索引个数,freq=“12M”：索引时间间隔12个Month
data_real = pd.Series(train_data + test_data, index=pd.date_range(start="2001", periods=len(test_data)+len(train_data),freq="12M"), dtype=float)
pridict_data = pd.Series(pridict_data, index=pd.date_range(start="2001", periods=len(pridict_data), freq="12M"), dtype=float)

fig = plt.figure(figsize=(12, 8))  # 画布
ax1 = fig.add_subplot(111)  # 坐标系
data_real.plot(ax=ax1)  # 原始样本值(2001-)
pridict_data[len(train_data)-1:].plot(ax=ax1, color="r")  # 预测值(2080-)
plt.title('RMSE: %.4f' % (np.sqrt(sum((np.array(test_data) - np.array(list(pridict_data[len(train_data):]))) ** 2) / len(test_data))))  # 实际值与预测值之间的方差
plt.show()