# -*-coding:gbk-*-
# -*-coding:utf-8-*-
"""
时间序列模型："自回归/差分/移动平均"时间序列混合模型(ARIMA)
自回归：AR
差分：I  消除趋势和季节性
移动平均：MA
ARMA(dta,(p,q)).fit()   # ARMA模型拟合函数：dta：时间序列数据;参数p(自回归函数（AR）的条件),q是移动平均数(MA)的条件
ARIMA(dta,(p,d,q)).fit() # ARIMA模型拟合函数：参数p,q同上,d:差分(一般取1或2)
"""
from __future__ import print_function  # 使得输出格式为：print()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA

def example_1():
     data=[10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,
          11151,8186,6422,6337,11649,11652,10310,12043,7937,6476,
          9662,9570,9981,9331,9449,6773,6304,9355,10477,10148,
          10395,11261,8713,7299,10424,10795,11069,11602,11427,9095,
          7707,10767,12136,12812,12006,12528,10329,7818,11719,11683,
          12603,11495,13670,11337,10232,13261,13230,15535,16837,19598,
          14823,11622,19391,18177,19994,14723,15694,13248,9543,12872,
          13101,15053,12619,13749,10228,9725,14729,12518,14564,15085,
          14722,11999,9390,13481,14795,15845,15271,14686,11054,10395]  # 90个数据
     dta = pd.Series(data, dtype=float)  # 90个数据作训练集（后面10个数据也作测试集）
     dta.index = pd.Index(sm.tsa.datetools.dates_from_range('2001', '2090'))
     # dta.plot(figsize=(12,8))
     # plt.show()

     """
     # 先对时间序列数据进行2阶差分，,然后再用ARMA模型(自回归AR/移动平均MA 时间序列模型)拟合
     dta_diff = dta.diff(2),sm.tsa.ARMA(dta_diff, (2, 0)).fit()
     并不等价与
     sm.tsa.ARIMA(dta,(2,2,0)).fit()  # 将差分I融合进ARMA模型得到ARIMA模型
     """
     dta_diff = dta.diff(2)  # shift(m)：移动操作,将数据移动m个位置; diff(n)”一阶差分“处理:隔开n个位置的数据相减(先移动n个位置,再数据相减)即有：dta.diff(n) = dta - dta.shift(n)

     dta_diff.dropna(inplace=True)  # 这句不能少:应该是diff后有无效数据(第一个数无效),直接去除
     # fig = plt.figure(figsize=(12,8))
     # ax1=fig.add_subplot(211)
     # fig = sm.graphics.tsa.plot_acf(dta_diff,lags=40,ax=ax1)
     # ax2 = fig.add_subplot(212)
     # fig = sm.graphics.tsa.plot_pacf(dta_diff,lags=40,ax=ax2)
     # plt.show()

     arma_mod20 = sm.tsa.ARMA(dta_diff, (2, 1)).fit()  # 模型拟合：将差分后的数据作为样本数据,对它进行建模拟合
     # print (arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic)
     # arma_mod30 = sm.tsa.ARMA(dta_diff,(0,1)).fit()
     # print arma_mod30.aic,arma_mod30.bic,arma_mod30.hqic
     # arma_mod40 = sm.tsa.ARMA(dta_diff,(7,1)).fit()
     # print arma_mod40.aic,arma_mod40.bic,arma_mod40.hqic
     # arma_mod50 = sm.tsa.ARMA(dta_diff,(8,0)).fit()
     # print arma_mod50.aic,arma_mod50.bic,arma_mod50.hqic

     # fig = plt.figure(figsize=(12,8))
     # ax1 = fig.add_subplot(211)
     # fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
     # ax2 = fig.add_subplot(212)
     # fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)

     predict_test = arma_mod20.predict()  # 使用模型来预测未来的估值(数据在差分后的基础上预测的)
     # print(predict_sunspots)  # 输出预测值
     predict_real = arma_mod20.predict('2091', '2100', dynamic=True)  # 使用模型来预测未来的估值(数据在差分后的基础上预测的)
     # print(predict_sunspots)  # 输出预测值

     # fig1, ax = plt.subplots(figsize=(12, 8))  # 两图放在同一坐标上
     # ax = dta_diff.ix['2001':].plot(ax=ax)  # 画出所有90条数据原始数据图
     # fig1 = arma_mod20.plot_predict('2081', '2090', dynamic=True, ax=ax, plot_insample=False)  # 画出差分后的预测数据图
     # plt.title("Pridict figure")
     # plt.show()
     # def mean_forecast_err(y, yhat):
     #     return y.sub(yhat).mean()
     # err1 = mean_forecast_err(dta_diff, predict_test)  # 差分后的真实值与差分后预测的新值：计算误差;
     # err2 = mean_forecast_err(dta_diff, arma_mod20.fittedvalues)  # 差分后的真实值与差分后拟合的值
     # print ("测试集上的误差：%f;%f"%(err1,err2))
     # dta_11 = np.array(dta_diff.ix["2080":"2090"].values)
     # dta_pri = np.array(predict_sunspots.values)
     # err1 = dta_11 - dta_pri
     # print (err.mean()-err)
     """  当然以上是对原始数据做了”一阶差分”(dta = dta.diff(1))后形成的新数据所做的预测,
          如何将预测结果反转回对原始数据的预测是下一步需要做的事情:如下
     """
     arma_mod20_diff = pd.Series(arma_mod20.fittedvalues, copy=True)  # 差分后模型拟合得到的拟合值(2002-2090)
     # print (arma_mod20_diff.head())
     arma_mod20_diff_cumsum1 = arma_mod20_diff.cumsum()  # 模型拟合后的差分值累加
     # arma_mod20_diff_cumsum2 = predict_test.add(dta.shift(2))  # 还原一阶差分？？？shift(m)：移动操作,将数据移动m个位置;
     # print (arma_mod20_diff_cumsum.head())
     arma_mod20_resource = pd.Series(dta.ix[0], index=dta.index)  # 定义时间序列变量,存放差分还原后的类原始值
     arma_mod20_resource = arma_mod20_resource.add(arma_mod20_diff_cumsum1, fill_value=0)  # 差分值倒回至原始趋势的值
     # print (arma_mod20_resource.head())
     plt.plot(dta)  #  原始样本值(2001-2090)
     plt.plot(arma_mod20_resource, "r")  # 经模型拟合后的类原始值(所谓的“时间序列预测模型”所产生的预测值)(2001-2090)
     plt.title('RMSE: %.4f' % np.sqrt(sum((arma_mod20_resource - dta) ** 2) / len(dta)))  #  原始值与预测值之间的方差
     plt.show()

"""主函数"""
example_1()
