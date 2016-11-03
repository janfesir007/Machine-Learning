# -*-coding:gbk-*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
"""测试数据的稳定性"""
def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(window=144).mean()
    rolstd = timeseries.rolling(window=144).std()

    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print dfoutput

"""ARIMA模型函数"""
def airport_pridict_ARIMA():

    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d-%H-%M')
    data = pd.read_csv("airport_Dataset/wifi_10mins_records_csv/E1/E1-1A-1<E1-1-01>.csv", parse_dates=[1], index_col=[1], date_parser=dateparse)
    # data = pd.read_csv("airport_Dataset/wifi_10mins_records_csv/E1/E1-1A-1<E1-1-01>.csv")
    # print data.head(10)
    # t1 = pd.datetime.strptime("2016-09-10-18-50", '%Y-%m-%d-%H-%M')
    # t2 = pd.datetime.strptime("2016-09-14-14-50", '%Y-%m-%d-%H-%M')
    # index = pd.date_range(start=t1, end=t2, freq="10Min")
    # ts.reindex(index=index)

    ts = data["counts"]  # 就变成只有时间和连接数两列
    ts = ts.asfreq(freq="10Min")  # 更改索引的时间间隔
    ts = pd.Series(ts, dtype=float)
    ts_diff = ts - ts.shift()  # 一阶差分
    ts_diff = ts_diff - ts_diff.shift()
    ts_diff.dropna(inplace=True)
    # test_stationarity(ts_diff)  # 测试一阶差分后数据的稳定性


    model = ARIMA(ts, order=(7, 1, 2))
    results_ARIMA = model.fit(disp=-1)
    # plt.plot(ts_diff)  # 二阶差分后的数据
    # plt.plot(results_ARIMA.fittedvalues, color='red')
    # plt.title('RSS: %.4f' % sum((results_ARIMA.fittedvalues - ts_diff) ** 2))
    # plt.show()

    # 使用模型来预测未来的估值(在差分后的值的基础上预测的)
    t1 = str(pd.datetime.strptime("2016-09-14-15-00", '%Y-%m-%d-%H-%M'))
    t2 = str(pd.datetime.strptime("2016-09-14-17-50", '%Y-%m-%d-%H-%M'))
    predict_sunspots = results_ARIMA.predict(t1, t2, dynamic=True)
    print predict_sunspots


    # 倒回到原始区间
    predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    predictions_ARIMA = pd.Series(ts.ix[0], index=ts.index)
    predictions_ARIMA = predictions_ARIMA.add(predictions_ARIMA_diff_cumsum, fill_value=0)
    # plt.plot(ts)
    # plt.plot(predictions_ARIMA, "r")
    # plt.title('RMSE: %.4f' % (np.sum((predictions_ARIMA - ts) ** 2) / len(ts)))
    # plt.show()







    # data_Seri = pd.Series(data[:len(data)-18], dtype=float)  # 后18个数据用作测试集
    # data_Seri.index = pd.Index(sm.tsa.datetools.dates_from_range('1965'))
    # data_diff = data_Seri.diff(1)
    # data_diff.dropna(inplace=True)
    # arma_mod = sm.tsa.ARMA(data_diff, (2, 0)).fit()
    # predict_sunspots = arma_mod.predict(str(len(data)-18), str(len(data)), dynamic=True)
    #
    # fig1, ax = plt.subplots(figsize=(12, 8))  # 两图放在同一坐标上
    # ax = data_diff.ix['1':].plot(ax=ax)  # 画出所有90条数据原始数据图
    # fig1 = arma_mod.plot_predict(str(len(data)-18), str(len(data), dynamic=True, ax=ax, plot_insample=False))  # 画出预测数据图
    # plt.show()
    # def mean_forecast_err(y, yhat):  # “差分后的数据”与“用差分后的数据预测的数据”计算误差;
    #     return y.sub(yhat).mean()
    # err = mean_forecast_err(data_diff, predict_sunspots)
    # print "差分后的数据的预测与真实值的误差均值：%f"%err
    #
    #
    # arma_mod_diff = pd.Series(arma_mod.fittedvalues, copy=True)  # 模型拟合后的差分值
    # arma_mod_diff_cumsum = arma_mod_diff.cumsum()  # 模型拟合后的差分值累加
    # arma_mod_resource = pd.Series(data_Seri.ix[0], index=data_Seri.index)
    # arma_mod_resource = arma_mod_resource.add(arma_mod_diff_cumsum, fill_value=0)
    # plt.plot(data_Seri)  # 原始样本值
    # plt.plot(arma_mod_resource, "r")  # 经模型拟合后的类原始值(所谓的“时间序列预测模型”所产生的预测值)
    # plt.title('RMSE: %.4f' % np.sqrt(sum((arma_mod_resource - data_Seri) ** 2) / len(data_Seri)))  # 原始值与预测值之间的方差
    # plt.show()

"""主函数"""
airport_pridict_ARIMA()