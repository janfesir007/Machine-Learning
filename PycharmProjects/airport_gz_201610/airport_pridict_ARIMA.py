# -*-coding:gbk-*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
"""�������ݵ��ȶ���"""
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

"""ARIMAģ�ͺ���"""
def airport_pridict_ARIMA():

    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d-%H-%M')
    data = pd.read_csv("airport_Dataset/wifi_10mins_records_csv/E1/E1-1A-1<E1-1-01>.csv", parse_dates=[1], index_col=[1], date_parser=dateparse)
    # data = pd.read_csv("airport_Dataset/wifi_10mins_records_csv/E1/E1-1A-1<E1-1-01>.csv")
    # print data.head(10)
    # t1 = pd.datetime.strptime("2016-09-10-18-50", '%Y-%m-%d-%H-%M')
    # t2 = pd.datetime.strptime("2016-09-14-14-50", '%Y-%m-%d-%H-%M')
    # index = pd.date_range(start=t1, end=t2, freq="10Min")
    # ts.reindex(index=index)

    ts = data["counts"]  # �ͱ��ֻ��ʱ�������������
    ts = ts.asfreq(freq="10Min")  # ����������ʱ����
    ts = pd.Series(ts, dtype=float)
    ts_diff = ts - ts.shift()  # һ�ײ��
    ts_diff = ts_diff - ts_diff.shift()
    ts_diff.dropna(inplace=True)
    # test_stationarity(ts_diff)  # ����һ�ײ�ֺ����ݵ��ȶ���


    model = ARIMA(ts, order=(7, 1, 2))
    results_ARIMA = model.fit(disp=-1)
    # plt.plot(ts_diff)  # ���ײ�ֺ������
    # plt.plot(results_ARIMA.fittedvalues, color='red')
    # plt.title('RSS: %.4f' % sum((results_ARIMA.fittedvalues - ts_diff) ** 2))
    # plt.show()

    # ʹ��ģ����Ԥ��δ���Ĺ�ֵ(�ڲ�ֺ��ֵ�Ļ�����Ԥ���)
    t1 = str(pd.datetime.strptime("2016-09-14-15-00", '%Y-%m-%d-%H-%M'))
    t2 = str(pd.datetime.strptime("2016-09-14-17-50", '%Y-%m-%d-%H-%M'))
    predict_sunspots = results_ARIMA.predict(t1, t2, dynamic=True)
    print predict_sunspots


    # ���ص�ԭʼ����
    predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    predictions_ARIMA = pd.Series(ts.ix[0], index=ts.index)
    predictions_ARIMA = predictions_ARIMA.add(predictions_ARIMA_diff_cumsum, fill_value=0)
    # plt.plot(ts)
    # plt.plot(predictions_ARIMA, "r")
    # plt.title('RMSE: %.4f' % (np.sum((predictions_ARIMA - ts) ** 2) / len(ts)))
    # plt.show()


"""������"""
airport_pridict_ARIMA()