# -*-coding:gbk-*-
# -*-coding:utf-8-*-
"""
ʱ������ģ�ͣ�"�Իع�/���/�ƶ�ƽ��"ʱ�����л��ģ��(ARIMA)
�Իع飺AR
��֣�I  �������ƺͼ�����
�ƶ�ƽ����MA
ARMA(dta,(p,q)).fit()   # ARMAģ����Ϻ�����dta��ʱ����������;����p(�Իع麯����AR��������),q���ƶ�ƽ����(MA)������
ARIMA(dta,(p,d,q)).fit() # ARIMAģ����Ϻ���������p,qͬ��,d:��ֵĴ���(ȡ0,1��2),���ǽ���������ARIMA�в��ֻ��1�ף�����ARIMAģ��ֻ��d��1�ײ�ֲ���.
ARIMAģ��:��һ�������ҳ�ʹ������ƽ�ȵĲ�ִ���d;�ڶ���������d,���ݡ�BIC׼�����ҳ�����p,q��������ģ��; ��������Ԥ�⣨�����ֻ�ԭ������
ARMA��ARIMAģ�͵Ĳ�ͬ��
    ������ݲ�ƽ��,���ִ���,ARMAģ��ѵ��ʱʹ�õ��ǲ�ֺ������,��Ԥ�����Ҫ��ֻ�ԭ����.
    ARIMAģ��ѵ��ʱʹ�õ���ԭʼ����,Ԥ��������ֻ�ԭ����.�ÿ����ARIMAģ��ֻ�ṩ��������εĲ��ARIMA(p,d,q),��d���ֻ��ȡ2.
    ��ĳ������Ҫ����3�����ϲ�ֲ�ƽ����ʹ��ARMAģ��.
"""
from __future__ import print_function  # ʹ�������ʽΪ��print()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller  # ADF��λ������
import statsmodels.api as sm

def example(datas):  # ��ʱ������Ԥ�⡱��������Ԥ��
    d = 0  # ��ִ���
    ts = pd.Series(datas, dtype=float)
    ts.index = pd.date_range(start="2001", periods=len(ts), freq="12M")  # periods���������� ;freq��ʱ�������ļ��
    # dta.plot(figsize=(12,8))
    # plt.show()
    # һ.���ʱ�����������Ƿ�ƽ��
    dfoutput = testStationarity(ts)  # ƽ���Լ�����dfoutput,���ص�����������ʱ������
    print("ԭʼ����p-value:", dfoutput["p-value"], "\n", dfoutput)  # p-valueֵԽСԽƽ��

    # ��.���ڲ�ƽ�ȵ�����,�ҳ��侭�����β�ֺ��ƽ�ȣ�d=0,1,2��
    if (dfoutput['p-value'] >= 0.05):
        dta_diff1 = ts.diff(1)  # shift(m)���ƶ�����,�������ƶ�m��λ��; diff(n)��һ�ײ�֡�����:����n��λ�õ��������(���ƶ�n��λ��,���������)���У�dta.diff(n) = dta - dta.shift(n)
        dta_diff1.dropna(inplace=True)  # ��䲻����:Ӧ����diff������Ч����(ǰ��������Ч),ֱ��ȥ��
        dfoutput = testStationarity(dta_diff1)  # �ٴμ�������Ƿ�ƽ��.ƽ���Լ�����dfoutput,���ص�����������ʱ������
        if dfoutput["p-value"] < 0.05:
            d = 1
            print("1��1�ײ�ִ����p-value:", dfoutput["p-value"], "\n", dfoutput)
        else:
            dta_diff2 = dta_diff1.diff(1)
            dta_diff2.dropna(inplace=True)
            dfoutput_end = testStationarity(dta_diff2)
            if dfoutput_end["p-value"] < 0.05:
                d = 2
                print("2��1�ײ�ִ����p-value:", dfoutput["p-value"], "\n", dfoutput)
            else:
                raise Exception("���棺��2��1�ײ�ֺ������Բ�ƽ�ȣ�ARIMAģ�Ͳ����ã�")  # �쳣���׳�,���������޷�ִ��

    # ��.ģ��ѵ��
    p, q, arima_model = proper_model_pq(ts, len(ts)/10, d)  # �ҳ�����ģ��arima_model,ģ��ѵ��ʹ��ԭʼ����ts
    # arima_model = ARIMA(ts, (q, d, q)).fit(disp=-1)
    # arima_model_fit = pd.Series(arima_model.fittedvalues, copy=True)  # ģ����ϵõ������ֵ(2002-2080)
    # arima_model_fit['2001'] = ts['2001']
    # arima_model_fit['2002'] = ts['2002']

    #  ��.ʹ��ģ����Ԥ��δ���Ĺ�ֵ
    # arima_model.summary2()
    # ARMA/ARIMAģ�͵�predict����������Բ�ֺ�ƽ�����е�Ԥ��,�ʣ�ʹ�øú������ն�Ҫ����һ������ԭ��֡��Ĳ���
    # ARMA��forecast()��predict()����.��ARIMA��forecast()����ֱ��Ԥ��,����Ҫ��ֻ�ԭ����.
    # �ܽ�1��ARMAģ��:���ݾ���ִ����,ʹ��predict()Ԥ��ƽ������,����в�ֻ�ԭ����.
    # �ܽ�2��ARIMAģ�ͣ�ʹ��forecast()ֱ��Ԥ��,����Ҫ��ֻ�ԭ����.
    # predict_values = arima_model.predict(str(len(datas) + 2001), str(len(datas) + 2003), dynamic=True)  # # Ԥ����1��ֵ,����������Serise
    # forecast_values = arima_model.forecast(5)# Ԥ����ֵ

    forecast_values = arima_model.forecast(1)[0][0] # Ԥ����1��ֵ

    # �µ�Ԥ��ֵ����ԭʼ���ݼ�,��ԭʼ���ݼ�һ����Ϊ��һ��Ԥ������뼯
    new_data = []
    new_data.append(forecast_values)
    new_datas = datas + new_data
    return new_datas  # ����ֵ��list����

"""
     ���ڸ��������ʱ������,���ǿ���ͨ���۲������ͼ��ƫ���ͼ������ģ��ʶ��(ȷ��p,qֵ),
     ��������Ҫ������ʱ���������϶�,����ҪԤ��ÿֻ��Ʊ������,���ǾͲ��������ȥ������.
     ��ʱ���ǿ�������BIC׼��ʶ��ģ�͵�p, qֵ,ͨ����ΪBICֵԽС��ģ����Ը���.
     BIC׼���ۺϿ����˲в��С���Ա����ĸ���,�в�ԽСBICֵԽС,�Ա�������Խ��BICֵԽС.
     �����proper_model(data_ts, maxLag)�����������ṩ��ƽ����������,���������ģ�͵�p,qֵ.
"""
def proper_model_pq(data_ts, maxLag, d):
    init_bic = sys.maxint
    init_p = 0
    init_q = 0
    init_properModel = None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            model = ARIMA(data_ts, order=(p, d, q))  # (���ϵڶ�������֤1�β�ֺ�����ƽ��,��d=1)
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

"""ADF��λ������,��֤���ݵ�ƽ����"""
def testStationarity(ts):
    dftest = adfuller(ts)  # ADF��λ������
    # ������������õ�ֵ������������
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return dfoutput

"""������"""
train_data = [10930, 10318, 10595, 10972, 7706, 6756, 9092, 10551, 9722, 10913,
              11151, 8186, 6422, 6337, 11649, 11652, 10310, 12043, 7937, 6476,
              9662, 9570, 9981, 9331, 9449, 6773, 6304, 9355, 10477, 10148,
              10395, 11261, 8713, 7299, 10424, 10795, 11069, 11602, 11427, 9095,
              7707, 10767, 12136, 12812, 12006, 12528, 10329, 7818, 11719, 11683,
              12603, 11495, 13670, 11337, 10232, 13261, 13230, 15535, 16837, 19598,
              14823, 11622, 19391, 18177, 19994, 14723, 15694, 13248, 9543, 12872,
              13101, 15053, 12619, 13749, 10228, 9725, 14729, 12518, 14564, 15085]  # 80������
test_data = [14722, 11999, 9390, 13481, 14795, 15845, 15271, 14686, 11054, 10395]  # 10�����������Լ�

pridict_data = train_data
for i in range(len(test_data)):  # ����ʽѭ��Ԥ��������ֵ,����������arma_model.pridict������һ��Ԥ����ֵ,���ڶ�׼ȷ�ȵĿ���
    pridict_data = example(pridict_data)  # pridict_data��Ԥ���õ����������ݣ�ԭʼ����+Ԥ�����ݣ�,list����

# ��ͼչʾ
# ��list����ת����Serise����,periods��ʾ���ݸ���,����Ӧ��������,freq=��12M��������ʱ����12��Month
data_real = pd.Series(train_data + test_data, index=pd.date_range(start="2001", periods=len(test_data)+len(train_data),freq="12M"), dtype=float)
pridict_data = pd.Series(pridict_data, index=pd.date_range(start="2001", periods=len(pridict_data), freq="12M"), dtype=float)

fig = plt.figure(figsize=(12, 8))  # ����
ax1 = fig.add_subplot(111)  # ����ϵ
data_real.plot(ax=ax1)  # ԭʼ����ֵ(2001-)
pridict_data[len(train_data)-1:].plot(ax=ax1, color="r")  # Ԥ��ֵ(2080-)
plt.title('RMSE: %.4f' % (np.sqrt(sum((np.array(test_data) - np.array(list(pridict_data[len(train_data):]))) ** 2) / len(test_data))))  # ʵ��ֵ��Ԥ��ֵ֮��ķ���
plt.show()