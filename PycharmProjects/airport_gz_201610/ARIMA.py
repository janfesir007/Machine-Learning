# -*-coding:gbk-*-
# -*-coding:utf-8-*-
"""
ʱ������ģ�ͣ�"�Իع�/���/�ƶ�ƽ��"ʱ�����л��ģ��(ARIMA)
�Իع飺AR
��֣�I  �������ƺͼ�����
�ƶ�ƽ����MA
ARMA(dta,(p,q)).fit()   # ARMAģ����Ϻ�����dta��ʱ����������;����p(�Իع麯����AR��������),q���ƶ�ƽ����(MA)������
ARIMA(dta,(p,d,q)).fit() # ARIMAģ����Ϻ���������p,qͬ��,d:���(һ��ȡ1��2)
"""
from __future__ import print_function  # ʹ�������ʽΪ��print()
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
          14722,11999,9390,13481,14795,15845,15271,14686,11054,10395]  # 90������
     dta = pd.Series(data, dtype=float)  # 90��������ѵ����������10������Ҳ�����Լ���
     dta.index = pd.Index(sm.tsa.datetools.dates_from_range('2001', '2090'))
     # dta.plot(figsize=(12,8))
     # plt.show()

     """
     # �ȶ�ʱ���������ݽ���2�ײ�֣�,Ȼ������ARMAģ��(�Իع�AR/�ƶ�ƽ��MA ʱ������ģ��)���
     dta_diff = dta.diff(2),sm.tsa.ARMA(dta_diff, (2, 0)).fit()
     �����ȼ���
     sm.tsa.ARIMA(dta,(2,2,0)).fit()  # �����I�ںϽ�ARMAģ�͵õ�ARIMAģ��
     """
     dta_diff = dta.diff(2)  # shift(m)���ƶ�����,�������ƶ�m��λ��; diff(n)��һ�ײ�֡�����:����n��λ�õ��������(���ƶ�n��λ��,���������)���У�dta.diff(n) = dta - dta.shift(n)

     dta_diff.dropna(inplace=True)  # ��䲻����:Ӧ����diff������Ч����(��һ������Ч),ֱ��ȥ��
     # fig = plt.figure(figsize=(12,8))
     # ax1=fig.add_subplot(211)
     # fig = sm.graphics.tsa.plot_acf(dta_diff,lags=40,ax=ax1)
     # ax2 = fig.add_subplot(212)
     # fig = sm.graphics.tsa.plot_pacf(dta_diff,lags=40,ax=ax2)
     # plt.show()

     arma_mod20 = sm.tsa.ARMA(dta_diff, (2, 1)).fit()  # ģ����ϣ�����ֺ��������Ϊ��������,�������н�ģ���
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

     predict_test = arma_mod20.predict()  # ʹ��ģ����Ԥ��δ���Ĺ�ֵ(�����ڲ�ֺ�Ļ�����Ԥ���)
     # print(predict_sunspots)  # ���Ԥ��ֵ
     predict_real = arma_mod20.predict('2091', '2100', dynamic=True)  # ʹ��ģ����Ԥ��δ���Ĺ�ֵ(�����ڲ�ֺ�Ļ�����Ԥ���)
     # print(predict_sunspots)  # ���Ԥ��ֵ

     # fig1, ax = plt.subplots(figsize=(12, 8))  # ��ͼ����ͬһ������
     # ax = dta_diff.ix['2001':].plot(ax=ax)  # ��������90������ԭʼ����ͼ
     # fig1 = arma_mod20.plot_predict('2081', '2090', dynamic=True, ax=ax, plot_insample=False)  # ������ֺ��Ԥ������ͼ
     # plt.title("Pridict figure")
     # plt.show()
     # def mean_forecast_err(y, yhat):
     #     return y.sub(yhat).mean()
     # err1 = mean_forecast_err(dta_diff, predict_test)  # ��ֺ����ʵֵ���ֺ�Ԥ�����ֵ���������;
     # err2 = mean_forecast_err(dta_diff, arma_mod20.fittedvalues)  # ��ֺ����ʵֵ���ֺ���ϵ�ֵ
     # print ("���Լ��ϵ���%f;%f"%(err1,err2))
     # dta_11 = np.array(dta_diff.ix["2080":"2090"].values)
     # dta_pri = np.array(predict_sunspots.values)
     # err1 = dta_11 - dta_pri
     # print (err.mean()-err)
     """  ��Ȼ�����Ƕ�ԭʼ�������ˡ�һ�ײ�֡�(dta = dta.diff(1))���γɵ�������������Ԥ��,
          ��ν�Ԥ������ת�ض�ԭʼ���ݵ�Ԥ������һ����Ҫ��������:����
     """
     arma_mod20_diff = pd.Series(arma_mod20.fittedvalues, copy=True)  # ��ֺ�ģ����ϵõ������ֵ(2002-2090)
     # print (arma_mod20_diff.head())
     arma_mod20_diff_cumsum1 = arma_mod20_diff.cumsum()  # ģ����Ϻ�Ĳ��ֵ�ۼ�
     # arma_mod20_diff_cumsum2 = predict_test.add(dta.shift(2))  # ��ԭһ�ײ�֣�����shift(m)���ƶ�����,�������ƶ�m��λ��;
     # print (arma_mod20_diff_cumsum.head())
     arma_mod20_resource = pd.Series(dta.ix[0], index=dta.index)  # ����ʱ�����б���,��Ų�ֻ�ԭ�����ԭʼֵ
     arma_mod20_resource = arma_mod20_resource.add(arma_mod20_diff_cumsum1, fill_value=0)  # ���ֵ������ԭʼ���Ƶ�ֵ
     # print (arma_mod20_resource.head())
     plt.plot(dta)  #  ԭʼ����ֵ(2001-2090)
     plt.plot(arma_mod20_resource, "r")  # ��ģ����Ϻ����ԭʼֵ(��ν�ġ�ʱ������Ԥ��ģ�͡���������Ԥ��ֵ)(2001-2090)
     plt.title('RMSE: %.4f' % np.sqrt(sum((arma_mod20_resource - dta) ** 2) / len(dta)))  #  ԭʼֵ��Ԥ��ֵ֮��ķ���
     plt.show()

"""������"""
example_1()
