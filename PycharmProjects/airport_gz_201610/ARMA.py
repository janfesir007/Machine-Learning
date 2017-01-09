# -*-coding:gbk-*-
# -*-coding:utf-8-*-
"""
ʱ������ģ�ͣ�"�Իع�/���/�ƶ�ƽ��"ʱ�����л��ģ��(ARIMA)
�Իع飺AR
��֣�I  �������ƺͼ�����
�ƶ�ƽ����MA
ARMA(dta,(p,q)).fit()   # ARMAģ����Ϻ�����dta��ʱ����������;����p(�Իع麯����AR��������),q���ƶ�ƽ����(MA)������
ARIMA(dta,(p,d,q)).fit() # ARIMAģ����Ϻ���������p,qͬ��,d:��ֵĴ���(ȡ0,1��2),���ǽ���������ARIMA�в��ֻ��1�ף�����ARIMAģ��ֻ��d��1�ײ�ֲ���.
ARMA��ARIMAģ�͵Ĳ�ͬ��
    ������ݲ�ƽ��,���ִ���,ARMAģ��ѵ��ʱʹ�õ��ǲ�ֺ������,��Ԥ�����Ҫ��ֻ�ԭ����.
    ARIMAģ��ѵ��ʱʹ�õ���ԭʼ����,Ԥ��������ֻ�ԭ����.�ÿ����ARIMAģ��ֻ�ṩ��������εĲ��ARIMA(p,d,q),��d���ֻ��ȡ2.
    ��ĳ������Ҫ����3�����ϲ�ֲ�ƽ����ʹ��ARMAģ��.
"""
from __future__ import print_function  # ʹ�������ʽΪ��print()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sys
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.api import qqplot

""" ��ʱ������Ԥ�⡱�����������"""
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
     dta = pd.Series(data, dtype=float)  # 90��������ѵ����
     dta.index = pd.Index(sm.tsa.datetools.dates_from_range('2001', '2090'))
     # dta.plot(figsize=(12,8))
     # plt.show()

     """
     1.�ȶ�ʱ���������ݽ���1�ײ�֣�,Ȼ������ARMAģ��(�Իع�AR/�ƶ�ƽ��MA ʱ������ģ��)���
       dta_diff = dta.diff(2)#����2ָ���ǲ�ֽ���; sm.tsa.ARMA(dta_diff, (3, 0)).fit()
     �����ȼ���
       sm.tsa.ARIMA(dta,(3,2,0)).fit()  # �����I�ںϽ�ARMAģ�͵õ�ARIMAģ��,����2ָ���ǲ�ִ���
     2.��Ҫ��ʽ��dta.diff(n) = dta - dta.shift(n)
       dta:ԭʼ����
       diff(n)��һ�ײ�֡�����:����n��λ�õ��������(���ƶ�n��λ��,����ԭʼ���ݼ�)
       shift(n)���ƶ�����,�������ƶ�n��λ��;
       �ʣ����ڷ�����ԭʼ����ʱ,�й�ʽ��dta = dta.diff(n)+dta.shift(n)
     """
     # һ.��ԭʼ���������
     dta_diff = dta.diff(2)  # shift(m)���ƶ�����,�������ƶ�m��λ��; �У�dta.diff(n) = dta - dta.shift(n)
     dta_diff.dropna(inplace=True)  # ��䲻����:diff������Ч����(��1,2����������Ч),ֱ��ȥ��

     # ��.�Բ�ֺ��������ģ�����
     arma_mod20 = sm.tsa.ARMA(dta_diff, (2, 1)).fit()  # ģ����ϣ�����ֺ��������Ϊ��������,�������н�ģ���
     # print (arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic)
     # arma_mod30 = sm.tsa.ARMA(dta_diff,(0,1)).fit()
     # print arma_mod30.aic,arma_mod30.bic,arma_mod30.hqic
     # arma_mod40 = sm.tsa.ARMA(dta_diff,(7,1)).fit()
     # print arma_mod40.aic,arma_mod40.bic,arma_mod40.hqic
     # arma_mod50 = sm.tsa.ARMA(dta_diff,(8,0)).fit()
     # print arma_mod50.aic,arma_mod50.bic,arma_mod50.hqic


     # predict_test = arma_mod20.predict('2081', '2090', dynamic=True)  # ʹ��ģ����Ԥ��δ���Ĺ�ֵ(�����ڲ�ֺ�Ļ�����Ԥ���)
     # print(predict_test)  # ���Ԥ��ֵ
     # predict_real = arma_mod20.predict('2091', '2100', dynamic=True)  # ʹ��ģ����Ԥ��δ���Ĺ�ֵ(�����ڲ�ֺ�Ļ�����Ԥ���)
     # print(predict_real)  # ���Ԥ��ֵ

     # fig1, ax = plt.subplots(figsize=(12, 8))  # ��ͼ����ͬһ������
     # ax = dta_diff.ix['2001':].plot(ax=ax)  # ��������90������ԭʼ����ͼ
     # fig1 = arma_mod20.plot_predict('2081', '2090', dynamic=True, ax=ax, plot_insample=False)  # ������ֺ��Ԥ������ͼ
     # plt.title("Pridict figure")
     # plt.show()
     # def mean_forecast_err(y, yhat):
     #     return y.sub(yhat).mean()
     # err1 = mean_forecast_err(dta_diff, predict_test)  # ��ֺ����ʵֵ���ֺ��Ԥ��ֵ���������;
     # err2 = mean_forecast_err(dta_diff, arma_mod20.fittedvalues)  # ��ֺ����ʵֵ���ֺ���ϵ�ֵ
     # print ("���Լ��ϵ���%f;%f"%(err1,err2))

     """  ��Ȼ�����Ƕ�ԭʼ�������ˡ�һ�ײ��(����Ϊ2)��(dta = dta.diff(2))���γɵ�������������Ԥ��,
          ��ν�Ԥ������ת�ض�ԭʼ���ݵ�Ԥ������һ����Ҫ��������:����
     """
     arma_mod20_diff = pd.Series(arma_mod20.fittedvalues, copy=True)  # ��ֺ�ģ����ϵõ������ֵfittedvalues(2003-2090)
     # arma_mod20_resource = pd.Series(dta.ix[0], index=dta.index)  # ����ʱ�����б���,��Ų�ֻ�ԭ�����ԭʼֵ,����ȫ����ʼ��ֵΪ��dta.ix[0]=dta����(ix)ֵΪ0���Ǹ���

     # ��.ͨ��ģ������ϣ����ݻ����ǣ����ֵ����,�����ݻ�ԭ�ز��ǰ״̬��ֵ
     arma_mod20_resource = arma_mod20_diff.add(dta.shift(2), fill_value=0)  # ���ֵ������ԭʼ���Ƶ�ֵ,��ʽ��dta = dta.diff(n)+dta.shift(n)
     arma_mod20_resource['2001'] = dta['2001']  # ����һ��ʼ��ֲ���,ǰ����������Ч,��Ҫ��ȫ
     arma_mod20_resource['2002'] = dta['2002']

     # ��ͼչʾ
     plt.plot(dta)  #  ԭʼ����ֵ(2001-2090)
     plt.plot(arma_mod20_resource, "r")  # ��ģ����Ϻ����ԭʼֵ(��ν�ġ�ʱ������Ԥ��ģ�͡������������ֵ)(2001-2090)
     plt.title('RMSE: %.4f' % np.sqrt(sum((arma_mod20_resource - dta) ** 2) / len(dta)))  #  ԭʼֵ�����ֵ֮��ķ���
     plt.show()

""" ��ʱ������Ԥ�⡱��������Ԥ��,����ֵ�ǡ�ԭʼ���ݼ�+��һʱ��Ԥ��ֵ�����Ա����Ԥ��"""
def example_2(datas):  # datas:list���͵�ѵ������
        is2diff = False  # ��������Ƿ񾭹����β�ִ���
        ts = pd.Series(datas, dtype=float)  # list���͵�ѵ��������ת��Ϊ��ʱ�����С�����
        ts.index = pd.date_range(start="2001", periods=len(datas), freq="12M")  # periods���������� ;freq��ʱ�������ļ��
        # a1 = ts.index[-5:]  # �ӵ�����5��λ�ÿ�ʼ�����ȡ������
        # a2 = ts.index[-5::][::n]  # �ӵ�����5��λ�ÿ�ʼ����ÿ��n-1�����س�ȡ
        # a3 = ts.index[-5::][::-n]  # �ӵ�����5��λ�ÿ�ʼ����ÿ��n-1�����س�ȡ,Ȼ��������ת
        dta = ts
        # fig = plt.figure(figsize=(12, 8))  # �ڻ���plt��Ȧ��һ�黭��fig
        # ax1 = fig.add_subplot(311)  # �����Ϸ�һ������ϵax1
        # dta.plot(ax=ax1)  # ����ϵax1��������
        # ax2 = fig.add_subplot(312)  # �������ٷ�һ������ϵax2
        # sm.graphics.tsa.plot_acf(dta, ax=ax2, lags=40)  # ����ϵax2��������
        # ax3 = fig.add_subplot(313)  # �������ٷ�һ������ϵax3
        # sm.graphics.tsa.plot_pacf(dta, ax=ax3, lags=40)  # ����ϵax3��������
        # plt.show()  # ��ʾ���������ϵ���������ϵ


        # һ.���ʱ�����������Ƿ�ƽ��
        dfoutput = testStationarity(dta)  # ƽ���Լ�����dfoutput,���ص�����������ʱ������
        print ("ԭʼ����p-value:", dfoutput["p-value"])  # p-valueֵԽСԽƽ��
        # ��.��ʱ���������ݲ�ƽ��,���ԭʼ���������,Ŀ���ǻ��ƽ����������
        if (dfoutput['p-value'] >= 0.05):
            dta_diff1 = dta.diff(1)  # shift(m)���ƶ�����,�������ƶ�m��λ��; diff(n)��һ�ײ�֡�����:����n��λ�õ��������(���ƶ�n��λ��,���������)���У�dta.diff(n) = dta - dta.shift(n)
            dta_diff1.dropna(inplace=True)  # ��䲻����:Ӧ����diff������Ч����(ǰ��������Ч),ֱ��ȥ��
            # plt.subplot(111)
            # dta_diff1.plot(figsize=(12, 8))
            # plt.show()
            dfoutput = testStationarity(dta_diff1)  # �ٴμ�������Ƿ�ƽ��.ƽ���Լ�����dfoutput,���ص�����������ʱ������
            print("һ��1�ײ�ִ����p-value:", dfoutput["p-value"])
            if (dfoutput['p-value'] >= 0.05):  # �����ݲ�ƽ�ȼ������еڶ���.
                dta_diff2 = dta_diff1.diff(1)
                dta_diff2.dropna(inplace=True)
                # plt.subplot(111)
                # dta_diff2.plot(figsize=(12, 8))
                # plt.show()
                dfoutput_end = testStationarity(dta_diff2)
                print(dfoutput_end)
                if dfoutput_end['p-value'] > dfoutput['p-value']:
                    dta = dta_diff1
                    print("���棺�����β�ֺ�ģ��pֵ��Ȼƫ��.��ѡ��һ��1�ײ�ִ���!")
                else:
                    dta = dta_diff2
                    is2diff = True
                    print("��ʾ����������1�ײ�ִ���!")
            else:
                dta = dta_diff1
                # print ("����һ��1�ײ�ִ���!")


        # ��.��ƽ���������ݣ�����������ִ���������ģ�����,��������Ҫ����Ĺؼ�������ARMA(p,q)�����Ų���p,q��ȷ������ʱ����ģ��Ҳ��ȷ����
        p, q, arma_model = proper_model_pq(dta, len(datas)/10)  # p=3,q=5,ģ��ѵ��ʹ�ò�ֺ������dta
        # arma_model = ARMA(dta, (p, q)).fit(disp=-1)# ģ����ϣ�����ֺ��������Ϊ��������,�������н�ģ���

        # ��.ģ�ͼ���,����ģ���Ƿ����3�ּ��� ����������/D-W����/�в���̫�ֲ�����

        # 1.�в�(resid)�İ��������飺�Բв�Ĺ������н���LB����,�ж����Ƿ��Ǹ�˹������.�������,��ô��˵��ARMAģ�Ͳ�����.��Ҫ����������ģ�����
        # ����Ľ�����ǿ����һ��(Proֵ)ǰʮ���еļ�����ʣ�һ��۲��ͺ�1~12�ף�������������С�ڸ�����������ˮƽ��
        # ����0.05��0.10�Ⱦ;ܾ�ԭ���裬��ԭ���������ϵ��Ϊ��,�ǰ���������.����ProֵԽ��˵���ǰ���������.
        r, m, n = sm.tsa.acf(arma_model.resid.values.squeeze(), qstat=True)
        data = np.c_[range(1, 41), r[1:], m, n]
        table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
        print(table.set_index('lag')[:12])  # ���ǰ12��
        # 2.�±�-��ɭ����,���D-W���飬��Ŀǰ�������������õķ���������ֻ�����ڼ���һ��������ԡ�
        # �������в��Ƿ������.DWֵ�ӽ��ڣ�ʱ,�򲻴��ڣ�һ�ף��������,��ģ�ͺ���.
        print("DW����ֵ��", sm.stats.durbin_watson(arma_model.resid.values), "\n\n")
        #3.ʹ��QQͼ��ֱ����֤�в��Ƿ�����ĳ����,���õ��Ǽ��������Ƿ���������̬�ֲ���
        # resid = arma_model.resid  # �в�
        # fig = plt.figure(figsize=(12, 8))
        # ax = fig.add_subplot(111)
        # qqplot(resid, line='q', ax=ax, fit=True)
        # plt.show()


        # ��.ģ�ͺ��������Ԥ��:��һʱ����Ԥ��ֵ,ʹ��ģ����Ԥ��δ���Ĺ�ֵ
        predict_values = arma_model.predict(str(len(datas) + 2001), str(len(datas) + 2001), dynamic=True)  # ����������Serise
        # forecast_values = arma_model.forecast(1)  # ����������tuple,forecast()������predict()
        # predict_test = arma_model.predict('2081', '2090', dynamic=True)  # Ԥ���������ʱ���
        # arma_model_fit_values = pd.Series(arma_model.fittedvalues, copy=True)  # ��ֺ�ģ����ϵõ������ֵ(2002-2080)


        # ���ģ��ѵ��ʹ�õ��Ǿ�����ִ��������,��ͨ��ģ����Ԥ���,��Ҫ�����ݻ�ԭ�ز��ǰ״̬��ֵ.�����Ĳ�ֲ�����Ĭ����n��1�ף�1�ף��ƶ�1��λ��������,��ͬ��/�׻�ԭ������һ��
        if is2diff == False:  # 1��1�ײ�ֻ�ԭ
            predict_values[0] = predict_values[0] + ts[-1]
        else:  # 2��1�ײ�ֻ�ԭ
            predict_values[0] = predict_values[0] + dta_diff1[-1] + ts[-1]

        #�µ�Ԥ��ֵ����ԭʼ���ݼ�,��ԭʼ���ݼ�һ����Ϊ��һ��Ԥ������뼯
        new_datas = list(ts.append(predict_values))
        return new_datas  # ����ֵ��list����


"""
     ���ڸ��������ʱ������,���ǿ���ͨ���۲������ͼ��ƫ���ͼ������ģ��ʶ��(ȷ��p,qֵ),
     ��������Ҫ������ʱ���������϶�,����ҪԤ��ÿֻ��Ʊ������,���ǾͲ��������ȥ������.
     ��ʱ���ǿ�������BIC׼��ʶ��ģ�͵�p, qֵ,ͨ����ΪBICֵԽС��ģ����Ը���.
     BIC׼���ۺϿ����˲в��С���Ա����ĸ���,�в�ԽСBICֵԽС,�Ա�������Խ��BICֵԽС.
     �����proper_model(data_ts, maxLag)�����������ṩ��ƽ����������,���������ģ�͵�p,qֵ.
"""
def proper_model_pq(data_ts, maxLag):
    init_bic = sys.maxint
    init_p = 0
    init_q = 0
    init_properModel = None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            model = ARMA(data_ts, order=(p, q))
            try:
                results_ARMA = model.fit(disp=-1)
            except:
                continue
            bic = results_ARMA.bic
            if bic < init_bic:
                init_p = p
                init_q = q
                init_properModel = results_ARMA
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
    pridict_data = example_2(pridict_data)  # pridict_data��Ԥ���õ����������ݣ�ԭʼ����+Ԥ�����ݣ�,list����

# ��ͼչʾ
# ��list����ת����Serise����,periods��ʾ���ݸ���,����Ӧ��������,freq=��12M��������ʱ����12��Month
data_real = pd.Series(train_data + test_data, index=pd.date_range(start="2001", periods=len(test_data)+len(train_data),freq="12M"), dtype=float)
pridict_data = pd.Series(pridict_data, index=pd.date_range(start="2001", periods=len(pridict_data), freq="12M"), dtype=float)
# a=pridict_data[0:1] # Serise����,ͨ����ֵ��������ͨ����һ��,����������ֵ1����Ӧ��ֵ
# b=pridict_data["2001":"2002"] # Serise����,ͨ����ʱ�䡱�����͡���ֵ��������һ��,������ʱ��������2002������Ӧ��ֵ

fig = plt.figure(figsize=(12, 8))  # ����
ax1 = fig.add_subplot(111)  # ����ϵ
data_real.plot(ax=ax1)  # ԭʼ����ֵ(2001-)
pridict_data[len(train_data)-1:].plot(ax=ax1, color="r")  # Ԥ��ֵ(2080-)
plt.title('RMSE: %.4f' % (np.sqrt(sum((np.array(test_data) - np.array(list(pridict_data[len(train_data):]))) ** 2) / len(test_data))))  # ���������RMSE:ƽ���������ݵ�ʵ��ֵ��Ԥ��ֵ֮���ƫ��
plt.show()