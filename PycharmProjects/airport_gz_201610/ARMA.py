# -*-coding:gbk-*-
# -*-coding:utf-8-*-
"""
时间序列模型："自回归/差分/移动平均"时间序列混合模型(ARIMA)
自回归：AR
差分：I  消除趋势和季节性
移动平均：MA
ARMA(dta,(p,q)).fit()   # ARMA模型拟合函数：dta：时间序列数据;参数p(自回归函数（AR）的条件),q是移动平均数(MA)的条件
ARIMA(dta,(p,d,q)).fit() # ARIMA模型拟合函数：参数p,q同上,d:差分的次数(取0,1或2),不是阶数！而且ARIMA中差分只是1阶！即：ARIMA模型只有d次1阶差分操作.
ARMA和ARIMA模型的不同：
    如果数据不平稳,需差分处理,ARMA模型训练时使用的是差分后的数据,故预测后还需要差分还原操作.
    ARIMA模型训练时使用的是原始数据,预测后无需差分还原操作.该库对于ARIMA模型只提供了最高两次的差分ARIMA(p,d,q),即d最大只能取2.
    若某数据需要经过3次以上差分才平稳则使用ARMA模型.
"""
from __future__ import print_function  # 使得输出格式为：print()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sys
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.api import qqplot

""" “时间序列预测”进行数据拟合"""
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
     dta = pd.Series(data, dtype=float)  # 90个数据作训练集
     dta.index = pd.Index(sm.tsa.datetools.dates_from_range('2001', '2090'))
     # dta.plot(figsize=(12,8))
     # plt.show()

     """
     1.先对时间序列数据进行1阶差分，,然后再用ARMA模型(自回归AR/移动平均MA 时间序列模型)拟合
       dta_diff = dta.diff(2)#参数2指的是差分阶数; sm.tsa.ARMA(dta_diff, (3, 0)).fit()
     并不等价与
       sm.tsa.ARIMA(dta,(3,2,0)).fit()  # 将差分I融合进ARMA模型得到ARIMA模型,参数2指的是差分次数
     2.重要公式：dta.diff(n) = dta - dta.shift(n)
       dta:原始数据
       diff(n)”一阶差分“处理:隔开n个位置的数据相减(先移动n个位置,再由原始数据减)
       shift(n)：移动操作,将数据移动n个位置;
       故：后期返回至原始数据时,有公式：dta = dta.diff(n)+dta.shift(n)
     """
     # 一.对原始数据做差分
     dta_diff = dta.diff(2)  # shift(m)：移动操作,将数据移动m个位置; 有：dta.diff(n) = dta - dta.shift(n)
     dta_diff.dropna(inplace=True)  # 这句不能少:diff后有无效数据(第1,2两个数据无效),直接去除

     # 二.对差分后的数据做模型拟合
     arma_mod20 = sm.tsa.ARMA(dta_diff, (2, 1)).fit()  # 模型拟合：将差分后的数据作为样本数据,对它进行建模拟合
     # print (arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic)
     # arma_mod30 = sm.tsa.ARMA(dta_diff,(0,1)).fit()
     # print arma_mod30.aic,arma_mod30.bic,arma_mod30.hqic
     # arma_mod40 = sm.tsa.ARMA(dta_diff,(7,1)).fit()
     # print arma_mod40.aic,arma_mod40.bic,arma_mod40.hqic
     # arma_mod50 = sm.tsa.ARMA(dta_diff,(8,0)).fit()
     # print arma_mod50.aic,arma_mod50.bic,arma_mod50.hqic


     # predict_test = arma_mod20.predict('2081', '2090', dynamic=True)  # 使用模型来预测未来的估值(数据在差分后的基础上预测的)
     # print(predict_test)  # 输出预测值
     # predict_real = arma_mod20.predict('2091', '2100', dynamic=True)  # 使用模型来预测未来的估值(数据在差分后的基础上预测的)
     # print(predict_real)  # 输出预测值

     # fig1, ax = plt.subplots(figsize=(12, 8))  # 两图放在同一坐标上
     # ax = dta_diff.ix['2001':].plot(ax=ax)  # 画出所有90条数据原始数据图
     # fig1 = arma_mod20.plot_predict('2081', '2090', dynamic=True, ax=ax, plot_insample=False)  # 画出差分后的预测数据图
     # plt.title("Pridict figure")
     # plt.show()
     # def mean_forecast_err(y, yhat):
     #     return y.sub(yhat).mean()
     # err1 = mean_forecast_err(dta_diff, predict_test)  # 差分后的真实值与差分后的预测值：计算误差;
     # err2 = mean_forecast_err(dta_diff, arma_mod20.fittedvalues)  # 差分后的真实值与差分后拟合的值
     # print ("测试集上的误差：%f;%f"%(err1,err2))

     """  当然以上是对原始数据做了”一阶差分(参数为2)”(dta = dta.diff(2))后形成的新数据所做的预测,
          如何将预测结果反转回对原始数据的预测是下一步需要做的事情:如下
     """
     arma_mod20_diff = pd.Series(arma_mod20.fittedvalues, copy=True)  # 差分后模型拟合得到的拟合值fittedvalues(2003-2090)
     # arma_mod20_resource = pd.Series(dta.ix[0], index=dta.index)  # 定义时间序列变量,存放差分还原后的类原始值,首先全部初始化值为：dta.ix[0]=dta索引(ix)值为0的那个数

     # 三.通过模型做拟合（数据基础是：差分值）后,将数据还原回差分前状态的值
     arma_mod20_resource = arma_mod20_diff.add(dta.shift(2), fill_value=0)  # 差分值倒回至原始趋势的值,公式：dta = dta.diff(n)+dta.shift(n)
     arma_mod20_resource['2001'] = dta['2001']  # 由于一开始差分操作,前两个数据无效,需要补全
     arma_mod20_resource['2002'] = dta['2002']

     # 作图展示
     plt.plot(dta)  #  原始样本值(2001-2090)
     plt.plot(arma_mod20_resource, "r")  # 经模型拟合后的类原始值(所谓的“时间序列预测模型”所产生的拟合值)(2001-2090)
     plt.title('RMSE: %.4f' % np.sqrt(sum((arma_mod20_resource - dta) ** 2) / len(dta)))  #  原始值与拟合值之间的方差
     plt.show()

""" “时间序列预测”进行数据预测,返回值是“原始数据集+下一时刻预测值”：以便后续预测"""
def example_2(datas):  # datas:list类型的训练数据
        is2diff = False  # 标记数据是否经过两次差分处理
        ts = pd.Series(datas, dtype=float)  # list类型的训练集数据转化为“时间序列”类型
        ts.index = pd.date_range(start="2001", periods=len(datas), freq="12M")  # periods：索引长度 ;freq：时间索引的间隔
        # a1 = ts.index[-5:]  # 从倒数第5个位置开始往后抽取所有数
        # a2 = ts.index[-5::][::n]  # 从倒数第5个位置开始往后每隔n-1个数地抽取
        # a3 = ts.index[-5::][::-n]  # 从倒数第5个位置开始往后每隔n-1个数地抽取,然后在逆序反转
        dta = ts
        # fig = plt.figure(figsize=(12, 8))  # 在画板plt上圈出一块画布fig
        # ax1 = fig.add_subplot(311)  # 画布上放一个坐标系ax1
        # dta.plot(ax=ax1)  # 坐标系ax1上描数据
        # ax2 = fig.add_subplot(312)  # 画布上再放一个坐标系ax2
        # sm.graphics.tsa.plot_acf(dta, ax=ax2, lags=40)  # 坐标系ax2上描数据
        # ax3 = fig.add_subplot(313)  # 画布上再放一个坐标系ax3
        # sm.graphics.tsa.plot_pacf(dta, ax=ax3, lags=40)  # 坐标系ax3上描数据
        # plt.show()  # 显示整个画板上的所有坐标系


        # 一.检测时间序列数据是否平稳
        dfoutput = testStationarity(dta)  # 平稳性检验结果dfoutput,返回的数据类型是时间序列
        print ("原始数据p-value:", dfoutput["p-value"])  # p-value值越小越平稳
        # 二.若时间序列数据不平稳,则对原始数据做差分,目的是获得平稳序列数据
        if (dfoutput['p-value'] >= 0.05):
            dta_diff1 = dta.diff(1)  # shift(m)：移动操作,将数据移动m个位置; diff(n)”一阶差分“处理:隔开n个位置的数据相减(先移动n个位置,再数据相减)即有：dta.diff(n) = dta - dta.shift(n)
            dta_diff1.dropna(inplace=True)  # 这句不能少:应该是diff后有无效数据(前两个数无效),直接去除
            # plt.subplot(111)
            # dta_diff1.plot(figsize=(12, 8))
            # plt.show()
            dfoutput = testStationarity(dta_diff1)  # 再次检测数据是否平稳.平稳性检验结果dfoutput,返回的数据类型是时间序列
            print("一次1阶差分处理后p-value:", dfoutput["p-value"])
            if (dfoutput['p-value'] >= 0.05):  # 若数据不平稳继续进行第二步.
                dta_diff2 = dta_diff1.diff(1)
                dta_diff2.dropna(inplace=True)
                # plt.subplot(111)
                # dta_diff2.plot(figsize=(12, 8))
                # plt.show()
                dfoutput_end = testStationarity(dta_diff2)
                print(dfoutput_end)
                if dfoutput_end['p-value'] > dfoutput['p-value']:
                    dta = dta_diff1
                    print("警告：经两次差分后模型p值仍然偏大.已选择一次1阶差分处理!")
                else:
                    dta = dta_diff2
                    is2diff = True
                    print("提示：经过两次1阶差分处理!")
            else:
                dta = dta_diff1
                # print ("经过一次1阶差分处理!")


        # 三.对平稳序列数据（往往经过差分处理）数据做模型拟合,其中首先要解决的关键问题是ARMA(p,q)中最优参数p,q的确定（此时最优模型也就确定）
        p, q, arma_model = proper_model_pq(dta, len(datas)/10)  # p=3,q=5,模型训练使用差分后的数据dta
        # arma_model = ARMA(dta, (p, q)).fit(disp=-1)# 模型拟合：将差分后的数据作为样本数据,对它进行建模拟合

        # 四.模型检验,检验模型是否合理：3种检验 白噪声检验/D-W检验/残差正太分布检验

        # 1.残差(resid)的白噪声检验：对残差的估计序列进行LB检验,判断其是否是高斯白噪声.如果不是,那么就说明ARMA模型不合理.需要重做第三步模型拟合
        # 检验的结果就是看最后一列(Pro值)前十二行的检验概率（一般观察滞后1~12阶），如果检验概率小于给定的显著性水平，
        # 比如0.05、0.10等就拒绝原假设，其原假设是相关系数为零,非白噪声序列.即：Pro值越大说明是白噪声序列.
        r, m, n = sm.tsa.acf(arma_model.resid.values.squeeze(), qstat=True)
        data = np.c_[range(1, 41), r[1:], m, n]
        table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
        print(table.set_index('lag')[:12])  # 输出前12行
        # 2.德宾-沃森检验,简称D-W检验，是目前检验自相关性最常用的方法，但它只适用于检验一阶自相关性。
        # 这里检验残差是否自相关.DW值接近于２时,则不存在（一阶）自相关性,即模型合理.
        print("DW检验值：", sm.stats.durbin_watson(arma_model.resid.values), "\n\n")
        #3.使用QQ图，直观验证残差是否来自某个分,常用的是检验数据是否来自于正态分布。
        # resid = arma_model.resid  # 残差
        # fig = plt.figure(figsize=(12, 8))
        # ax = fig.add_subplot(111)
        # qqplot(resid, line='q', ax=ax, fit=True)
        # plt.show()


        # 五.模型合理则进行预测:下一时间点的预测值,使用模型来预测未来的估值
        predict_values = arma_model.predict(str(len(datas) + 2001), str(len(datas) + 2001), dynamic=True)  # 返回类型是Serise
        # forecast_values = arma_model.forecast(1)  # 返回类型是tuple,forecast()类似于predict()
        # predict_test = arma_model.predict('2081', '2090', dynamic=True)  # 预测连续多个时间点
        # arma_model_fit_values = pd.Series(arma_model.fittedvalues, copy=True)  # 差分后模型拟合得到的拟合值(2002-2080)


        # 如果模型训练使用的是经过差分处理的数据,则通过模型做预测后,需要将数据还原回差分前状态的值.本例的差分操作都默认是n次1阶（1阶：移动1个位置做差）差分,不同次/阶还原操作不一样
        if is2diff == False:  # 1次1阶差分还原
            predict_values[0] = predict_values[0] + ts[-1]
        else:  # 2次1阶差分还原
            predict_values[0] = predict_values[0] + dta_diff1[-1] + ts[-1]

        #新的预测值加入原始数据集,和原始数据集一起作为下一次预测的输入集
        new_datas = list(ts.append(predict_values))
        return new_datas  # 返回值是list类型


"""
     对于个数不多的时序数据,我们可以通过观察自相关图和偏相关图来进行模型识别(确定p,q值),
     倘若我们要分析的时序数据量较多,例如要预测每只股票的走势,我们就不可能逐个去调参了.
     这时我们可以依据BIC准则识别模型的p, q值,通常认为BIC值越小的模型相对更优.
     BIC准则综合考虑了残差大小和自变量的个数,残差越小BIC值越小,自变量个数越少BIC值越小.
     下面的proper_model(data_ts, maxLag)函数：根据提供的平稳序列数据,求出其最优模型的p,q值.
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
    pridict_data = example_2(pridict_data)  # pridict_data：预测后得到的完整数据（原始数据+预测数据）,list类型

# 作图展示
# 将list类型转化成Serise类型,periods表示数据个数,即对应索引个数,freq=“12M”：索引时间间隔12个Month
data_real = pd.Series(train_data + test_data, index=pd.date_range(start="2001", periods=len(test_data)+len(train_data),freq="12M"), dtype=float)
pridict_data = pd.Series(pridict_data, index=pd.date_range(start="2001", periods=len(pridict_data), freq="12M"), dtype=float)
# a=pridict_data[0:1] # Serise类型,通过数值索引和普通索引一样,不包含索引值1所对应的值
# b=pridict_data["2001":"2002"] # Serise类型,通过“时间”索引和“数值”索引不一样,它包含时间索引“2002”所对应的值

fig = plt.figure(figsize=(12, 8))  # 画布
ax1 = fig.add_subplot(111)  # 坐标系
data_real.plot(ax=ax1)  # 原始样本值(2001-)
pridict_data[len(train_data)-1:].plot(ax=ax1, color="r")  # 预测值(2080-)
plt.title('RMSE: %.4f' % (np.sqrt(sum((np.array(test_data) - np.array(list(pridict_data[len(train_data):]))) ** 2) / len(test_data))))  # 均方根误差RMSE:平均单个数据的实际值与预测值之间的偏差
plt.show()