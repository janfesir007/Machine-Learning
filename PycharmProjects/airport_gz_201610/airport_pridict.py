# -*-encoding:gbk-*-
from operator import itemgetter  # 引入排序操作
import numpy as np
"""遍历文件夹"""
def list_files(dir_path):
    import os
    list_file = os.listdir(dir_path)
    list_file.sort()
    return list_file


"""
用过去三天（11-13日/22-24日）同一时间段的连接数的平均值作为14日/25日的预测值
"""
def get_pridict_through_avg():
    import time
    from operator import itemgetter
    dir_file = "airport_Dataset/every_points_WIFI_AP_Records/"
    list_file = list_files(dir_file)  # 调用函数,遍历文件夹下面的文件
    list_file.sort()
    for file_name in list_file:
        wifi_records_10min = {}  # {(wifi名称,10min时间段):连接数量 }:这是整个预测的结果Y,我们还需要找到影响该结果的输入变量X(特征)
        with open(dir_file + file_name, "r") as f:
            datalines = f.readlines()
            for line in datalines:
                arraylines = line.replace("\n", "")
                arraydata = arraylines.split(",")

                wifi_name = arraydata[0]
                connect_count = int(arraydata[1])
                date = arraydata[2]
                date_10min = date[:15]  # 如：将2016-9-10-18-50～59统统计为2016-9-10-18-5：表示为这10分钟的时间段
                stru_date_10min = time.strptime(date_10min, "%Y-%m-%d-%H-%M")  # 转换为可比较的时间
                records_id = (wifi_name, date_10min)
                if not records_id in wifi_records_10min:
                    # if time.strptime("2016-9-11-15-0", "%Y-%m-%d-%H-%M")<stru_date_10min <time.strptime("2016-9-11-17-5", "%Y-%m-%d-%H-%M"):
                    #     wifi_records_10min_E1[records_id] = connect_count  # 2016-9-10这一天没记录
                    if time.strptime("2016-09-11-15-0", "%Y-%m-%d-%H-%M") <= stru_date_10min <= time.strptime(
                            "2016-9-11-17-5", "%Y-%m-%d-%H-%M"):
                        wifi_records_10min[records_id] = connect_count
                    if time.strptime("2016-09-12-15-0", "%Y-%m-%d-%H-%M") <= stru_date_10min <= time.strptime(
                            "2016-9-12-17-5", "%Y-%m-%d-%H-%M"):
                        wifi_records_10min[records_id] = connect_count
                    if time.strptime("2016-09-13-15-0", "%Y-%m-%d-%H-%M") <= stru_date_10min <= time.strptime(
                            "2016-9-13-17-5", "%Y-%m-%d-%H-%M"):
                        wifi_records_10min[records_id] = connect_count
                else:
                    wifi_records_10min[records_id] += connect_count

        avg_wifi_records_10min = {}
        for dic in wifi_records_10min:
            wifi = dic[0]
            date_time = dic[1]
            connect_counts = wifi_records_10min[dic]
            new_date_time = "2016-9-14" + date_time[10:15]
            uid = (wifi, new_date_time)
            if not uid in avg_wifi_records_10min:
                avg_wifi_records_10min[uid] = connect_counts
            else:
                avg_wifi_records_10min[uid] += connect_counts
        avg_wifi_records_10min_tuple = sorted(avg_wifi_records_10min.iteritems(), key=itemgetter(0))  # 按键排序,返回元组

        f_pri = open("airport_Dataset/pridict_results/airport_gz_passenger_predict.csv", "a+")
        if len(f_pri.readlines()) > 0:
            # f_pri.writelines('passengerCount,WIFIAPTag,slice10min'+"\n")
            for dic_line in avg_wifi_records_10min_tuple:
                list_result = [
                    str(format(dic_line[1] / 3.0, ".1f")) + "," + dic_line[0][0] + "," + dic_line[0][1] + "\n"]
                f_pri.writelines(list_result)
        else:
            f_pri.writelines('passengerCount,WIFIAPTag,slice10min' + "\n")
            for dic_line in avg_wifi_records_10min_tuple:
                list_result = [
                    str(format(dic_line[1] / 3.0, ".1f")) + "," + dic_line[0][0] + "," + dic_line[0][1] + "\n"]
                f_pri.writelines(list_result)
        f_pri.close()
        print file_name + "文件完成预测！\n"
    print "全部完成！"

"""
用过去三天（11-13日/22-24日）同一时间段的连接数的带不同权重的期望值(11日的连接数×0.2+12日的×0.3+13日的×0.5)作为14日的预测值
"""
def get_pridict_through_weight_avg():
    import time
    from operator import itemgetter
    dir_file = "airport_Dataset/Second_round/every_points_WIFI_AP_Records/"
    list_file = list_files(dir_file)  # 调用函数,遍历文件夹下面的文件
    list_file.sort()
    for file_name in list_file:
        weight_wifi_records_10min = {}  # 3天15：00-17：59{(wifi名称,10min时间段):连接数量(带权重归一化后) }
        with open(dir_file + file_name, "r") as f:
            datalines = f.readlines()
            for line in datalines:
                arraylines = line.replace("\n", "")
                arraydata = arraylines.split(",")

                wifi_name = arraydata[0]
                connect_count = int(arraydata[1])
                date = arraydata[2]
                date_10min = date[:15]  # 如：将2016-9-10-18-50～59统统计为2016-9-10-18-5：表示为这10分钟的时间段
                stru_date_10min = time.strptime(date_10min, "%Y-%m-%d-%H-%M")  # 转换为可比较的时间
                records_id = (wifi_name, date_10min)

                # if time.strptime("2016-9-11-15-0", "%Y-%m-%d-%H-%M")<stru_date_10min <time.strptime("2016-9-11-17-5", "%Y-%m-%d-%H-%M"):
                #     wifi_records_10min_E1[records_id] = connect_count  # 2016-9-10这一天没记录
                if time.strptime("2016-09-22-15-0", "%Y-%m-%d-%H-%M") <= stru_date_10min <= time.strptime(
                            "2016-9-22-17-5", "%Y-%m-%d-%H-%M"):
                    if not records_id in weight_wifi_records_10min:
                        weight_wifi_records_10min[records_id] = connect_count*0.2
                    else:
                        weight_wifi_records_10min[records_id] += connect_count * 0.2


                if time.strptime("2016-09-23-15-0", "%Y-%m-%d-%H-%M") <= stru_date_10min <= time.strptime(
                            "2016-9-23-17-5", "%Y-%m-%d-%H-%M"):
                    if not records_id in weight_wifi_records_10min:
                        weight_wifi_records_10min[records_id] = connect_count*0.3
                    else:
                        weight_wifi_records_10min[records_id] += connect_count * 0.3
                if time.strptime("2016-09-24-15-0", "%Y-%m-%d-%H-%M") <= stru_date_10min <= time.strptime(
                            "2016-9-24-17-5", "%Y-%m-%d-%H-%M"):
                    if not records_id in weight_wifi_records_10min:
                        weight_wifi_records_10min[records_id] = connect_count*0.5
                    else:
                        weight_wifi_records_10min[records_id] += connect_count * 0.5
        weight_avg_wifi_records_10min = {}
        for dic in weight_wifi_records_10min:
            wifi = dic[0]
            date_time = dic[1]
            connect_counts = weight_wifi_records_10min[dic]
            new_date_time = "2016-9-25" + date_time[10:15]
            uid = (wifi, new_date_time)
            if not uid in weight_avg_wifi_records_10min:
                weight_avg_wifi_records_10min[uid] = connect_counts
            else:
                weight_avg_wifi_records_10min[uid] += connect_counts
        avg_wifi_records_10min_tuple = sorted(weight_avg_wifi_records_10min.iteritems(), key=itemgetter(0))  # 按键排序,返回元组

        f_pri = open("airport_Dataset/Second_round/pridict_results/airport_gz_passenger_predict.csv", "a+")
        if len(f_pri.readlines()) > 0:
            # f_pri.writelines('passengerCount,WIFIAPTag,slice10min'+"\n")
            for dic_line in avg_wifi_records_10min_tuple:
                list_result = [
                    str(format(dic_line[1], ".1f")) + "," + dic_line[0][0] + "," + dic_line[0][1] + "\n"]
                f_pri.writelines(list_result)
        else:
            f_pri.writelines('passengerCount,WIFIAPTag,slice10min' + "\n")
            for dic_line in avg_wifi_records_10min_tuple:
                list_result = [
                    str(format(dic_line[1], ".1f")) + "," + dic_line[0][0] + "," + dic_line[0][1] + "\n"]
                f_pri.writelines(list_result)
        f_pri.close()
        print file_name + "文件完成预测！\n"
    print "全部完成！"



"""
细化处理数据
"""

"""获取某个区域每个wifi在10分钟间隔内的连接数,并按“wifi+日期”排好序,以元组形式存储"""
def get_wifi_records_10min_tuple(file_path):
    wifi_records_10min = {}  # {(wifi名称,10min时间段):连接数量 }:这是整个预测的结果Y,我们还需要找到影响该结果的输入变量X(特征)
    with open(file_path, "r") as f:
        datalines = f.readlines()
        for line in datalines:
            arraylines = line.replace("\n", "")
            arraydata = arraylines.split(",")

            wifi_name = arraydata[0]
            connect_count = int(arraydata[1])
            date = arraydata[2]
            date_10min = date[:15]+"0"  # 如：将2016-9-10-18-50～59统统计为2016-9-10-18-50：表示为这10分钟的时间段
            records_id = (wifi_name, date_10min)
            if not records_id in wifi_records_10min:
                wifi_records_10min[records_id] = connect_count
            else:
                wifi_records_10min[records_id] += connect_count
    wifi_records_10min_tuple = sorted(wifi_records_10min.iteritems(), key=itemgetter(0))  # 按键(wifi_name,date)排序,字典类型则以元组类型返回
    return wifi_records_10min_tuple

"""某个区域中所有wifi随着时间变化(3天时间内每隔10分钟)的连接数,以.csv格式存储"""
def get_everywifi_10min_records_csv(file_out_path,wifi_10min_tuple):
    wifi_names = []
    for record in wifi_10min_tuple:
        wifi_names.append(record[0][0])
    uniq_wifi_names = list(set(wifi_names))
    for wifi in uniq_wifi_names:
        every_wifi_10min_records = []
        for record_tuple in wifi_10min_tuple:
            if record_tuple[0][0] == wifi:
                every_wifi_10min_records.append(record_tuple[0][0]+","+record_tuple[0][1]+","+str(record_tuple[1])+"\n")
        with open(file_out_path+"%s.csv" % wifi, "w") as f:
            f.writelines(every_wifi_10min_records)

"""可视化：某个区域中所有wifi随着时间变化(3天时间内每隔10分钟)的连接数,以图表形式存储"""
def everywifi_10min_figures(input_csv_path, out_figure_path):
    # 第一种画图方法：使用matplotlib.pylab画图库
    # import matplotlib.pylab as pltl
    # from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    #
    # xmajorLocator = MultipleLocator(20)  # 将x主刻度标签设置为20的倍数
    # xmajorFormatter = FormatStrFormatter('%1.0f')  # 设置x轴标签文本的格式
    # xminorLocator = MultipleLocator(5)  # 将x轴次刻度标签设置为5的倍数
    #
    # ymajorLocator = MultipleLocator(20)  # 将y轴主刻度标签设置为0.5的倍数
    # ymajorFormatter = FormatStrFormatter('%1.1f')  # 设置y轴标签文本的格式
    # yminorLocator = MultipleLocator(5)  # 将此y轴次刻度标签设置为0.1的倍数
    #
    # ax = pltl.subplot(111)  # 注意:一般都在ax中设置,不再plot中设置
    #
    # # 设置主刻度标签的位置,标签文本的格式
    # ax.xaxis.set_major_locator(xmajorLocator)
    # ax.xaxis.set_major_formatter(xmajorFormatter)
    # ax.yaxis.set_major_locator(ymajorLocator)
    # ax.yaxis.set_major_formatter(ymajorFormatter)
    #
    # # 显示次刻度标签的位置,没有标签文本
    # ax.xaxis.set_minor_locator(xminorLocator)
    # ax.yaxis.set_minor_locator(yminorLocator)
    #
    # ax.xaxis.grid(True, which='major')  # x坐标轴的网格使用主刻度
    # ax.yaxis.grid(True, which='minor')  # y坐标轴的网格使用次刻度

    # 第二种画图方法：使用matplotlib.pyplot画图库
    import matplotlib.pyplot as plt
    list_file = list_files(input_csv_path)
    for wifi in list_file:  # 遍历文件夹下的所有文件,将.csv文件画成图表文件
        with open(input_csv_path + wifi, "r") as f:
            context = f.readlines()
            y = []
            i = 0
            for line in context:
                i += 1
                record = line.replace("\n", "")
                wifi_record = record.split(",")
                if i > 31:  # 去掉9.10日的记录,从9.11日0点开始记录
                    y.append(wifi_record[2])
        x = np.arange(0, len(y), 1)
        plt.figure(figsize=(8, 6))  # 设置图片大小
        plt.plot(x, y, "b-")
        plt.title(wifi.split(".")[0])
        ax = plt.gca()  # 获取当前坐标轴内的所有内容
        ax.set_xticks(np.linspace(0, 576, 13))  # 设置x刻度:从0-576,设置12个间隔点
        ax.set_yticks(np.linspace(0, 500, 21))  # 设置刻度
        # plt.axis([40, 160, 0, 500])  # 设置坐标刻度：returns the current axes limits [xmin, xmax, ymin, ymax]
        plt.savefig(out_figure_path+"%s.png" % wifi.split(".")[0])
        plt.close()
        # plt.show()

"""每个区域的每个wifi点的csv文件和图表的生成"""
def get_every_area_csv_and_figure_file(area):
    import os
    _file = "airport_Dataset/Second_round/every_points_WIFI_AP_Records/%s.csv" % area
    csv_file_out = "airport_Dataset/Second_round/wifi_10mins_records_csv/%s/" % area
    input_csv_path = csv_file_out
    figure_file_out = "airport_Dataset/Second_round/wifi_10mins_records_pic/%s/" % area
    if not os.path.exists(_file):
        print "文件不存在！"
    if not os.path.exists(csv_file_out):
        os.mkdir(csv_file_out)  # 若文件目录不存在则新建
    if not os.path.exists(figure_file_out):
        os.mkdir(figure_file_out)
    wifi_10min_tuple = get_wifi_records_10min_tuple(_file)  # 调用函数
    get_everywifi_10min_records_csv(csv_file_out, wifi_10min_tuple)  # 调用函数
    # everywifi_10min_figures(input_csv_path, figure_file_out)  # 调用函数

"""主函数"""
# get_every_area_csv_and_figure_file("E1")
# get_every_area_csv_and_figure_file("E2")
# get_every_area_csv_and_figure_file("E3")
# get_every_area_csv_and_figure_file("EC")
# get_every_area_csv_and_figure_file("W1")
# get_every_area_csv_and_figure_file("W2")
# get_every_area_csv_and_figure_file("W3")
# get_every_area_csv_and_figure_file("WC")
# get_every_area_csv_and_figure_file("T1")



