# -*-encoding:gbk-*-
from operator import itemgetter  # �����������
import numpy as np
"""�����ļ���"""
def list_files(dir_path):
    import os
    list_file = os.listdir(dir_path)
    list_file.sort()
    return list_file


"""
�ù�ȥ���죨11-13��/22-24�գ�ͬһʱ��ε���������ƽ��ֵ��Ϊ14��/25�յ�Ԥ��ֵ
"""
def get_pridict_through_avg():
    import time
    from operator import itemgetter
    dir_file = "airport_Dataset/every_points_WIFI_AP_Records/"
    list_file = list_files(dir_file)  # ���ú���,�����ļ���������ļ�
    list_file.sort()
    for file_name in list_file:
        wifi_records_10min = {}  # {(wifi����,10minʱ���):�������� }:��������Ԥ��Ľ��Y,���ǻ���Ҫ�ҵ�Ӱ��ý�����������X(����)
        with open(dir_file + file_name, "r") as f:
            datalines = f.readlines()
            for line in datalines:
                arraylines = line.replace("\n", "")
                arraydata = arraylines.split(",")

                wifi_name = arraydata[0]
                connect_count = int(arraydata[1])
                date = arraydata[2]
                date_10min = date[:15]  # �磺��2016-9-10-18-50��59ͳͳ��Ϊ2016-9-10-18-5����ʾΪ��10���ӵ�ʱ���
                stru_date_10min = time.strptime(date_10min, "%Y-%m-%d-%H-%M")  # ת��Ϊ�ɱȽϵ�ʱ��
                records_id = (wifi_name, date_10min)
                if not records_id in wifi_records_10min:
                    # if time.strptime("2016-9-11-15-0", "%Y-%m-%d-%H-%M")<stru_date_10min <time.strptime("2016-9-11-17-5", "%Y-%m-%d-%H-%M"):
                    #     wifi_records_10min_E1[records_id] = connect_count  # 2016-9-10��һ��û��¼
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
        avg_wifi_records_10min_tuple = sorted(avg_wifi_records_10min.iteritems(), key=itemgetter(0))  # ��������,����Ԫ��

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
        print file_name + "�ļ����Ԥ�⣡\n"
    print "ȫ����ɣ�"

"""
�ù�ȥ���죨11-13��/22-24�գ�ͬһʱ��ε��������Ĵ���ͬȨ�ص�����ֵ(11�յ���������0.2+12�յġ�0.3+13�յġ�0.5)��Ϊ14�յ�Ԥ��ֵ
"""
def get_pridict_through_weight_avg():
    import time
    from operator import itemgetter
    dir_file = "airport_Dataset/Second_round/every_points_WIFI_AP_Records/"
    list_file = list_files(dir_file)  # ���ú���,�����ļ���������ļ�
    list_file.sort()
    for file_name in list_file:
        weight_wifi_records_10min = {}  # 3��15��00-17��59{(wifi����,10minʱ���):��������(��Ȩ�ع�һ����) }
        with open(dir_file + file_name, "r") as f:
            datalines = f.readlines()
            for line in datalines:
                arraylines = line.replace("\n", "")
                arraydata = arraylines.split(",")

                wifi_name = arraydata[0]
                connect_count = int(arraydata[1])
                date = arraydata[2]
                date_10min = date[:15]  # �磺��2016-9-10-18-50��59ͳͳ��Ϊ2016-9-10-18-5����ʾΪ��10���ӵ�ʱ���
                stru_date_10min = time.strptime(date_10min, "%Y-%m-%d-%H-%M")  # ת��Ϊ�ɱȽϵ�ʱ��
                records_id = (wifi_name, date_10min)

                # if time.strptime("2016-9-11-15-0", "%Y-%m-%d-%H-%M")<stru_date_10min <time.strptime("2016-9-11-17-5", "%Y-%m-%d-%H-%M"):
                #     wifi_records_10min_E1[records_id] = connect_count  # 2016-9-10��һ��û��¼
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
        avg_wifi_records_10min_tuple = sorted(weight_avg_wifi_records_10min.iteritems(), key=itemgetter(0))  # ��������,����Ԫ��

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
        print file_name + "�ļ����Ԥ�⣡\n"
    print "ȫ����ɣ�"



"""
ϸ����������
"""

"""��ȡĳ������ÿ��wifi��10���Ӽ���ڵ�������,������wifi+���ڡ��ź���,��Ԫ����ʽ�洢"""
def get_wifi_records_10min_tuple(file_path):
    wifi_records_10min = {}  # {(wifi����,10minʱ���):�������� }:��������Ԥ��Ľ��Y,���ǻ���Ҫ�ҵ�Ӱ��ý�����������X(����)
    with open(file_path, "r") as f:
        datalines = f.readlines()
        for line in datalines:
            arraylines = line.replace("\n", "")
            arraydata = arraylines.split(",")

            wifi_name = arraydata[0]
            connect_count = int(arraydata[1])
            date = arraydata[2]
            date_10min = date[:15]+"0"  # �磺��2016-9-10-18-50��59ͳͳ��Ϊ2016-9-10-18-50����ʾΪ��10���ӵ�ʱ���
            records_id = (wifi_name, date_10min)
            if not records_id in wifi_records_10min:
                wifi_records_10min[records_id] = connect_count
            else:
                wifi_records_10min[records_id] += connect_count
    wifi_records_10min_tuple = sorted(wifi_records_10min.iteritems(), key=itemgetter(0))  # ����(wifi_name,date)����,�ֵ���������Ԫ�����ͷ���
    return wifi_records_10min_tuple

"""ĳ������������wifi����ʱ��仯(3��ʱ����ÿ��10����)��������,��.csv��ʽ�洢"""
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

"""���ӻ���ĳ������������wifi����ʱ��仯(3��ʱ����ÿ��10����)��������,��ͼ����ʽ�洢"""
def everywifi_10min_figures(input_csv_path, out_figure_path):
    # ��һ�ֻ�ͼ������ʹ��matplotlib.pylab��ͼ��
    # import matplotlib.pylab as pltl
    # from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    #
    # xmajorLocator = MultipleLocator(20)  # ��x���̶ȱ�ǩ����Ϊ20�ı���
    # xmajorFormatter = FormatStrFormatter('%1.0f')  # ����x���ǩ�ı��ĸ�ʽ
    # xminorLocator = MultipleLocator(5)  # ��x��ο̶ȱ�ǩ����Ϊ5�ı���
    #
    # ymajorLocator = MultipleLocator(20)  # ��y�����̶ȱ�ǩ����Ϊ0.5�ı���
    # ymajorFormatter = FormatStrFormatter('%1.1f')  # ����y���ǩ�ı��ĸ�ʽ
    # yminorLocator = MultipleLocator(5)  # ����y��ο̶ȱ�ǩ����Ϊ0.1�ı���
    #
    # ax = pltl.subplot(111)  # ע��:һ�㶼��ax������,����plot������
    #
    # # �������̶ȱ�ǩ��λ��,��ǩ�ı��ĸ�ʽ
    # ax.xaxis.set_major_locator(xmajorLocator)
    # ax.xaxis.set_major_formatter(xmajorFormatter)
    # ax.yaxis.set_major_locator(ymajorLocator)
    # ax.yaxis.set_major_formatter(ymajorFormatter)
    #
    # # ��ʾ�ο̶ȱ�ǩ��λ��,û�б�ǩ�ı�
    # ax.xaxis.set_minor_locator(xminorLocator)
    # ax.yaxis.set_minor_locator(yminorLocator)
    #
    # ax.xaxis.grid(True, which='major')  # x�����������ʹ�����̶�
    # ax.yaxis.grid(True, which='minor')  # y�����������ʹ�ôο̶�

    # �ڶ��ֻ�ͼ������ʹ��matplotlib.pyplot��ͼ��
    import matplotlib.pyplot as plt
    list_file = list_files(input_csv_path)
    for wifi in list_file:  # �����ļ����µ������ļ�,��.csv�ļ�����ͼ���ļ�
        with open(input_csv_path + wifi, "r") as f:
            context = f.readlines()
            y = []
            i = 0
            for line in context:
                i += 1
                record = line.replace("\n", "")
                wifi_record = record.split(",")
                if i > 31:  # ȥ��9.10�յļ�¼,��9.11��0�㿪ʼ��¼
                    y.append(wifi_record[2])
        x = np.arange(0, len(y), 1)
        plt.figure(figsize=(8, 6))  # ����ͼƬ��С
        plt.plot(x, y, "b-")
        plt.title(wifi.split(".")[0])
        ax = plt.gca()  # ��ȡ��ǰ�������ڵ���������
        ax.set_xticks(np.linspace(0, 576, 13))  # ����x�̶�:��0-576,����12�������
        ax.set_yticks(np.linspace(0, 500, 21))  # ���ÿ̶�
        # plt.axis([40, 160, 0, 500])  # ��������̶ȣ�returns the current axes limits [xmin, xmax, ymin, ymax]
        plt.savefig(out_figure_path+"%s.png" % wifi.split(".")[0])
        plt.close()
        # plt.show()

"""ÿ�������ÿ��wifi���csv�ļ���ͼ�������"""
def get_every_area_csv_and_figure_file(area):
    import os
    _file = "airport_Dataset/Second_round/every_points_WIFI_AP_Records/%s.csv" % area
    csv_file_out = "airport_Dataset/Second_round/wifi_10mins_records_csv/%s/" % area
    input_csv_path = csv_file_out
    figure_file_out = "airport_Dataset/Second_round/wifi_10mins_records_pic/%s/" % area
    if not os.path.exists(_file):
        print "�ļ������ڣ�"
    if not os.path.exists(csv_file_out):
        os.mkdir(csv_file_out)  # ���ļ�Ŀ¼���������½�
    if not os.path.exists(figure_file_out):
        os.mkdir(figure_file_out)
    wifi_10min_tuple = get_wifi_records_10min_tuple(_file)  # ���ú���
    get_everywifi_10min_records_csv(csv_file_out, wifi_10min_tuple)  # ���ú���
    # everywifi_10min_figures(input_csv_path, figure_file_out)  # ���ú���

"""������"""
# get_every_area_csv_and_figure_file("E1")
# get_every_area_csv_and_figure_file("E2")
# get_every_area_csv_and_figure_file("E3")
# get_every_area_csv_and_figure_file("EC")
# get_every_area_csv_and_figure_file("W1")
# get_every_area_csv_and_figure_file("W2")
# get_every_area_csv_and_figure_file("W3")
# get_every_area_csv_and_figure_file("WC")
# get_every_area_csv_and_figure_file("T1")



