# -*-encoding:gbk-*-
import time
from operator import itemgetter  # �����������
"""
ԭʼ������ϴ��������ԭʼ��������749��wifi��(4133492����¼),
����Ч�۲ⷶΧ��ֻ��746��(4116917����¼)
"""
# with open("airport_Dataset/Second_round/WIFI_AP_Passenger_Records_chusai_2ndround.csv", "r") as f:
#     context = f.readlines()
#     with open("airport_Dataset/Second_round/WIFI_AP_Passenger_Records_chusai_2ndround_fitDataset.csv", "w") as wf:
#         i = 0
#         for line in context:
#             line = line.replace("\n", "")
#             array_list = line.split(",")
#             if array_list[0] == "WIFIAPTag":  # ȥ����һ�з�������
#                 wf.write(line + "\n")
#                 continue
#             # ȥ�����ڹ۲ⷶΧ�����wifi��
#             if array_list[0][:2] in ["E1", "E2", "E3", "EC", "T1", "W1", "W2", "W3", "WC"]:
#                 wf.write(line+"\n")

"""��ÿ�������wifi������ݶ��������,��9������"""
E1=[]  # E1�ǻ��ڵ�wifi���ݼ�¼
E2=[]
E3=[]
W1=[]
W2=[]
W3=[]
EC=[]
WC=[]
T1=[]
time1 = time.time()
with open("airport_Dataset/Second_round/WIFI_AP_Passenger_Records_chusai_2ndround_fitDataset.csv", "r") as f:
    data_lines = f.readlines()  # ���ļ������б�list,�б����ÿ��Ԫ����str����
    for line in data_lines[8000000:]:  # �ļ�̫��,�����α���[:800��]��[800��:]
        lines = line.replace("\n", "")
        arraydata = lines.split(",")
        if arraydata[0][:2] == "E1":
            E1.append(arraydata)
        elif arraydata[0][:2] == "E2":
            E2.append(arraydata)
        elif arraydata[0][:2] == "E3":
            E3.append(arraydata)
        elif arraydata[0][:2] == "W1":
            W1.append(arraydata)
        elif arraydata[0][:2] == "W2":
            W2.append(arraydata)
        elif arraydata[0][:2] == "W3":
            W3.append(arraydata)
        elif arraydata[0][:2] == "EC":
            EC.append(arraydata)
        elif arraydata[0][:2] == "WC":
            WC.append(arraydata)
        elif arraydata[0][:2] == "T1":
            T1.append(arraydata)

"""��ÿ�������wifi���ݼ�¼,��wifi����,����,����������˳������ú�д���ļ�"""
with open("airport_Dataset/Second_round/every_points_WIFI_AP_Records/E1.csv", "a") as fE1:
    E1=sorted(E1, key=itemgetter(0, 2))  # ���򣺰��б�Ԫ���е��±�ֵ0��wifi���ƣ�,2(����),1(��������)��˳���������
    for line in E1:
        arraydataE1=",".join(line)  # join(): ��split()"�ָ�"���������������
        fE1.write(arraydataE1+"\n")
with open("airport_Dataset/Second_round/every_points_WIFI_AP_Records/E2.csv", "a") as fE2:
    E2 =sorted(E2, key=itemgetter(0, 2))
    for line in E2:
        arraydataE2=",".join(line)
        fE2.writelines(arraydataE2+"\n")
with open("airport_Dataset/Second_round/every_points_WIFI_AP_Records/E3.csv", "a") as fE3:
    E3 =sorted(E3, key=itemgetter(0, 2))
    for line in E3:
        arraydataE3=",".join(line)
        fE3.writelines(arraydataE3+"\n")
with open("airport_Dataset/Second_round/every_points_WIFI_AP_Records/W1.csv", "a") as fW1:
    W1=sorted(W1, key=itemgetter(0, 2))
    for line in W1:
        arraydataW1=",".join(line)
        fW1.writelines(arraydataW1+"\n")
with open("airport_Dataset/Second_round/every_points_WIFI_AP_Records/W2.csv", "a") as fW2:
    W2=sorted(W2, key=itemgetter(0, 2))
    for line in W2:
        arraydataW2=",".join(line)
        fW2.writelines(arraydataW2+"\n")
with open("airport_Dataset/Second_round/every_points_WIFI_AP_Records/W3.csv", "a") as fW3:
    W3 =sorted(W3, key=itemgetter(0, 2))
    for line in W3:
        arraydataW3=",".join(line)
        fW3.writelines(arraydataW3+"\n")
with open("airport_Dataset/Second_round/every_points_WIFI_AP_Records/EC.csv", "a") as fEC:
    EC=sorted(EC, key=itemgetter(0, 2))
    for line in EC:
        arraydataEC=",".join(line)
        fEC.writelines(arraydataEC+"\n")
with open("airport_Dataset/Second_round/every_points_WIFI_AP_Records/WC.csv", "a") as fWC:
    WC=sorted(WC, key=itemgetter(0, 2))
    for line in WC:
        arraydataWC=",".join(line)
        fWC.writelines(arraydataWC+"\n")
with open("airport_Dataset/Second_round/every_points_WIFI_AP_Records/T1.csv", "a") as fT1:
    T1=sorted(T1, key=itemgetter(0, 2))
    for line in T1:
        arraydataT1=",".join(line)
        fT1.writelines(arraydataT1+"\n")
time2 = time.time()
print "��ʱ%f"%(time2-time1)




