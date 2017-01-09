# -*-coding:gbk-*-
# -*-coding:utf-8-*-
# -*- coding: utf-8 -*-
"""Ubuntu的多个csv文件合并成一个名为full的csv文件命令： cat *.csv > full.csv"""
"""将xls文件格式转化为csv格式"""
import pandas as pd
data_xls = pd.read_excel('pems1-3Mon.xls')
data_xls.to_csv('flow1-3Mon.csv', encoding='utf-8')