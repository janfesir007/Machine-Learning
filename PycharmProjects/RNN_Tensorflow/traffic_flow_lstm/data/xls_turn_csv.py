# -*-coding:gbk-*-
# -*-coding:utf-8-*-
# -*- coding: utf-8 -*-
"""Ubuntu�Ķ��csv�ļ��ϲ���һ����Ϊfull��csv�ļ���� cat *.csv > full.csv"""
"""��xls�ļ���ʽת��Ϊcsv��ʽ"""
import pandas as pd
data_xls = pd.read_excel('pems1-3Mon.xls')
data_xls.to_csv('flow1-3Mon.csv', encoding='utf-8')