# -*-encoding:gbk-*-
# from pylab import *
# from matplotlib.ticker import MultipleLocator, FormatStrFormatter
#
# xmajorLocator = MultipleLocator(20)  # ��x���̶ȱ�ǩ����Ϊ20�ı���
# xmajorFormatter = FormatStrFormatter('%1.1f')  # ����x���ǩ�ı��ĸ�ʽ
# xminorLocator = MultipleLocator(5)  # ��x��ο̶ȱ�ǩ����Ϊ5�ı���
#
# ymajorLocator = MultipleLocator(0.5)  # ��y�����̶ȱ�ǩ����Ϊ0.5�ı���
# ymajorFormatter = FormatStrFormatter('%1.1f')  # ����y���ǩ�ı��ĸ�ʽ
# yminorLocator = MultipleLocator(0.1)  # ����y��ο̶ȱ�ǩ����Ϊ0.1�ı���
#
# t = arange(0.0, 100.0, 1)
# s = sin(0.1 * pi * t) * exp(-t * 0.01)
#
# ax = subplot(111)  # ע��:һ�㶼��ax������,����plot������
# plot(t, s, '--b*')
#
# # �������̶ȱ�ǩ��λ��,��ǩ�ı��ĸ�ʽ
# ax.xaxis.set_major_locator(xmajorLocator)
# ax.xaxis.set_major_formatter(xmajorFormatter)
#
# ax.yaxis.set_major_locator(ymajorLocator)
# ax.yaxis.set_major_formatter(ymajorFormatter)
#
# # ��ʾ�ο̶ȱ�ǩ��λ��,û�б�ǩ�ı�
# ax.xaxis.set_minor_locator(xminorLocator)
# ax.yaxis.set_minor_locator(yminorLocator)
#
# ax.xaxis.grid(True, which='major')  # x�����������ʹ�����̶�
# ax.yaxis.grid(True, which='minor')  # y�����������ʹ�ôο̶�
#
# show()


# import numpy as np
# import matplotlib.pyplot as plt
#
# fig, ax = plt.subplots()
#
# x = [1, 2, 3, 4, 5]
# y = [0, 2, 5, 9, 15]
#
# # ax is the axes instance
# group_labels = ['a', 'b', 'c', 'd']
#
# plt.plot(x, y)
# plt.xticks(x, group_labels, rotation=0)
# plt.grid()
# plt.show()


"""����ͼƬ"""
import numpy as np

