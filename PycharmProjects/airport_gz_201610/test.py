# -*-encoding:gbk-*-
# from pylab import *
# from matplotlib.ticker import MultipleLocator, FormatStrFormatter
#
# xmajorLocator = MultipleLocator(20)  # 将x主刻度标签设置为20的倍数
# xmajorFormatter = FormatStrFormatter('%1.1f')  # 设置x轴标签文本的格式
# xminorLocator = MultipleLocator(5)  # 将x轴次刻度标签设置为5的倍数
#
# ymajorLocator = MultipleLocator(0.5)  # 将y轴主刻度标签设置为0.5的倍数
# ymajorFormatter = FormatStrFormatter('%1.1f')  # 设置y轴标签文本的格式
# yminorLocator = MultipleLocator(0.1)  # 将此y轴次刻度标签设置为0.1的倍数
#
# t = arange(0.0, 100.0, 1)
# s = sin(0.1 * pi * t) * exp(-t * 0.01)
#
# ax = subplot(111)  # 注意:一般都在ax中设置,不再plot中设置
# plot(t, s, '--b*')
#
# # 设置主刻度标签的位置,标签文本的格式
# ax.xaxis.set_major_locator(xmajorLocator)
# ax.xaxis.set_major_formatter(xmajorFormatter)
#
# ax.yaxis.set_major_locator(ymajorLocator)
# ax.yaxis.set_major_formatter(ymajorFormatter)
#
# # 显示次刻度标签的位置,没有标签文本
# ax.xaxis.set_minor_locator(xminorLocator)
# ax.yaxis.set_minor_locator(yminorLocator)
#
# ax.xaxis.grid(True, which='major')  # x坐标轴的网格使用主刻度
# ax.yaxis.grid(True, which='minor')  # y坐标轴的网格使用次刻度
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


"""保存图片"""
import numpy as np

