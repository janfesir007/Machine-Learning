# -*-coding:gbk-*-
"""本例以MNIST数据为例
   实现了 ‘最近邻算法’ ：计算预测样本于已知样本的距离，然后取距离最近的那一个最为预测值。
   ‘k-最近临’：计算预测样本于已知样本的距离，先选取k个距离最小的样本，然后统计这k个样本分别属于那一个类别（0～9）
   取统计个数最多的那个类别为最终预测值
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
"""
 TensorFlow 程序使用 tensor 数据结构来代表所有的数据, 计算图中, 操作间传递的数据都是 tensor.
 你可以把 TensorFlow tensor 看作是一个 n 维的数组或列表. 一个 tensor 包含一个静态类型 rank,
 和 一个动态类型shape. """
# 计算图
X_train = tf.placeholder("float", shape=[5000, 784])
Y_train = tf.placeholder("float", shape=[5000, 10])
x_test = tf.placeholder("float", shape=[784])  # 一张一张图片去做预测

# 计算预测点和已知点（训练数据集）的“曼哈顿距离”：例如在平面上，坐标（x1, y1）的i点与坐标（x2, y2）的j点的曼哈顿距离为：d(i,j)=|X1-X2|+|Y1-Y2|.
# distance = tf.reduce_sum(tf.abs(tf.add(X_train, tf.neg(X_test))), reduction_indices=1)  # neg(x):将x取负值
distance = tf.reduce_sum(tf.abs(x_test-X_train), reduction_indices=1)
y_row = tf.arg_min(distance, 0)  # 最近临：取距离最近的那一个
# y_predict = tf.arg_max(Y_train[y_row], 0) # y_row是一个张量，不是一个值，该语句会出现语法错误。在计算图中无法通过某个值进行数据传递。
init = tf.initialize_all_variables()

# 实例化计算图
with tf.Session() as sess:
    sess.run(init)
    X_tr, Y_tr = mnist.train.next_batch(5000)
    X_te, Y_te = mnist.test.next_batch(200)
    i = 0.0
    for (x, y) in zip(X_te, Y_te):
        y_row_index = sess.run(y_row, feed_dict={X_train: X_tr, Y_train: Y_tr, x_test: x})  #see.run()将计算图中的tensor数据类型转变为Python中的narray数据类型
        # y_predict = tf.arg_max(Y_tr[y_row_index], 0).eval()  # Tensorflow中的方法,得出结果的数据类型为：tensor。通过eval()将其转化为Python中的数据类型narray
        y_predict = np.argmax(Y_tr[y_row_index], 0)  # numpy中的方法argmax()和Tensorflow中的arg_max()实现一样的功能
        # y_true = tf.arg_max(y, 0).eval()
        y_true = np.argmax(y, 0)
        if y_predict == y_true:
            i += 1
        print "Pridict:", y_predict, "True Class", y_true
    accuracy = i/len(X_te)
    print "准确率为：%.3f" % accuracy
