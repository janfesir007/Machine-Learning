# -*-coding:gbk-*-
"""������MNIST����Ϊ��
   ʵ���� ��������㷨�� ������Ԥ����������֪�����ľ��룬Ȼ��ȡ�����������һ����ΪԤ��ֵ��
   ��k-����١�������Ԥ����������֪�����ľ��룬��ѡȡk��������С��������Ȼ��ͳ����k�������ֱ�������һ�����0��9��
   ȡͳ�Ƹ��������Ǹ����Ϊ����Ԥ��ֵ
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
"""
 TensorFlow ����ʹ�� tensor ���ݽṹ���������е�����, ����ͼ��, �����䴫�ݵ����ݶ��� tensor.
 ����԰� TensorFlow tensor ������һ�� n ά��������б�. һ�� tensor ����һ����̬���� rank,
 �� һ����̬����shape. """
# ����ͼ
X_train = tf.placeholder("float", shape=[5000, 784])
Y_train = tf.placeholder("float", shape=[5000, 10])
x_test = tf.placeholder("float", shape=[784])  # һ��һ��ͼƬȥ��Ԥ��

# ����Ԥ������֪�㣨ѵ�����ݼ����ġ������پ��롱��������ƽ���ϣ����꣨x1, y1����i�������꣨x2, y2����j��������پ���Ϊ��d(i,j)=|X1-X2|+|Y1-Y2|.
# distance = tf.reduce_sum(tf.abs(tf.add(X_train, tf.neg(X_test))), reduction_indices=1)  # neg(x):��xȡ��ֵ
distance = tf.reduce_sum(tf.abs(x_test-X_train), reduction_indices=1)
y_row = tf.arg_min(distance, 0)  # ����٣�ȡ�����������һ��
# y_predict = tf.arg_max(Y_train[y_row], 0) # y_row��һ������������һ��ֵ������������﷨�����ڼ���ͼ���޷�ͨ��ĳ��ֵ�������ݴ��ݡ�
init = tf.initialize_all_variables()

# ʵ��������ͼ
with tf.Session() as sess:
    sess.run(init)
    X_tr, Y_tr = mnist.train.next_batch(5000)
    X_te, Y_te = mnist.test.next_batch(200)
    i = 0.0
    for (x, y) in zip(X_te, Y_te):
        y_row_index = sess.run(y_row, feed_dict={X_train: X_tr, Y_train: Y_tr, x_test: x})  #see.run()������ͼ�е�tensor��������ת��ΪPython�е�narray��������
        # y_predict = tf.arg_max(Y_tr[y_row_index], 0).eval()  # Tensorflow�еķ���,�ó��������������Ϊ��tensor��ͨ��eval()����ת��ΪPython�е���������narray
        y_predict = np.argmax(Y_tr[y_row_index], 0)  # numpy�еķ���argmax()��Tensorflow�е�arg_max()ʵ��һ���Ĺ���
        # y_true = tf.arg_max(y, 0).eval()
        y_true = np.argmax(y, 0)
        if y_predict == y_true:
            i += 1
        print "Pridict:", y_predict, "True Class", y_true
    accuracy = i/len(X_te)
    print "׼ȷ��Ϊ��%.3f" % accuracy
