# -*-coding:gbk-*-
"""�򵥵�BP������,������"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mist = input_data.read_data_sets('MNIST_data', one_hot=True)

# ����x,���y,ʹ��ռλ��
x = tf.placeholder("float", shape=[None, 784])   # None��ʾ��ֵ��С��ȷ��(ͼƬ����)��784=28��28��ÿһ��ͼƬ��ʾ�ɵ�one-hot������
y = tf.placeholder("float", shape=[None, 10])  # ����ͼƬ�ܹ��ֳ�10�ࣨ0-9���֣�

# Ȩ��w��ƫ��b��ʹ�ñ���
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))  # [10]:һά���飨ÿ��Ԫ�ؿ���ͬʱ���ж������㣩

# Ԥ��ֵy[None,10],������cross_entropy
y_ = tf.nn.softmax(tf.matmul(x, w)+b)  # tf.matmul()==numpy.doc():�ڻ�/���; ά����(None*784,784*10)+([10])
cross_entropy = -tf.reduce_sum(y*tf.log(y_))

# train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)   # �ݶ��½����Ż�������,����ֵ��һ������op�������ݶ��½������²���w��b����С�������أ�
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
correct_pridiction = tf.equal(tf.arg_max(y_, 1), tf.arg_max(y, 1))  # arg_max(y, 1):����y��ά��Ϊ1�����������ֵ������ֵ
accuracy = tf.reduce_mean(tf.cast(correct_pridiction, "float"))  # cast():��������ת��Ϊ��ֵ����

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())   # ��ʼ�����б���
    for i in range(1000):  # ѵ��ģ��1000��
        batch = mist.train.next_batch(50)  # ÿ50����Ϊһ��batch������ֵ batch[ͼƬ,��ǩ]
        train_step.run(feed_dict={x: batch[0], y: batch[1]})  # ��min-batch�ݶ��½���,�ڼ���ͼ�У�������feed_dict������κ������������������滻ռλ����

        if (i+1) % 100 == 0:  # ÿѵ��100������һ��ģ����Ԥ�⼯�ϵ�Ԥ����
            test_accuracy = accuracy.eval(feed_dict={x: mist.test.images, y: mist.test.labels})
            print test_accuracy
    test_accuracy = accuracy.eval(feed_dict={x: mist.test.images, y: mist.test.labels})  # 1000�κ������Ԥ����
    print test_accuracy





