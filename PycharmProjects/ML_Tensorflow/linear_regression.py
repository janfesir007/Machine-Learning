# -*-coding:gbk-*-
"""���Իع�"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ���ó�����
learning_rate = 0.01
training_epochs = 1000
display_step = 100
# ����һЩ����
train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                      7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                      2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
n_samples = train_X.shape[0]

X = tf.placeholder("float")
Y = tf.placeholder("float")
w = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")
# ����һ�����Իع�ģ��
pred = tf.add(tf.mul(X, w), b)
# pred = tf.add(tf.matmul(X, w), b)
# Ŀ�꺯�����������
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# ����ݶȣ����Ż�Ŀ�꺯��
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  # �˹��̼����w��bֵ,Ψһ��Խ������Ӱ��Ĳ���
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)  # AdamOptimizer��һ���Ż���
"""
�ܽ᣺���costֵ�����ַ���
# print "cost", cost.eval(feed_dict={X: train_X, Y: train_Y})
# print "cost", sess.run(cost, feed_dict={X: train_X, Y: train_Y})

��ÿ�� epoch������/ѵ�������� ���뵥�����ݵ�,�ⱻ��Ϊ"����ݶ��½�"��stochastic gradient descent��;
��ÿ�� epoch ����һ�����ݵ㣬�ⱻ��Ϊ "mini-batch �ݶ��½�";
��һ�� epoch һ�����������е����ݵ㣬�ⱻ��Ϊ "batch �ݶ��½�"(ͨ����˵�ġ�ȫ���ݶ��½���)
"""
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for epoch in range(training_epochs):
        # for (x, y) in zip(train_X, train_Y):  # Ѱ��cost��Сֵ�������ض�cost��ƫ��������ʽ�ظ��²���w��b
        #     sess.run(optimizer, feed_dict={X: x, Y: y})  # ����ݶ��½������Ż�cost���²���w��b
            # x1 = []
            # y1 = []
            # x1.append(x)
            # y1.append(y)
            # xx = np.asarray(x1)
            # yy = np.asarray(y1)
            # n_cost = sess.run(cost, feed_dict={X: xx, Y: yy})
            # print "cost", n_cost, "w=", sess.run(w), "b=", sess.run(b)

        sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})  # ȫ���ݶ��½�
        if (epoch + 1) % display_step == 0:  # ���cost/w/bֵ
            c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})  # �ò�������Խ�������κ�Ӱ�죬��Ϊw��b�Ѿ���optimizerȷ�������Դ�ʱcostֵҲ�Ѿ�ȷ����
            print "Epoch:", '%04d' % (epoch + 1), "cost=", "{:.7f}".format(c), \
                "w=", sess.run(w), "b=", sess.run(b)
    print "�Ż����!"
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})  # cost��w��b����
    print "Training cost=", training_cost, "W=", sess.run(w), "b=", sess.run(b), '\n'
    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(w) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
