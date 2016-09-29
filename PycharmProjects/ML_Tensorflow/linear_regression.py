# -*-coding:gbk-*-
"""线性回归"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 设置超参数
learning_rate = 0.01
training_epochs = 1000
display_step = 100
# 构造一些数据
train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                      7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                      2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
n_samples = train_X.shape[0]

X = tf.placeholder("float")
Y = tf.placeholder("float")
w = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")
# 构建一个线性回归模型
pred = tf.add(tf.mul(X, w), b)
# pred = tf.add(tf.matmul(X, w), b)
# 目标函数：均方误差
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# 随机梯度，最优化目标函数
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  # 此过程计算出w和b值,唯一会对结果产生影响的操作
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)  # AdamOptimizer另一个优化器
"""
总结：输出cost值的两种方法
# print "cost", cost.eval(feed_dict={X: train_X, Y: train_Y})
# print "cost", sess.run(cost, feed_dict={X: train_X, Y: train_Y})

在每个 epoch（周期/训练次数） 送入单个数据点,这被称为"随机梯度下降"（stochastic gradient descent）;
在每个 epoch 送入一堆数据点，这被称为 "mini-batch 梯度下降";
在一个 epoch 一次性送入所有的数据点，这被称为 "batch 梯度下降"(通常所说的“全局梯度下降”)
"""
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for epoch in range(training_epochs):
        # for (x, y) in zip(train_X, train_Y):  # 寻找cost最小值，迭代地对cost求偏导，迭代式地更新参数w和b
        #     sess.run(optimizer, feed_dict={X: x, Y: y})  # 随机梯度下降，最优化cost更新参数w和b
            # x1 = []
            # y1 = []
            # x1.append(x)
            # y1.append(y)
            # xx = np.asarray(x1)
            # yy = np.asarray(y1)
            # n_cost = sess.run(cost, feed_dict={X: xx, Y: yy})
            # print "cost", n_cost, "w=", sess.run(w), "b=", sess.run(b)

        sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})  # 全局梯度下降
        if (epoch + 1) % display_step == 0:  # 输出cost/w/b值
            c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})  # 该操作不会对结果产生任何影响，因为w和b已经由optimizer确定！所以此时cost值也已经确定！
            print "Epoch:", '%04d' % (epoch + 1), "cost=", "{:.7f}".format(c), \
                "w=", sess.run(w), "b=", sess.run(b)
    print "优化完成!"
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})  # cost由w和b决定
    print "Training cost=", training_cost, "W=", sess.run(w), "b=", sess.run(b), '\n'
    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(w) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
