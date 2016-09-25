# -*-coding:gbk-*-
"""简单的BP神经网络,单隐层"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 输入x,输出y,使用占位符
x = tf.placeholder("float", shape=[None, 784])   # None表示其值大小不确定(图片张数)，784=28×28（每一张图片表示成的one-hot向量）
y = tf.placeholder("float", shape=[None, 10])  # 所有图片总共分成10类（0-9数字）

# 权重w，偏置b，使用变量
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))  # [10]:一维数组（每个元素可以同时进行独立运算）

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())   # 初始化所有变量

    # 预测值y[None,10],交叉熵cross_entropy
    y_ = tf.nn.softmax(tf.matmul(x, w)+b)  # tf.matmul()==numpy.doc():内积/点积; 维数：(None*784,784*10)+([10])
    cross_entropy = -tf.reduce_sum(y*tf.log(y_))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)   # 随机梯度最优化交叉熵,返回值是一个操作op（利用梯度下降，更新参数，最小化交叉熵）
    for i in range(1000):
        batch = mist.train.next_batch(50)  # 每50张作为一个batch，返回值 batch[图片,标签]
        correct_pridiction = tf.equal(tf.arg_max(y_, 1), tf.arg_max(y, 1))  # arg_max(y, 1):返回y中维度为1的向量的最大值的索引值
        accuracy = tf.reduce_mean(tf.cast(correct_pridiction, "float"))  # cast():布尔类型转换为数值类型

        train_step.run(feed_dict={x: batch[0], y: batch[1]})  # 在计算图中，可以用feed_dict来替代任何张量，并不仅限于替换占位符。

        test_accuracy = accuracy.eval(feed_dict={x: mist.test.images, y: mist.test.labels})
        if i % 100 == 0:
            # print accuracy.eval(feed_dict={x: mist.train.images, y: mist.train.labels})
            print test_accuracy




