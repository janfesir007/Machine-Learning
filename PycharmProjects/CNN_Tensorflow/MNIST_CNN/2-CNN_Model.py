# -*-coding:gbk-*-
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mist_data = input_data.read_data_sets("MNIST_data", one_hot=True)
x = tf.placeholder("float", shape=[None, 784])
y = tf.placeholder("float", shape=[None, 10])
learn_rate = tf.placeholder("float")  # 学习率可变化

"""  权重/偏置的初始化：
为了创建这个模型，我们需要创建大量的权重和偏置项。
这个模型中的权重在初始化时应该加入少量的噪声来打破对称性以及避免0梯度---> "截断正态分布"
由于我们使用的是ReLU神经元，因此比较好的做法是用一个较小的正数来初始化偏置项，
以避免神经元节点输出恒为0的问题（dead neurons）。
为了不在建立模型的时候反复做初始化操作，我们定义两个函数用于初始化
"""
def weight_variable(shape):  # 初始化权重 w =[height, width, in_channels, out_channels]=[权重窗口大小的长, 权重窗口大小的宽, 输入通道数, 输出通道数]
    #在卷积层，权重窗口和滑动窗口大小是相同的，因为他们是做点积运算的
    initial_w = tf.truncated_normal(shape=shape, stddev=0.1)  # 由“截断正态分布(truncated normal distribution)”随机产生数值;stddev指“标准差”
    return tf.Variable(initial_w)
def bias_variable(shape):  #初始化偏置项b=0.1常量
    initial_b = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_b)

"""
卷积层 (= 卷积+激励函数) 和 池化层
"""
def conv2d(input_x, input_w):
    """
    卷积层：卷积核窗口设置为n×n（由权重向量给定）,滑动步长strides设置为1×1
    输入x：4维向量[batch, in_height, in_width, in_channels]  #in_channels:通道数
    输入w（filter/kernel tensor）：4维向量[filter_height, filter_width, in_channels, out_channels]
    strides:滑动窗口在每一个维度(N-H-W-C四个维度)方向上的步长（步幅）
    NHWC四个维度分别代表:[batch, in_height, in_width, in_channels]
    Returns:A `Tensor`. Has the same type as `input`.
    不越过边缘取样会得到Valid Padding， 越过边缘取样会得到Same Padding
    """
    return tf.nn.conv2d(input_x, input_w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(value_x):
    """
    #池化层 滑动窗口设置为2×2,滑动步长strides设置为2×2
    value_x: 4维 [batch, height, width, channels]=[数据块，滑动窗口高，滑动窗口的宽，通道数]
    ksize: 输入张量（value_x）的每一维度的窗口大小，中间两维代表滑动窗口大小
    strides:滑动窗口在每一个维度(N-H-W-C四个维度)的步长（步幅）
    Returns: A Tensor,The max pooled output tensor.返回每个滑动窗口的最大值所组成的张量
    """
    return tf.nn.max_pool(value_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

"""
    第一层卷积+池化层
    本例卷积时使用了[5,5,1,32]的卷积核及[1,1,1,1]的步幅。
    总结：
         1.填充方式padding = 'SAME'（边缘填充）:(权重窗口)卷积核[m×m],滑动步幅为[n,n],则卷积/池化后图片大小缩小至原来的1/n
         2.填充方式padding = 'VALID'（边缘不填充）:(权重窗口)卷积核[m×m],滑动步幅为[1,1],图片大小p×p,则卷积/池化后图片大小:p-(m-1)
                                    (权重窗口)卷积核[m×m],滑动步幅为[n,n],图片大小p×p,则卷积/池化后图片大小:以边缘不填充方式计算
    共同对卷积后的张量大小产生影响
"""
w_conv1 = weight_variable([5, 5, 1, 32])  # 卷积层权重张量前两维（权重窗口/patch/kernel/filter）大小 = 卷积层滑动窗口的大小(设置为5×5)
                                          # 1个输入通道（一张图片），手动设置32个输出通道数（32个特征输出）
b_conv1 = bias_variable([32])  # 偏置项个数与输出通道数相同
x_image_conv1 = tf.reshape(x, [-1, 28, 28, 1])  # 原本输入x是2d向量，为了用这一层，我们把x变成一个4d向量，其第2、3维对应图片的宽、高（28×28=784），最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)。
h_conv1_relu1 = tf.nn.relu(conv2d(x_image_conv1, w_conv1) + b_conv1)  # 原本1张输入图片先经过卷积层得到32张28×28×1特征图片，再经过激活函数
h_pool1 = max_pool_2x2(h_conv1_relu1)  # 第一层池化之后得到32张14×14×1的特征图片（因为滑动步长设置为2,故图片大小缩小一倍）

"""第二层卷积层+池化层"""
w_conv2 = weight_variable([5, 5, 32, 64])  # 第一层32个输出通道作为第二层的输入通道，手动设置第二层输出通道64
b_conv2 = bias_variable([64])
x_image_conv2 = h_pool1  # 图片x经过第一层后得到的输出（32张14×14×1的特征图片）作为第二层卷积层的输入
h_conv2_relu2 = tf.nn.relu(conv2d(x_image_conv2, w_conv2) + b_conv2)  # 先经过卷积层得到64张14×14×1的特征图片，再经过激励函数
h_pool2 = max_pool_2x2(h_conv2_relu2)  # 第二层池化后得到64张7×7×1的特征图片

""" fully connected
全连接层：所有图片的所有像素点与所有神经元都有权重连接
    现在，图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，
    用于处理整个图片。我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU
"""
w_fc = weight_variable([64*7*7, 1024])
b_fc = bias_variable([1024])
x_image_fc = tf.reshape(h_pool2, [-1, 64*7*7])
h_fc_relu = tf.nn.relu(tf.matmul(x_image_fc, w_fc) + b_fc)  # matmul():矩阵相乘,得到n×1024矩阵

"""
    Dropout:
    为了减少过拟合，我们在输出层之前加入dropout。我们用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。
    这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。
    TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的scale。
    所以用dropout的时候可以不用考虑scale。
"""
keep_droop_prob = tf.placeholder("float")
h_fc_droop = tf.nn.dropout(h_fc_relu, keep_droop_prob)

"""
    输出层:
    最后，我们添加一个softmax层，就像前面的单层softmax regression一样。
"""
w_soft = weight_variable([1024, 10])
b_soft = bias_variable([10])
x_image_soft = h_fc_droop
y_output = tf.nn.softmax(tf.matmul(x_image_soft, w_soft) + b_soft)  # 得到n×10矩阵

"""评估模型: 交叉熵损失最小化"""
cross_entropy = tf.reduce_sum(-y*tf.log(y_output))
train_step = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cross_entropy)  # 梯度下降优化，准确率99.26;学习率可变化
# train_step = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(cross_entropy)  # 20000次训练，用时约5小时，准确率98.9
correct_pridict = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pridict, "float"))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())  # 所有变量要先定义后初始化！否则会报错：在该操作之后才定义的变量未初始化！
    # train_step = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cross_entropy)  # 报错：某些变量未初始化！
    start_time = time.clock()
    for i in range(1000):
        batch = mist_data.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y: batch[1], learn_rate: 0.1, keep_droop_prob: 0.5})  # 训练
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_droop_prob: 1.0})
            print "训练第%d步，准确率为%f" % (i, train_accuracy)
            # test_accuracy = accuracy.eval(feed_dict={x: mist_data.test.images, y: mist_data.test.labels, keep_droop_prob: 1.0})
            # print "测试第%d步，准确率为%f" % (i, test_accuracy)
    end_time = time.clock()
    test_accuracy = accuracy.eval(feed_dict={x: mist_data.test.images, y: mist_data.test.labels, keep_droop_prob: 1.0})
    print "最终测试准确率为%f" % test_accuracy
    print "用时%f秒" % (end_time-start_time)
