# -*-coding:gbk-*-
"""简单的BP神经网络,单隐层"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mist = input_data.read_data_sets('MNIST_data', one_hot=True)
logs_path = 'tensorflow_logs'
# 输入x,输出y,使用占位符
x = tf.placeholder("float", shape=[None, 784])   # None表示其值大小不确定(图片张数)，784=28×28（每一张图片表示成的one-hot向量）
y = tf.placeholder("float", shape=[None, 10])  # 所有图片总共分成10类（0-9数字）

# 权重w，偏置b，使用变量
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))  # [10]:一维数组（每个元素可以同时进行独立运算）


with tf.name_scope("Model"):  # 预测值y_[None,10]
    y_ = tf.nn.softmax(tf.matmul(x, w)+b)  # tf.matmul()==numpy.doc():内积/点积; 维数：(None*784,784*10)+([10])
with tf.name_scope("Loss"):  # 交叉熵cross_entropy
    cross_entropy = -tf.reduce_sum(y*tf.log(y_))

with tf.name_scope("SGD"):  # 梯度下降最优化
    # train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)   # 梯度下降最优化交叉熵,返回值是一个操作op（利用梯度下降，更新参数w和b，最小化交叉熵）
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
with tf.name_scope("Accuracy"):   #准确率
    correct_pridiction = tf.equal(tf.arg_max(y_, 1), tf.arg_max(y, 1))  # arg_max(y, 1):返回y中维度为1的向量的最大值的索引值
    accuracy = tf.reduce_mean(tf.cast(correct_pridiction, "float"))  # cast():布尔类型转换为数值类型

""" 可视化： 创建一个summary监控tensor """
tf.scalar_summary("1-Loss", cross_entropy)  # Create a summary to monitor cross_entropy tensor
tf.scalar_summary("1-Accuracy", accuracy)
merged_summary_op = tf.merge_all_summaries()  # Merge all summaries into a single op

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())   # 初始化所有变量

    # op to write logs to Tensorboard
    summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

    for i in range(1000):  # 训练模型1000次
        batch = mist.train.next_batch(50)  # 每50张作为一个batch，返回值 batch[图片,标签]

        # 执行/运行：最优化（train_step）, 损失函数（cross_entropy）和summary node（merged_summary_op）
        # “min-batch梯度下降”,在计算图中，可以用feed_dict来替代任何张量，并不仅限于替换占位符。
        _, c, summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                 feed_dict={x: batch[0], y: batch[1]})

        # 将每一迭代产生的loss/accuracy写入事件日志，以便在Tensorboard展示
        summary_writer.add_summary(summary, global_step=i)

        if (i+1) % 100 == 0:  # 每训练100次输入一次模型在预测集上的预测结果
            test_accuracy = accuracy.eval(feed_dict={x: mist.test.images, y: mist.test.labels})
            print test_accuracy
    test_accuracy = accuracy.eval(feed_dict={x: mist.test.images, y: mist.test.labels})  # 1000次后的最终预测结果
    print test_accuracy


