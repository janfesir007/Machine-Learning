# -*-coding:gbk-*-
"""�򵥵�BP������,������"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mist = input_data.read_data_sets('MNIST_data', one_hot=True)
logs_path = 'tensorflow_logs'
# ����x,���y,ʹ��ռλ��
x = tf.placeholder("float", shape=[None, 784])   # None��ʾ��ֵ��С��ȷ��(ͼƬ����)��784=28��28��ÿһ��ͼƬ��ʾ�ɵ�one-hot������
y = tf.placeholder("float", shape=[None, 10])  # ����ͼƬ�ܹ��ֳ�10�ࣨ0-9���֣�

# Ȩ��w��ƫ��b��ʹ�ñ���
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))  # [10]:һά���飨ÿ��Ԫ�ؿ���ͬʱ���ж������㣩


with tf.name_scope("Model"):  # Ԥ��ֵy_[None,10]
    y_ = tf.nn.softmax(tf.matmul(x, w)+b)  # tf.matmul()==numpy.doc():�ڻ�/���; ά����(None*784,784*10)+([10])
with tf.name_scope("Loss"):  # ������cross_entropy
    cross_entropy = -tf.reduce_sum(y*tf.log(y_))

with tf.name_scope("SGD"):  # �ݶ��½����Ż�
    # train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)   # �ݶ��½����Ż�������,����ֵ��һ������op�������ݶ��½������²���w��b����С�������أ�
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
with tf.name_scope("Accuracy"):   #׼ȷ��
    correct_pridiction = tf.equal(tf.arg_max(y_, 1), tf.arg_max(y, 1))  # arg_max(y, 1):����y��ά��Ϊ1�����������ֵ������ֵ
    accuracy = tf.reduce_mean(tf.cast(correct_pridiction, "float"))  # cast():��������ת��Ϊ��ֵ����

""" ���ӻ��� ����һ��summary���tensor """
tf.scalar_summary("1-Loss", cross_entropy)  # Create a summary to monitor cross_entropy tensor
tf.scalar_summary("1-Accuracy", accuracy)
merged_summary_op = tf.merge_all_summaries()  # Merge all summaries into a single op

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())   # ��ʼ�����б���

    # op to write logs to Tensorboard
    summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

    for i in range(1000):  # ѵ��ģ��1000��
        batch = mist.train.next_batch(50)  # ÿ50����Ϊһ��batch������ֵ batch[ͼƬ,��ǩ]

        # ִ��/���У����Ż���train_step��, ��ʧ������cross_entropy����summary node��merged_summary_op��
        # ��min-batch�ݶ��½���,�ڼ���ͼ�У�������feed_dict������κ������������������滻ռλ����
        _, c, summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                 feed_dict={x: batch[0], y: batch[1]})

        # ��ÿһ����������loss/accuracyд���¼���־���Ա���Tensorboardչʾ
        summary_writer.add_summary(summary, global_step=i)

        if (i+1) % 100 == 0:  # ÿѵ��100������һ��ģ����Ԥ�⼯�ϵ�Ԥ����
            test_accuracy = accuracy.eval(feed_dict={x: mist.test.images, y: mist.test.labels})
            print test_accuracy
    test_accuracy = accuracy.eval(feed_dict={x: mist.test.images, y: mist.test.labels})  # 1000�κ������Ԥ����
    print test_accuracy


