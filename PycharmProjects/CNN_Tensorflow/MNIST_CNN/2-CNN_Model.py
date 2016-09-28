# -*-coding:gbk-*-
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mist_data = input_data.read_data_sets("MNIST_data", one_hot=True)
x = tf.placeholder("float", shape=[None, 784])
y = tf.placeholder("float", shape=[None, 10])
learn_rate = tf.placeholder("float")  # ѧϰ�ʿɱ仯

"""  Ȩ��/ƫ�õĳ�ʼ����
Ϊ�˴������ģ�ͣ�������Ҫ����������Ȩ�غ�ƫ���
���ģ���е�Ȩ���ڳ�ʼ��ʱӦ�ü������������������ƶԳ����Լ�����0�ݶ�---> "�ض���̬�ֲ�"
��������ʹ�õ���ReLU��Ԫ����˱ȽϺõ���������һ����С����������ʼ��ƫ���
�Ա�����Ԫ�ڵ������Ϊ0�����⣨dead neurons����
Ϊ�˲��ڽ���ģ�͵�ʱ�򷴸�����ʼ�����������Ƕ��������������ڳ�ʼ��
"""
def weight_variable(shape):  # ��ʼ��Ȩ�� w =[height, width, in_channels, out_channels]=[Ȩ�ش��ڴ�С�ĳ�, Ȩ�ش��ڴ�С�Ŀ�, ����ͨ����, ���ͨ����]
    #�ھ���㣬Ȩ�ش��ںͻ������ڴ�С����ͬ�ģ���Ϊ����������������
    initial_w = tf.truncated_normal(shape=shape, stddev=0.1)  # �ɡ��ض���̬�ֲ�(truncated normal distribution)�����������ֵ;stddevָ����׼�
    return tf.Variable(initial_w)
def bias_variable(shape):  #��ʼ��ƫ����b=0.1����
    initial_b = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_b)

"""
����� (= ���+��������) �� �ػ���
"""
def conv2d(input_x, input_w):
    """
    ����㣺����˴�������Ϊn��n����Ȩ������������,��������strides����Ϊ1��1
    ����x��4ά����[batch, in_height, in_width, in_channels]  #in_channels:ͨ����
    ����w��filter/kernel tensor����4ά����[filter_height, filter_width, in_channels, out_channels]
    strides:����������ÿһ��ά��(N-H-W-C�ĸ�ά��)�����ϵĲ�����������
    NHWC�ĸ�ά�ȷֱ����:[batch, in_height, in_width, in_channels]
    Returns:A `Tensor`. Has the same type as `input`.
    ��Խ����Եȡ����õ�Valid Padding�� Խ����Եȡ����õ�Same Padding
    """
    return tf.nn.conv2d(input_x, input_w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(value_x):
    """
    #�ػ��� ������������Ϊ2��2,��������strides����Ϊ2��2
    value_x: 4ά [batch, height, width, channels]=[���ݿ飬�������ڸߣ��������ڵĿ�ͨ����]
    ksize: ����������value_x����ÿһά�ȵĴ��ڴ�С���м���ά���������ڴ�С
    strides:����������ÿһ��ά��(N-H-W-C�ĸ�ά��)�Ĳ�����������
    Returns: A Tensor,The max pooled output tensor.����ÿ���������ڵ����ֵ����ɵ�����
    """
    return tf.nn.max_pool(value_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

"""
    ��һ����+�ػ���
    �������ʱʹ����[5,5,1,32]�ľ���˼�[1,1,1,1]�Ĳ�����
    �ܽ᣺
         1.��䷽ʽpadding = 'SAME'����Ե��䣩:(Ȩ�ش���)�����[m��m],��������Ϊ[n,n],����/�ػ���ͼƬ��С��С��ԭ����1/n
         2.��䷽ʽpadding = 'VALID'����Ե����䣩:(Ȩ�ش���)�����[m��m],��������Ϊ[1,1],ͼƬ��Сp��p,����/�ػ���ͼƬ��С:p-(m-1)
                                    (Ȩ�ش���)�����[m��m],��������Ϊ[n,n],ͼƬ��Сp��p,����/�ػ���ͼƬ��С:�Ա�Ե����䷽ʽ����
    ��ͬ�Ծ�����������С����Ӱ��
"""
w_conv1 = weight_variable([5, 5, 1, 32])  # �����Ȩ������ǰ��ά��Ȩ�ش���/patch/kernel/filter����С = ����㻬�����ڵĴ�С(����Ϊ5��5)
                                          # 1������ͨ����һ��ͼƬ�����ֶ�����32�����ͨ������32�����������
b_conv1 = bias_variable([32])  # ƫ������������ͨ������ͬ
x_image_conv1 = tf.reshape(x, [-1, 28, 28, 1])  # ԭ������x��2d������Ϊ������һ�㣬���ǰ�x���һ��4d���������2��3ά��ӦͼƬ�Ŀ��ߣ�28��28=784�������һά����ͼƬ����ɫͨ����(��Ϊ�ǻҶ�ͼ���������ͨ����Ϊ1�������rgb��ɫͼ����Ϊ3)��
h_conv1_relu1 = tf.nn.relu(conv2d(x_image_conv1, w_conv1) + b_conv1)  # ԭ��1������ͼƬ�Ⱦ��������õ�32��28��28��1����ͼƬ���پ��������
h_pool1 = max_pool_2x2(h_conv1_relu1)  # ��һ��ػ�֮��õ�32��14��14��1������ͼƬ����Ϊ������������Ϊ2,��ͼƬ��С��Сһ����

"""�ڶ�������+�ػ���"""
w_conv2 = weight_variable([5, 5, 32, 64])  # ��һ��32�����ͨ����Ϊ�ڶ��������ͨ�����ֶ����õڶ������ͨ��64
b_conv2 = bias_variable([64])
x_image_conv2 = h_pool1  # ͼƬx������һ���õ��������32��14��14��1������ͼƬ����Ϊ�ڶ������������
h_conv2_relu2 = tf.nn.relu(conv2d(x_image_conv2, w_conv2) + b_conv2)  # �Ⱦ��������õ�64��14��14��1������ͼƬ���پ�����������
h_pool2 = max_pool_2x2(h_conv2_relu2)  # �ڶ���ػ���õ�64��7��7��1������ͼƬ

""" fully connected
ȫ���Ӳ㣺����ͼƬ���������ص���������Ԫ����Ȩ������
    ���ڣ�ͼƬ�ߴ��С��7x7�����Ǽ���һ����1024����Ԫ��ȫ���Ӳ㣬
    ���ڴ�������ͼƬ�����ǰѳػ������������reshape��һЩ����������Ȩ�ؾ��󣬼���ƫ�ã�Ȼ�����ʹ��ReLU
"""
w_fc = weight_variable([64*7*7, 1024])
b_fc = bias_variable([1024])
x_image_fc = tf.reshape(h_pool2, [-1, 64*7*7])
h_fc_relu = tf.nn.relu(tf.matmul(x_image_fc, w_fc) + b_fc)  # matmul():�������,�õ�n��1024����

"""
    Dropout:
    Ϊ�˼��ٹ���ϣ������������֮ǰ����dropout��������һ��placeholder������һ����Ԫ�������dropout�б��ֲ���ĸ��ʡ�
    �������ǿ�����ѵ������������dropout���ڲ��Թ����йر�dropout��
    TensorFlow��tf.nn.dropout�������˿���������Ԫ������⣬�����Զ�������Ԫ���ֵ��scale��
    ������dropout��ʱ����Բ��ÿ���scale��
"""
keep_droop_prob = tf.placeholder("float")
h_fc_droop = tf.nn.dropout(h_fc_relu, keep_droop_prob)

"""
    �����:
    ����������һ��softmax�㣬����ǰ��ĵ���softmax regressionһ����
"""
w_soft = weight_variable([1024, 10])
b_soft = bias_variable([10])
x_image_soft = h_fc_droop
y_output = tf.nn.softmax(tf.matmul(x_image_soft, w_soft) + b_soft)  # �õ�n��10����

"""����ģ��: ��������ʧ��С��"""
cross_entropy = tf.reduce_sum(-y*tf.log(y_output))
train_step = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cross_entropy)  # �ݶ��½��Ż���׼ȷ��99.26;ѧϰ�ʿɱ仯
# train_step = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(cross_entropy)  # 20000��ѵ������ʱԼ5Сʱ��׼ȷ��98.9
correct_pridict = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pridict, "float"))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())  # ���б���Ҫ�ȶ�����ʼ��������ᱨ���ڸò���֮��Ŷ���ı���δ��ʼ����
    # train_step = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cross_entropy)  # ����ĳЩ����δ��ʼ����
    start_time = time.clock()
    for i in range(1000):
        batch = mist_data.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y: batch[1], learn_rate: 0.1, keep_droop_prob: 0.5})  # ѵ��
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_droop_prob: 1.0})
            print "ѵ����%d����׼ȷ��Ϊ%f" % (i, train_accuracy)
            # test_accuracy = accuracy.eval(feed_dict={x: mist_data.test.images, y: mist_data.test.labels, keep_droop_prob: 1.0})
            # print "���Ե�%d����׼ȷ��Ϊ%f" % (i, test_accuracy)
    end_time = time.clock()
    test_accuracy = accuracy.eval(feed_dict={x: mist_data.test.images, y: mist_data.test.labels, keep_droop_prob: 1.0})
    print "���ղ���׼ȷ��Ϊ%f" % test_accuracy
    print "��ʱ%f��" % (end_time-start_time)
