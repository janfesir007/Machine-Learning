# -*-coding:gbk-*-

"""
����̽��Tensorfolw�еı���Variable���÷�������Ҫ��
�Ƚ������������Ĳ��
state = tf.assign(state, new_value)
state = new_value
"""
import tensorflow as tf
state = tf.Variable(0, name='counter')  # ����
one = tf.constant(1)  # ����
new_value = tf.add(state, one)
state = new_value  # �����1 1 1
# state = tf.assign(state, new_value)  # �����1 2 3 assign():���ڶ���������ֵ����һ������
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
       print sess.run(state)
