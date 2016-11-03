# -*-coding:gbk-*-

"""
本例探讨Tensorfolw中的变量Variable的用法，很重要！
比较如下两条语句的差别：
state = tf.assign(state, new_value)
state = new_value
"""
import tensorflow as tf
state = tf.Variable(0, name='counter')  # 变量
one = tf.constant(1)  # 常量
new_value = tf.add(state, one)
state = new_value  # 结果：1 1 1
# state = tf.assign(state, new_value)  # 结果：1 2 3 assign():将第二个参数赋值给第一个参数
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
       print sess.run(state)
