# TensorFLow变量的使用

import tensorflow as tf

# 增加一个变量op
x = tf.Variable([1, 2])
# 增加一个减法op
a = tf.constant([3, 3])
# 增加一个减法op
sub = tf.subtract(x, a)
# 增加一个加法op
add = tf.add(x, sub)

# 变量在使用之前需要进行初始化
init = tf.global_variables_initializer()

with tf.Session() as ses:
    ses.run(init)
    print(ses.run(sub))
    print(ses.run(add))
