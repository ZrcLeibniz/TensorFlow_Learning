# Fetch and Feed

import tensorflow as tf

# Fetch
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input2, input3)
mul = tf.multiply(input1, add)

with tf.Session() as ses:
    result = ses.run([mul, add])
    print(result)

# Feed
# 创建占位符
input01 = tf.placeholder(tf.float32)
input02 = tf.placeholder(tf.float32)
output = tf.multiply(input01, input02)

with tf.Session() as ses:
    print(ses.run(output, feed_dict={input01: [7.0], input02: [2.0]}))
