# 测试一个变量的循环自增

import tensorflow as tf

# 创建一个变量初始化为0
state = tf.Variable(0, name='counter')
# 创建一个op，作用是使state+1
new_value = tf.add(state, 1)
# 创建一个op，用来赋值
update = tf.assign(state, new_value)
init = tf.initialize_all_variables()

with tf.Session() as ses:
    ses.run(init)
    print(ses.run(state))
    for _ in range(5):
        ses.run(update)
        print(ses.run(state))
