# 创建图和启动图

import tensorflow as tf
# 创建一个常量op(一行两列的矩阵)
m1 = tf.constant([[3, 3]])
# 创建一个常量op(两行一列的矩阵)
m2 = tf.constant([[2], [3]])
# 创建一个矩阵乘法op,把m1和m2传入
product = tf.matmul(m1, m2)

# 定义一个会话，启动默认的图
ses = tf.Session()
# 调用ses的run方法来执行矩阵乘法op
# run(product)出发了图中的三个op
result = ses.run(product)
print(result)
ses.close()
