import numpy as np
import tensorflow as tf

# 模拟函数 y=x*w+b
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.5  # 定义方程式 y = x * 0.1 + 0.5

weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # 生成随机数 初始化w
biases = tf.Variable(tf.zeros([1]))
y = weight * x_data + biases # 定义拟合函数

loss = tf.reduce_mean(tf.square(y - y_data)) # 计算误差方差
optimizer = tf.train.GradientDescentOptimizer(0.5) # 应该是梯度下降
train = optimizer.minimize(loss) # 建立训练器 使误差最小

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in range(400):  # 循环训练 400次
    sess.run(train) # 调度训练器
    if step%20 == 0:
        print(step, sess.run(weight), sess.run(biases))