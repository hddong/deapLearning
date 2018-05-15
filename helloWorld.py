# 参考：http://cwiki.apachecn.org/pages/viewpage.action?pageId=10029377

import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

sess = tf.Session()
print(sess.run([node1, node2]))

node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))

# 使用变量赋值
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
print(sess.run(adder_node, {a: 4, b: 10}))

# 参数变换
adder_and_triple = adder_node * 3
print(sess.run(adder_and_triple, {a: 5, b: 6}))

# constant初始化常数，其值永远不会变
# variable定义变量，必须显式用global_variables_initialize 初始化
w = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)

liner_model = w * x + b

# 初始化变量
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(liner_model, {x: [1, 2, 3, 4]}))

# 定义模型损失
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(liner_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# 修改变量的值
fixW = tf.assign(w, [-1.])
fixB = tf.assign(b, [1.])
sess.run([fixW, fixB])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# train api
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init)
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
print(sess.run([w, b]))