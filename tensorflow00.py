import tensorflow as tf
import time
# author:freecina
# time:2017-03-24
# addr:chengdu
time1 = time.time()
node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
# print(node1, node2)
sess = tf.Session()
node3 = tf.add(node1, node2)
print('node3:', node3)
print(sess.run(node3))

#placeholdes
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b 
add_and_triple = adder_node*9
print(sess.run(add_and_triple, {a:3, b:8}))
print(sess.run(adder_node, {a:[3,9], b:[5,8]}))

#simple linear model
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x:[1,2,3,4]}))


#loss function
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
# this loss is 23.66
#optimal parameters 
fixW = tf.assign(W, [-1.0])
fixb = tf.assign(b, [1.0])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
#this loss change to 0


#gradient descent
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init) #reset values to incorrect default
for i in range(1000):
	sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
	print('step is %d'%i)
	# print('loss is :',sess.run(loss,{x:[1,2,3,4], y:[0,-1,-2,-3]}))
	print(sess.run([W,b]))
# W is -0.9999969 , b is 0.99999082 
time2 = time.time()
print('time is : %.2f s'%(time2-time1))