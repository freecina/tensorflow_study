import numpy as np 
import tensorflow as tf 
import time

# author : freecina
# time = 2017-03-24
# addr = chengdu
t1 = time.time()
def model(features, labels, mode):
	#Buil a linear model and predict values
	W = tf.get_variable("W",[1], dtype=tf.float64)
	b = tf.get_variable("b",[1], dtype=tf.float64)
	y = W*features['x'] + b
	#Loss sub-graph
	loss = tf.reduce_sum(tf.square(y - labels))
	global_step = tf.train.get_global_step()
	optimizer = tf.train.GradientDescentOptimizer(0.01)
	#ModelFnOps connects subgrahs we build to the appropriate functionality
	train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step,1))
	return tf.contrib.learn.ModelFnOps(mode=mode, predictions=y, loss=loss, train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)
# define data set
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, 4, num_epochs=1000)

#train
estimator.fit(input_fn=input_fn, steps=1000)
#evaluate model
print(estimator.evaluate(input_fn=input_fn, steps=10))
t2 = time.time()
print('time is : %.2f s'%(t2-t1))
#results : {'loss': 1.042303e-10, 'global_step': 1000}









