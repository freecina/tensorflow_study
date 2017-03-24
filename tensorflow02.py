import tensorflow as tf 
import numpy as np
import time

# author : freecina
# time : 2017-03-24
# addr :chengdu

t1 = time.time()
#tf.contrib.learn
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

x = np.array([1.,2.,3.,4.])
y = np.array([0.,-1.,-2.,-3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4, num_epochs=1000)

estimator.fit(input_fn=input_fn, steps=1000)

print estimator.evaluate(input_fn=input_fn)
t2 = time.time() - t1 
print('time is %.2f s'%t2)













