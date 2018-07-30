import numpy as np
import tensorflow as tf

# x = np.array([[1, 1, 1, 0, 1],
#               [1, 1, 1, 0, 0],
#               [0, 0, 0, 1, 1],
#               [0, 0, 0, 0, 1],
#               [1, 1, 1, 1, 0],
#               [0, 0, 0, 1, 0]])

x = np.array([[1, 1, 1, 0, 1]])

num_visible = x.shape[1]
num_hidden = 3
use_sampling = False

with tf.Session() as sess:
    weights = tf.random_normal((num_visible, num_hidden))
    forward_bias = tf.random_normal((1, num_hidden))
    backward_bias = tf.random_normal((1, num_visible))
    visible = tf.placeholder(tf.float32, [None, num_visible])
    hidden_activation = tf.add(tf.matmul(visible, weights),forward_bias)
    hidden_prob = tf.divide(1.0, tf.add(1.0, tf.exp(tf.negative(hidden_activation))))
    print(sess.run(hidden_prob, {visible: x}))