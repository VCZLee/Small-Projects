import numpy as np
import tensorflow as tf

x = np.array([[1, 1, 1, 0, 1],
              [1, 1, 1, 0, 0],
              [0, 0, 0, 1, 1],
              [0, 0, 0, 0, 1],
              [1, 1, 1, 1, 0],
              [0, 0, 0, 1, 0]])

num_visible = x.shape[1]
num_hidden = 3

with tf.Session() as sess:
    tf_weights = tf.random_normal((num_visible, num_hidden))
    tf_forward_bias = tf.random_normal((1, num_hidden))
    tf_backward_bias = tf.random_normal((1, num_visible))