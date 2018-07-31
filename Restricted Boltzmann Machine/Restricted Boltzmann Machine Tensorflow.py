import numpy as np
import tensorflow as tf

x = np.array([[1, 1, 1, 0, 1],
              [1, 1, 1, 0, 0],
              [0, 0, 0, 1, 1],
              [0, 0, 0, 0, 1],
              [1, 1, 1, 1, 0],
              [0, 0, 0, 1, 0]])

# x = np.array([[1, 0, 0, 0, 0]])

num_visible = x.shape[1]
num_hidden = 2
use_sampling = True
learning_rate = 0.1

with tf.Session() as sess:
    weights = tf.random_normal((num_visible, num_hidden))
    forward_bias = tf.random_normal((1, num_hidden))
    backward_bias = tf.random_normal((1, num_visible))
    visible = tf.placeholder(tf.float32, [None, num_visible])
    hidden_activation = tf.add(tf.matmul(visible, weights), forward_bias)
    hidden_prob = tf.divide(1.0, tf.add(1.0, tf.exp(tf.negative(hidden_activation))))
    if use_sampling:
        hidden = tf.to_float(tf.less(tf.random_uniform((1, num_hidden)), hidden_prob))
    else:
        hidden = hidden_prob
    recon_activation = tf.add(tf.matmul(hidden, weights, transpose_b=True), backward_bias)
    recon_prob = tf.divide(1.0, tf.add(1.0, tf.exp(tf.negative(recon_activation))))
    if use_sampling:
        recon = tf.to_float(tf.less(tf.random_uniform((1, num_visible)), recon_prob))
    else:
        recon = recon_prob
    recon_hidden_activation = tf.add(tf.matmul(recon, weights), forward_bias)
    recon_hidden_prob = tf.divide(1.0, tf.add(1.0, tf.exp(tf.negative(recon_hidden_activation))))
    if use_sampling:
        recon_hidden = tf.to_float(tf.less(tf.random_uniform((1, num_hidden)), recon_hidden_prob))
    else:
        recon_hidden = recon_hidden_prob
    first_outer_product = tf.matmul(visible, hidden, transpose_a=True)
    second_outer_product = tf.matmul(recon, recon_hidden, transpose_a=True)
    contrastive_divergence = tf.divide(tf.subtract(first_outer_product, second_outer_product), x.shape[0])
    forward_bias_update = tf.reduce_mean(tf.subtract(hidden, recon_hidden), 0)
    backward_bias_update = tf.reduce_mean(tf.subtract(visible, recon), 0)
    print(sess.run(backward_bias_update, {visible: x}))