import numpy as np
import tensorflow as tf


class RestrictedBoltzmannMachine:
    def __init__(self, num_hidden):
        self.weights = None
        self.forward_bias = None
        self.backward_bias = None
        self.num_hidden = num_hidden
        self.num_visible = None

    def fit(self, x, learning_rate, batch_size, num_epochs, num_cd_iterations=1, use_sampling=False):
        with tf.Session() as sess:
            self.num_visible = x.shape[1]
            if self.weights is not None:
                weights = tf.Variable(self.weights)
            else:
                weights = tf.Variable(tf.random_normal((self.num_visible, self.num_hidden)))
            if self.forward_bias is not None:
                forward_bias = tf.Variable(self.forward_bias)
            else:
                forward_bias = tf.Variable(tf.random_normal((1, self.num_hidden)))
            if self.backward_bias is not None:
                backward_bias = tf.Variable(self.backward_bias)
            else:
                backward_bias = tf.Variable(tf.random_normal((1, self.num_visible)))

            visible = tf.placeholder(tf.float32, [None, self.num_visible])
            hidden_activation = tf.add(tf.matmul(visible, weights), forward_bias)
            hidden_prob = tf.divide(1.0, tf.add(1.0, tf.exp(tf.negative(hidden_activation)))) #sigmoid activation
            if use_sampling:
                hidden = tf.to_float(tf.less(tf.random_uniform((1, self.num_hidden)), hidden_prob))
            else:
                hidden = hidden_prob
            hidden_intermediary = hidden

            for cd_iteration in range(num_cd_iterations):
                recon_activation = tf.add(tf.matmul(hidden_intermediary, weights, transpose_b=True), backward_bias)
                recon_prob = tf.divide(1.0, tf.add(1.0, tf.exp(tf.negative(recon_activation))))
                if use_sampling:
                    recon = tf.to_float(tf.less(tf.random_uniform((1, self.num_visible)), recon_prob))
                else:
                    recon = recon_prob

                recon_hidden_activation = tf.add(tf.matmul(recon, weights), forward_bias)
                recon_hidden_prob = tf.divide(1.0, tf.add(1.0, tf.exp(tf.negative(recon_hidden_activation))))
                if use_sampling:
                    recon_hidden = tf.to_float(tf.less(tf.random_uniform((1, self.num_hidden)), recon_hidden_prob))
                else:
                    recon_hidden = recon_hidden_prob

                hidden_intermediary = recon_hidden

            first_outer_product = tf.matmul(visible, hidden, transpose_a=True)
            second_outer_product = tf.matmul(recon, recon_hidden, transpose_a=True)

            contrastive_divergence = tf.divide(tf.subtract(first_outer_product, second_outer_product), x.shape[0])
            forward_bias_update = tf.reduce_mean(tf.subtract(hidden, recon_hidden), 0, True)
            backward_bias_update = tf.reduce_mean(tf.subtract(visible, recon), 0, True)

            update_weights = tf.assign_add(weights, tf.multiply(learning_rate, contrastive_divergence))
            update_forward_bias = tf.assign_add(forward_bias, tf.multiply(learning_rate, forward_bias_update))
            update_backward_bias = tf.assign_add(backward_bias, tf.multiply(learning_rate, backward_bias_update))

            sess.run(tf.global_variables_initializer())
            for epoch in range(num_epochs):
                random_rows = np.random.choice(x.shape[0], batch_size, replace=False)
                sess.run((update_weights, update_forward_bias, update_backward_bias), {visible: x[random_rows]})

            self.weights = weights.eval()
            self.forward_bias = forward_bias.eval()
            self.backward_bias = backward_bias.eval()

            return self

    def transform(self, x, use_sampling=False):
        with tf.Session() as sess:
            visible = tf.placeholder(tf.float32, [None, self.num_visible])
            hidden_activation = tf.add(tf.matmul(visible, self.weights), self.forward_bias)
            hidden_prob = tf.divide(1.0, tf.add(1.0, tf.exp(tf.negative(hidden_activation)))) #sigmoid activation
            if use_sampling:
                hidden = tf.to_float(tf.less(tf.random_uniform((1, self.num_hidden)), hidden_prob))
            else:
                hidden = hidden_prob

            return sess.run(hidden, {visible: x})

    def inverse_transform(self, x, use_sampling=False):
        with tf.Session() as sess:
            hidden = tf.placeholder(tf.float32, [None, self.num_hidden])
            recon_activation = tf.add(tf.matmul(hidden, self.weights, transpose_b=True), self.backward_bias)
            recon_prob = tf.divide(1.0, tf.add(1.0, tf.exp(tf.negative(recon_activation))))
            if use_sampling:
                recon = tf.to_float(tf.less(tf.random_uniform((1, self.num_visible)), recon_prob))
            else:
                recon = recon_prob

            return sess.run(recon, {hidden: x})

    def reconstruct(self, x, use_sampling=False):
        with tf.Session() as sess:
            visible = tf.placeholder(tf.float32, [None, self.num_visible])
            hidden_activation = tf.add(tf.matmul(visible, self.weights), self.forward_bias)
            hidden_prob = tf.divide(1.0, tf.add(1.0, tf.exp(tf.negative(hidden_activation)))) #sigmoid activation
            if use_sampling:
                hidden = tf.to_float(tf.less(tf.random_uniform((1, self.num_hidden)), hidden_prob))
            else:
                hidden = hidden_prob

            recon_activation = tf.add(tf.matmul(hidden, self.weights, transpose_b=True), self.backward_bias)
            recon_prob = tf.divide(1.0, tf.add(1.0, tf.exp(tf.negative(recon_activation))))
            if use_sampling:
                recon = tf.to_float(tf.less(tf.random_uniform((1, self.num_visible)), recon_prob))
            else:
                recon = recon_prob

            return sess.run(recon, {visible: x})


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Binarizer

data = pd.read_csv("/media/jintoboy/Main Storage/ML_Projects/MNIST/train.csv", index_col = None)
data = data.loc[(data.label == 7)]
data = data.drop('label', 1)

data = data.values

quantizer = Binarizer(threshold = 127.5).fit(data)
data = quantizer.transform(data)

random_rows = np.random.choice(data.shape[0], 20, replace=False)

ROW = 4
COLUMN = 5
for i in range(ROW * COLUMN):
    image = data[random_rows, :][i].reshape((28, 28))
    plt.subplot(ROW, COLUMN, i+1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # do not show axis value
plt.tight_layout()   # automatic padding between subplots
plt.show()

rbm = RestrictedBoltzmannMachine(num_hidden=100)
rbm = rbm.fit(data, 0.15, 3, 10000)

reconstruction = rbm.reconstruct(data[random_rows, :])

for i in range(ROW * COLUMN):
    image = reconstruction[i].reshape((28, 28))
    plt.subplot(ROW, COLUMN, i+1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()