import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Binarizer


class RestrictedBoltzmannMachine:
    def __init__(self, num_hidden):
        self.weights = None
        self.forward_bias = None
        self.backward_bias = None
        self.num_hidden = num_hidden
        self.num_visible = None

    def fit(self, x, learning_rate, batch_size, cd_iterations, epochs, use_sampling=False):
        if self.num_visible != x.shape[1]:
            self.num_visible = x.shape[1]
            self.weights = np.random.randn(self.num_visible, self.num_hidden)
            self.forward_bias = np.random.randn(1, self.num_hidden)
            self.backward_bias = np.random.randn(1, self.num_visible)
        for epoch in range(epochs):
            random_rows = np.random.choice(x.shape[0], batch_size, replace=False)
            train = x[random_rows, :]
            contrastive_divergence_list = []
            forward_bias_list = []
            backward_bias_list = []
            for row in range(train.shape[0]):
                visible = train[row, :]
                hidden_activation = visible.dot(self.weights) + self.forward_bias
                hidden_prob = 1 / (1 + np.exp(-hidden_activation))
                if use_sampling:
                    hidden = (np.random.uniform(size=self.num_hidden) < hidden_prob).astype(int)
                else:
                    hidden = hidden_prob
                for iteration in range(cd_iterations):
                    recon_activation = hidden.dot(np.transpose(self.weights)) + self.backward_bias
                    recon_prob = 1 / (1 + np.exp(-recon_activation))
                    if use_sampling:
                        recon = (np.random.uniform(size=self.num_visible) < recon_prob).astype(int)
                    else:
                        recon = recon_prob
                    recon_hidden_activation = recon.dot(self.weights) + self.forward_bias
                    recon_hidden_prob = 1 / (1 + np.exp(-recon_hidden_activation))
                    if use_sampling:
                        recon_hidden = (np.random.uniform(size=self.num_hidden) < recon_hidden_prob).astype(int)
                    else:
                        recon_hidden = recon_hidden_prob
                    if iteration < cd_iterations - 1:
                        hidden = recon_hidden
                contrastive_divergence_list.append(np.outer(visible, hidden) - np.outer(recon, recon_hidden))
                forward_bias_list.append(hidden - recon_hidden)
                backward_bias_list.append(visible - recon)
            self.weights = self.weights + learning_rate * np.mean(contrastive_divergence_list, axis=0)
            self.forward_bias = self.forward_bias + learning_rate * np.mean(forward_bias_list, axis=0)
            self.backward_bias = self.backward_bias + learning_rate * np.mean(backward_bias_list, axis=0)
        return self

    def reconstruct(self, x, use_sampling=False):
        output = np.empty((0, x.shape[1])).astype(int)
        for row in range(x.shape[0]):
            visible = x[row, :]
            hidden_activation = visible.dot(self.weights) + self.forward_bias
            hidden_prob = 1 / (1 + np.exp(-hidden_activation))
            if use_sampling:
                hidden = (np.random.uniform(size=self.num_hidden) < hidden_prob).astype(int)
            else:
                hidden = hidden_prob
            recon_activation = hidden.dot(np.transpose(self.weights)) + self.backward_bias
            recon_prob = 1 / (1 + np.exp(-recon_activation))
            if use_sampling:
                recon = (np.random.uniform(size=self.num_visible) < recon_prob).astype(int)
            else:
                recon = recon_prob
            output = np.append(output, recon, axis=0)
        return output

    def transform(self, x, use_sampling=False):
        output = np.empty((0, self.num_hidden)).astype(int)
        for row in range(x.shape[0]):
            visible = x[row, :]
            hidden_activation = visible.dot(self.weights) + self.forward_bias
            hidden_prob = 1 / (1 + np.exp(-hidden_activation))
            if use_sampling:
                hidden = (np.random.uniform(size=self.num_hidden) < hidden_prob).astype(int)
            else:
                hidden = hidden_prob
            output = np.append(output, hidden, axis=0)
        return output

    def inverse_transform(self, x, use_sampling=False):
        output = np.empty((0, self.num_visible)).astype(int)
        for row in range(x.shape[0]):
            hidden = x[row, :]
            recon_activation = hidden.dot(np.transpose(self.weights)) + self.backward_bias
            recon_prob = 1 / (1 + np.exp(-recon_activation))
            if use_sampling:
                recon = (np.random.uniform(size=self.num_visible) < recon_prob).astype(int)
            else: recon = recon_prob
            output = np.append(output, recon, axis=0)
        return output


data = pd.read_csv("C:/Users/Jintoboy/PycharmProjects/Jupyter-Notebooks/Kaggle/MNIST/Data/train.csv", index_col = None)
data = data.loc[(data.label == 7)]
data = data.drop('label', 1)

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
rbm = rbm.fit(data, learning_rate=0.15, batch_size=3, cd_iterations=1, epochs=10000)


reconstruction = rbm.reconstruct(data[random_rows, :])

for i in range(ROW * COLUMN):
    image = reconstruction[i].reshape((28, 28))
    plt.subplot(ROW, COLUMN, i+1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()