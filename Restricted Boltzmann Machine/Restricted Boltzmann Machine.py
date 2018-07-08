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
        self.weights = None
        self.forward_bias = None
        self.backward_bias = None

    def fit(self, x, learning_rate, batch_size, cd_iterations, epochs):
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
                hidden = (np.random.uniform(size=self.num_hidden) < hidden_prob).astype(int)
                for iteration in range(cd_iterations):
                    recon_activation = hidden.dot(np.transpose(self.weights)) + self.backward_bias
                    recon_prob = 1 / (1 + np.exp(-recon_activation))
                    recon = (np.random.uniform(size=self.num_visible) < recon_prob).astype(int)
                    recon_hidden_activation = recon_prob.dot(self.weights) + self.forward_bias
                    recon_hidden_prob = 1 / (1 + np.exp(-recon_hidden_activation))
                    recon_hidden = (np.random.uniform(size=self.num_hidden) < recon_hidden_prob).astype(int)
                    if iteration < cd_iterations - 1:
                        hidden = recon_hidden
                contrastive_divergence_list.append(np.outer(visible, hidden) - np.outer(recon, recon_hidden))
                forward_bias_list.append(hidden - recon_hidden)
                backward_bias_list.append(visible - recon)
            self.weights = self.weights + learning_rate * np.mean(contrastive_divergence_list, axis=0)
            self.forward_bias = self.forward_bias + learning_rate * np.mean(forward_bias_list, axis=0)
            self.backward_bias = self.backward_bias + learning_rate * np.mean(backward_bias_list, axis=0)
        return self

    def reconstruct(self,x):
        output = np.empty((0, x.shape[1])).astype(int)
        for row in range(x.shape[0]):
            visible = x[row, :]
            hidden_activation = visible.dot(self.weights) + self.forward_bias
            hidden_prob = 1 / (1 + np.exp(-hidden_activation))
            hidden = (np.random.uniform(size=self.num_hidden) < hidden_prob).astype(int)
            recon_activation = hidden.dot(np.transpose(self.weights)) + self.backward_bias
            recon_prob = 1 / (1 + np.exp(-recon_activation))
            recon = (np.random.uniform(size=self.num_visible) < recon_prob).astype(int)
            output = np.append(output, recon, axis=0)
        return output


data = pd.read_csv("C:/Users/Jintoboy/PycharmProjects/Jupyter-Notebooks/Kaggle/MNIST/Data/train.csv", index_col = None)
data = data.loc[(data.label == 3)]
data = data.drop('label', 1)

quantizer = Binarizer(threshold = 127.5).fit(data)
data = quantizer.transform(data)

ROW = 4
COLUMN = 5
for i in range(ROW * COLUMN):
    image = data[i].reshape((28,28))
    plt.subplot(ROW, COLUMN, i+1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # do not show axis value
plt.tight_layout()   # automatic padding between subplots
plt.show()


rbm = RestrictedBoltzmannMachine(num_hidden=25)
rbm = rbm.fit(data,learning_rate=0.1, batch_size=5, cd_iterations=1, epochs=2500)

reconstruction = rbm.reconstruct(data[1:21])

for i in range(ROW * COLUMN):
    image = reconstruction[i].reshape((28,28))
    plt.subplot(ROW, COLUMN, i+1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()