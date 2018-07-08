import numpy as np

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


data = np.array([[1, 1, 1, 0, 0, 0],
                 [1, 0, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [0, 0, 1, 1, 1, 0],
                 [0, 0, 1, 1, 0, 0],
                 [0, 0, 1, 1, 1, 0],
                 [1, 1, 1, 0, 0, 1]])

rbm = RestrictedBoltzmannMachine(num_hidden=4)
rbm = rbm.fit(data, 0.1, 3, 1, 10000)
print(rbm.reconstruct(data))