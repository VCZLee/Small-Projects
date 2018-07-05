import numpy as np
np.random.seed(10)

num_visible = 4
num_hidden = 3
learning_rate = 0.1

num_weights = num_visible*num_hidden

visible = np.random.randint(2, size=(1, num_visible))
weights = np.random.randn(num_visible, num_hidden)
forward_bias = np.random.randn(1, num_hidden)

hidden_activation = visible.dot(weights) + forward_bias
hidden_prob = 1/(1+np.exp(-hidden_activation))
hidden = (np.random.uniform(size=num_hidden) < hidden_prob).astype(int)
backward_bias = np.random.randn(1, num_visible)

for epoch in range(25):
    recon_activation = hidden.dot(np.transpose(weights)) + backward_bias
    recon_prob = 1/(1+np.exp(-recon_activation))
    recon = (np.random.uniform(size=num_visible) < recon_prob).astype(int)

    recon_hidden_activation = recon_prob.dot(weights) + forward_bias
    recon_hidden_prob = 1/(1+np.exp(-recon_hidden_activation))
    recon_hidden = (np.random.uniform(size=num_hidden) < recon_hidden_prob).astype(int)
    hidden = recon_hidden
    contrastive_divergence = np.outer(visible, hidden) - np.outer(recon, recon_hidden)
    weights = weights + learning_rate*contrastive_divergence

