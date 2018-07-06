import numpy as np

num_visible = 6
num_hidden = 3
learning_rate = 0.1
batch_size = 7
cd_iterations = 1
num_epochs = 2500

num_weights = num_visible*num_hidden

data = np.array([[1, 1, 1, 0, 0, 0],
                 [1, 0, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [0, 0, 1, 1, 1, 0],
                 [0, 0, 1, 1, 0, 0],
                 [0, 0, 1, 1, 1, 0],
                 [1, 1, 1, 0, 0, 1]])

weights = np.random.randn(num_visible, num_hidden)
forward_bias = np.random.randn(1, num_hidden)
backward_bias = np.random.randn(1, num_visible)


for epoch in range(num_epochs):
    random_rows = np.random.choice(data.shape[0], batch_size, replace=False)
    train = data[random_rows, :]
    contrastive_divergence_list = []
    forward_bias_list = []
    backward_bias_list = []
    for row in range(train.shape[0]):
        visible = train[row,:]
        hidden_activation = visible.dot(weights) + forward_bias
        hidden_prob = 1 / (1 + np.exp(-hidden_activation))
        hidden = (np.random.uniform(size=num_hidden) < hidden_prob).astype(int)

        for iteration in range(cd_iterations):
            recon_activation = hidden.dot(np.transpose(weights)) + backward_bias
            recon_prob = 1/(1+np.exp(-recon_activation))
            recon = (np.random.uniform(size=num_visible) < recon_prob).astype(int)

            recon_hidden_activation = recon_prob.dot(weights) + forward_bias
            recon_hidden_prob = 1/(1+np.exp(-recon_hidden_activation))
            recon_hidden = (np.random.uniform(size=num_hidden) < recon_hidden_prob).astype(int)
            if iteration < cd_iterations - 1:
                hidden = recon_hidden
            contrastive_divergence_list.append(np.outer(visible, hidden) - np.outer(recon, recon_hidden))
            forward_bias_list.append(hidden - recon_hidden)
            backward_bias_list.append(visible - recon)
    weights = weights + learning_rate*np.mean(contrastive_divergence_list, axis=0)
    forward_bias = forward_bias + learning_rate*np.mean(forward_bias_list, axis=0)
    backward_bias = backward_bias + learning_rate*np.mean(backward_bias_list, axis=0)

output = np.empty((0,data.shape[1])).astype(int)

for row in range(data.shape[0]):
    visible = data[row,:]
    hidden_activation = visible.dot(weights) + forward_bias
    hidden_prob = 1 / (1 + np.exp(-hidden_activation))
    hidden = (np.random.uniform(size=num_hidden) < hidden_prob).astype(int)
    recon_activation = hidden.dot(np.transpose(weights)) + backward_bias
    recon_prob = 1/(1+np.exp(-recon_activation))
    recon = (np.random.uniform(size=num_visible) < recon_prob).astype(int)
    output = np.append(output, recon, axis=0)

print(output-data)

class RestrictedBoltzmannMachine:
    def __init__(self,num_hidden):
        self.weights = np.random.randn(num_visible, num_hidden)
        self.forward_bias = np.random.randn(1, num_hidden)
        self.backward_bias = np.random.randn(1, num_visible)







# weights = weights + learning_rate*np.mean(contrastive_divergence_list, axis=0)
# forward_bias = forward_bias + learning_rate*np.mean(forward_bias_list, axis=0)
# backward_bias = backward_bias + learning_rate*np.mean(backward_bias_list, axis=0)


# print(np.mean([a,b], axis=0))

# num_visible = 6
# num_hidden = 2
# learning_rate = 0.1
#
# num_weights = num_visible*num_hidden
#
# visible = np.random.randint(2, size=(1, num_visible))
# weights = np.random.randn(num_visible, num_hidden)
# forward_bias = np.random.randn(1, num_hidden)
#
# hidden_activation = visible.dot(weights) + forward_bias
# hidden_prob = 1/(1+np.exp(-hidden_activation))
# hidden = (np.random.uniform(size=num_hidden) < hidden_prob).astype(int)
# backward_bias = np.random.randn(1, num_visible)
#
# for epoch in range(1000):
#     recon_activation = hidden.dot(np.transpose(weights)) + backward_bias
#     recon_prob = 1/(1+np.exp(-recon_activation))
#     recon = (np.random.uniform(size=num_visible) < recon_prob).astype(int)
#
#     recon_hidden_activation = recon_prob.dot(weights) + forward_bias
#     recon_hidden_prob = 1/(1+np.exp(-recon_hidden_activation))
#     recon_hidden = (np.random.uniform(size=num_hidden) < recon_hidden_prob).astype(int)
#     hidden = recon_hidden
#
#     contrastive_divergence = np.outer(visible, hidden) - np.outer(recon, recon_hidden)
#     weights = weights + learning_rate*contrastive_divergence
#     forward_bias =  forward_bias + learning_rate*(hidden - recon_hidden)
#     backward_bias = backward_bias + learning_rate*(visible - recon)