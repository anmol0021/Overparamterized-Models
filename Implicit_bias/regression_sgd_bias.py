import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.random.set_seed(42)                
np.random.seed(42)

n = 1000      # number of samples
true_d = 10   # true dimension of the data

# creating synthetic linear dataset
X = tf.random.normal(shape=(n, true_d))

# ground truth weight parameter
theta_star = tf.random.normal((true_d, 1))

# noiseless labels
y = tf.matmul(X, theta_star)     # this is the true linear model without noise

# feature expanision function
def expand_features(X, d):
    """
    Convert feature dimension to d.

    Cases:
    1. d == original dimension:
         keep same features

    2. d < original dimension:
         truncate features

    3. d > original dimension:
         append random noisy features
    """

    n, d0 = X.shape

    if d == d0:
        return X
    elif d < d0:
        return X[:, :d]
    else:
        extra = tf.random.normal((n, d - d0))
        return tf.concat([X, extra], axis=1)        # here we are concatenating coklumns wise so both arrays should have same number of rows
    
# linear model
class LinearModel(tf.Module):
    def __init__(self, d):
        super().__init__()
        self.theta = tf.Variable(tf.zeros((d, 1)), trainable=True, dtype=tf.float32)

    def __call__(self, X):
        return tf.matmul(X, self.theta)



# sgd training function
def train_sgd(X, y, d, lr=0.01, epochs=20000):
    Xd = expand_features(X, d)
    model = LinearModel(d)
    
    optimizer = tf.optimizers.SGD(learning_rate=lr)
    norms = []
    n_samples = X.shape[0]

    for epoch in range(epochs):
        # sgd step with bacth size of 1
        idx = np.random.randint(0, n_samples)             # randomly select one sample

        xb = tf.expand_dims(Xd[idx], axis=0)              # get the corresponding feature vector and reshape it to (1, d) -> row vector
        yb = tf.expand_dims(y[idx], axis=0)              # get the corresponding label and reshape it to (1, 1)

        # compute the loss and gradients
        with tf.GradientTape() as tape:                     
            pred = model(xb)                                 # forward pass
            loss = tf.reduce_mean((pred - yb) ** 2)        # mean squared error loss

        grads = tape.gradient(loss, model.trainable_variables)     # compute gradients
        optimizer.apply_gradients(zip(grads, model.trainable_variables))   # update parameters. zip is used to pair each gradient with its corresponding variable

        # track the norm of the weight vector
        norm = tf.norm(model.theta).numpy()     # compute the L2 norm of the weight vector and convert it to a numpy scalar
        norms.append(norm)

    return norms


d_list = [10, 50, 100, 2000]

plt.figure(figsize=(10, 6))

for d in d_list:
    norms = train_sgd(X, y, d)
    plt.plot(norms, label=f'd={d}')

plt.xlabel("Epoch")
plt.ylabel(r"$||w||_2$")
plt.title("SGD implicit bias: norm evolution vs epochs")

plt.grid(True)
plt.legend()

plt.savefig("norm_convergence.png", dpi=300)

plt.show()

