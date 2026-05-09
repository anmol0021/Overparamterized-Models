import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification      # to generate synthetic data

# reproducibility
tf.random.set_seed(0)
np.random.seed(0)           

# generating separable synthetic data

X, y = make_classification(n_samples=2000, n_features=20, n_classes=2, n_informative=20, n_redundant=0, n_clusters_per_class=1, class_sep=3.0, flip_y=0)

# convert labels from {0,1} to {-1,1}
y = 2 * y - 1
# convert to TensorFlow tensors
X = tf.convert_to_tensor(X, dtype=tf.float32)
y = tf.convert_to_tensor(y, dtype=tf.float32)

# deep linear netwok
class DeepLinear(tf.keras.Model):      # we pass tf,keras.Model in the class because we want to use the functionalities of keras.Model like fit, compile, etc.
    def __init__(self, d, depth=5):
        super().__init__()
        self.layers_list = []
        
        for i in range(depth):
            layer = tf.keras.layers.Dense(d, use_bias=False)
            self.layers_list.append(layer)                         # [layer1, layer2, layer3, layer4, layer5,......]

    def call(self, x):
        for layer in self.layers_list:
            x = layer(x)                                            # x->W1x->W2(W1x) -> W3(W2(W1x)) -> W4(W3(W2(W1x))) -> W5(W4(W3(W2(W1x)))) -> .....
        return x


# effective weight of whole deep linear network
def effective_weight(model):
    # first layer's weights
    W = model.layers_list[0].weights[0]              # taking weights of the first layer. here, number of nuerons is d but features is 20, so the shape of W is (20,d) because we have 20 features and d neurons in the first layer. we are only interested in the weights of the first layer because the weights of the subsequent layers will be multiplied with the weights of the first layer to get the effective weight vector.
    W = tf.matmul(layer.weights[0], W)           

    ones = tf.ones((W.shape[1], 1))
    
    w = tf.matmul(tf.transpose(W), ones)
    return tf.squeeze(w)                                    # we squeeze the output to get a 1D tensor of shape (d,) instead of (d,1). usually tensor will operate on size (d,1) but for plotting and visualization we want to have a 1D tensor of shape (d,)

# computing the margin of the effective weight vector

def compute_margin(model, X, y):
    w = effective_weight(model)
    logits = tf.matmul(X, tf.expand_dims(w, axis=1))
    logits = tf.squeeze(logits)
    margins = y * logits
    min_margin = tf.reduce_min(margins)
    margin = min_margin / (tf.norm(w) + 1e-8)

    return margin.numpy()

# norm
def weight_norm(model):
    w = effective_weight(model)
    return tf.norm(w).numpy()

# training
def train(optimizer_name):
    model = DeepLinear(d=20, depth=5)

    # building model once
    dummy_input = tf.random.normal((1, 20))                 # this will create a dummy input of shape (1,20) which is the same as the number of features in our dataset. we need to build the model once before we can access the weights of the layers. this is because the weights are created when the model is called for the first time. so we need to call the model with a dummy input to create the weights.
    model(dummy_input)

    if optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    elif optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    elif optimizer_name == 'momentum':
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

    margins = []
    norms = []

    # training loop
    for epoch in range(1000):
        with tf.GradientTape() as tape:
            logits = model(X)

            y_binary = (y + 1) / 2   # convert labels from {-1,1} to {0,1} for binary cross-entropy loss
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_binary, logits=tf.squeeze(logits)))
            grads = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            margins.append(compute_margin(model, X, y))
            norms.append(weight_norm(model))

            return margins, norms
        

sgd_margins, sgd_norms = train('sgd')
adam_margins, adam_norms = train('adam')
momentum_margins, momentum_norms = train('momentum')

plt.figure(figsize=(12, 5))
plt.plot(sgd_margins, label='SGD')
plt.plot(adam_margins, label='Adam')
plt.plot(momentum_margins, label='Momentum')
plt.xlabel('Epoch')
plt.ylabel('Margin')
plt.title('Margin Growth')
plt.legend()
plt.grid(True)
plt.show()


























