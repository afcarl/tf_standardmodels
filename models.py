from sklearn.base import TransformerMixin
from sklearn.utils import check_array
import tensorflow as tf
import numpy

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class AutoEncoder(TransformerMixin):
    """Auto-Encoder model for feature extraction.

    Implement a (possibly multi-layer) auto-encoder to perform feature extraction using TensorFlow.

    Attributes:
    -----------
    feature_layer_sizes: tuple of ints, default (100, )
        Number of neurons in each layer of the auto-encoder.
        Decoder layers should not be included in the tuple since their size is inferred from those of the encoder.
    activation: {'relu'}, default 'relu'
        Activation function for the hidden layers
        * 'relu', the rectified linear unit function, returns :math:`f(x) = max(0, x)`
        * 'logistic', the logistic sigmoid function, returns :math:`f(x) = 1 / (1 + exp(-x))`.
    learning_rate: float, default: 0.01
        The learning rate used. It controls the step-size in updating the weights.
    n_iter: int, default: 20
        The number of iterations in the learning process.
    radom_seed: int, default: None
        The seed used to initialize TensorFlow at each fit call.
        If None, no seed is forced at fit time.


    References:
    -----------
        Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
        learning applied to document recognition." Proceedings of the IEEE,
        86(11):2278-2324, November 1998.

    Links:
    ------
        [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
    """
    def __init__(self, feature_layer_sizes=(100,), activation="relu", learning_rate=0.01, n_iter=20, batch_size=256,
                 random_seed=None):
        self.feature_layer_sizes_ = feature_layer_sizes
        self.n_feature_layers = len(self.feature_layer_sizes_)
        self.activation = activation
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_seed = random_seed
        self.batch_size = batch_size

        if self.activation == "relu":
            self.activation_fun = tf.nn.relu
        elif self.activation == "logistic":
            self.activation_fun = tf.nn.sigmoid
        else:
            raise ValueError("Invalid activation function name %s" % self.activation)

        self.weights_encoder_ = []
        self.weights_decoder_ = []
        self.bias_encoder_ = []
        self.bias_decoder_ = []

        self.encoder_layers_ = []
        self.decoder_layers_ = []

        self.X_ = None
        self.tf_session_ = None
        self.cost_ = None
        self.optimizer_ = None

    @property
    def final_features_(self):
        return self.encoder_layers_[-1]

    @property
    def X_reconstructed_(self):
        return self.decoder_layers_[-1]

    def fit(self, X):
        if self.random_seed is not None:
            tf.set_random_seed(self.random_seed)
        X = check_array(X, ensure_2d=True, copy=True)
        n_input = X.shape[1]
        n_batches = int(X.shape[0] / self.batch_size)
        self._init_weights_and_bias(n_input)
        self._build_encoder()
        self._build_decoder()
        self.cost_ = tf.reduce_mean(tf.pow(self.X_ - self.X_reconstructed_, 2))
        self.optimizer_ = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost_)

        init = tf.global_variables_initializer()
        self.tf_session_ = tf.Session()
        self.tf_session_.run(init)
        for iteration in range(self.n_iter):
            for batch in range(n_batches):
                indices_batch = numpy.random.choice(X.shape[0], size=self.batch_size)
                self.tf_session_.run(self.optimizer_, feed_dict={self.X_: X[indices_batch]})
            c = self.tf_session_.run(self.cost_, feed_dict={self.X_: X})
            print("Epoch: %04d, cost=%.9f" % (iteration + 1, c))
        print("Optimization finished after %d iterations" % (self.n_iter))

    def transform(self, X):
        return self.tf_session_.run(self.final_features_, feed_dict={self.X_: X})

    def transform_and_back(self, X):
        return self.tf_session_.run(self.X_reconstructed_, feed_dict={self.X_: X})

    def inverse_transform(self, X):
        return self.tf_session_.run(self.X_reconstructed_, feed_dict={self.final_features_: X})

    def _init_weights_and_bias(self, n_input):
        """Init weights and biases with the correct number of neurons.

        Examples:
        ---------
        >>> m = AutoEncoder(feature_layer_sizes=(5, 5))
        >>> m._init_weights_and_bias(n_input=10)
        >>> len(m.weights_encoder_)
        2
        >>> len(m.weights_decoder_)
        2
        """
        self.X_ = tf.placeholder("float", [None, n_input])

        self.weights_encoder_ = []
        self.bias_encoder_ = []
        self.weights_decoder_ = []
        self.bias_decoder_ = []

        # Encoder weights / bias
        self.weights_encoder_.append(tf.Variable(tf.random_normal([n_input, self.feature_layer_sizes_[0]])))
        for layer in range(1, self.n_feature_layers):
            self.weights_encoder_.append(tf.Variable(tf.random_normal([self.feature_layer_sizes_[layer - 1],
                                                                       self.feature_layer_sizes_[layer]])))
        self.bias_encoder_.append(tf.Variable(tf.random_normal([self.feature_layer_sizes_[0]])))
        for layer in range(1, self.n_feature_layers):
            self.bias_encoder_.append(tf.Variable(tf.random_normal([self.feature_layer_sizes_[layer]])))

        # Decoder weights / bias
        for layer in range(self.n_feature_layers - 1, 0, -1):
            self.weights_decoder_.append(tf.Variable(tf.random_normal([self.feature_layer_sizes_[layer],
                                                                       self.feature_layer_sizes_[layer - 1]])))
        self.weights_decoder_.append(tf.Variable(tf.random_normal([self.feature_layer_sizes_[0], n_input])))
        for layer in range(self.n_feature_layers - 1, 0, -1):
            self.bias_decoder_.append(tf.Variable(tf.random_normal([self.feature_layer_sizes_[layer - 1]])))
        self.bias_decoder_.append(tf.Variable(tf.random_normal([n_input])))

    def _build_encoder(self):
        self.encoder_layers_ = []

        linear_part = tf.add(tf.matmul(self.X_, self.weights_encoder_[0]), self.bias_encoder_[0])
        self.encoder_layers_.append(self.activation_fun(linear_part))

        for layer in range(1, self.n_feature_layers):
            linear_part = tf.add(tf.matmul(self.encoder_layers_[layer - 1], self.weights_encoder_[layer]),
                                 self.bias_encoder_[layer])
            self.encoder_layers_.append(self.activation_fun(linear_part))

    def _build_decoder(self):
        self.decoder_layers_ = []

        linear_part = tf.add(tf.matmul(self.final_features_, self.weights_decoder_[0]), self.bias_decoder_[0])
        self.decoder_layers_.append(self.activation_fun(linear_part))

        for layer in range(1, self.n_feature_layers):
            linear_part = tf.add(tf.matmul(self.decoder_layers_[layer - 1], self.weights_decoder_[layer]),
                                 self.bias_decoder_[layer])
            self.decoder_layers_.append(self.activation_fun(linear_part))

    def __str__(self):
        """Represent the NN as a string.

        Example:
        --------
        >>> m = AutoEncoder(feature_layer_sizes=(5, 6), n_iter=0)
        >>> m.fit(numpy.zeros((2, 10)))
        Optimization finished after 0 iterations
        >>> print(m)
        10 -> W:10x5, b:5 -> 5 -> W:5x6, b:6 -> 6 -> W:6x5, b:5 -> 5 -> W:5x10, b:10 -> 10
        """
        s = "%d -> " % shape(self.X_)[1]
        # Encoder
        for layer in range(self.n_feature_layers):
            w_i, w_o = shape(self.weights_encoder_[layer])
            b = shape(self.bias_encoder_[layer])[0]
            neurons = shape(self.encoder_layers_[layer])[1]
            s += "W:%dx%d, b:%d -> %d -> " % (w_i, w_o, b, neurons)
        # Decoder
        for layer in range(self.n_feature_layers):
            w_i, w_o = shape(self.weights_decoder_[layer])
            b = shape(self.bias_decoder_[layer])[0]
            neurons = shape(self.decoder_layers_[layer])[1]
            if layer < self.n_feature_layers - 1:
                s += "W:%dx%d, b:%d -> %d -> " % (w_i, w_o, b, neurons)
            else:
                s += "W:%dx%d, b:%d -> " % (w_i, w_o, b)
        s += "%d" % shape(self.X_reconstructed_)[1]
        return s

    def __del__(self):
        if self.tf_session_ is not None:
            print("Properly closing TensorFlow session.")
            self.tf_session_.close()
        else:
            print("No need to close any session, since nothing was fitted.")

def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])

if __name__ == "__main__":
    ae = AutoEncoder(feature_layer_sizes=(256, 128))
    print(ae.n_feature_layers)
    #ae.fit()
    del ae