from sklearn.base import TransformerMixin
from sklearn.utils import check_array
import tensorflow as tf
import numpy

from utils import tf_shape

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class AutoEncoder(TransformerMixin):
    """Auto-Encoder model for feature extraction.

    Implement a (possibly multi-layer) auto-encoder to perform feature extraction using TensorFlow.

    Attributes
    ----------
    feature_layer_sizes: tuple of ints, default (100, )
        Number of neurons in each layer of the auto-encoder.
        Decoder layers should not be included in the tuple since their size is inferred from those of the encoder.
    activation: {'relu', 'logistic'}, default 'relu'
        Activation function for the hidden layers
        * 'relu', the rectified linear unit function, returns :math:`f(x) = max(0, x)`
        * 'logistic', the logistic sigmoid function, returns :math:`f(x) = 1 / (1 + exp(-x))`.
    learning_rate: float, default: 0.01
        The learning rate used. It controls the step-size in updating the weights.
    n_iter: int, default: 20
        The number of iterations in the learning process.
    batch_size: int, default: 256
        The number of training samples in each batch of the batch stochatic gradient descent.
    radom_seed: int, default: None
        The seed used to initialize TensorFlow at each fit call.
        If None, no seed is forced at fit time.


    References
    ----------
        Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
        learning applied to document recognition." Proceedings of the IEEE,
        86(11):2278-2324, November 1998.

    Links
    -----
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

        self.encoder_weights_ = []
        self.decoder_weights_ = []
        self.encoder_biases_ = []
        self.decoder_biases_ = []

        self.X_ = None
        self.cost_ = None
        self.optimizer_ = None

    def fit(self, X):
        if self.random_seed is not None:
            tf.set_random_seed(self.random_seed)
            numpy.random.seed(self.random_seed)
        X = check_array(X, ensure_2d=True, copy=True)
        n_input = X.shape[1]
        n_batches = int(X.shape[0] / self.batch_size)
        input_X, encoder, decoder = self._init_weights_and_biases(n_input)
        tr_X = self._build_encoder(encoder, input_X)
        rec_X = self._build_decoder(decoder, tr_X)
        self.cost_ = tf.reduce_mean(tf.pow(input_X - rec_X, 2))
        self.optimizer_ = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost_)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        for iteration in range(self.n_iter):
            for batch in range(n_batches):
                indices_batch = numpy.random.choice(X.shape[0], size=self.batch_size)
                sess.run(self.optimizer_, feed_dict={input_X: X[indices_batch]})
            c = sess.run(self.cost_, feed_dict={input_X: X})
            print("Epoch: %04d, cost=%.9f" % (iteration + 1, c))
        print("Optimization finished after %d iterations" % self.n_iter)
        self._save_weights_biases(encoder, decoder, sess)
        sess.close()
        return self

    def _save_weights_biases(self, encoder, decoder, sess):
        for enc, dec in zip(encoder, decoder):
            self.encoder_weights_.append(enc["weights"].eval(session=sess))
            self.encoder_biases_.append(enc["biases"].eval(session=sess))
            self.decoder_weights_.append(dec["weights"].eval(session=sess))
            self.decoder_biases_.append(dec["biases"].eval(session=sess))

    def transform(self, X):
        X = check_array(X, ensure_2d=True, copy=True)
        tf.reset_default_graph()
        sess = tf.Session()
        X_cur = tf.constant(X, dtype=tf.float32)
        for layer in range(self.n_feature_layers):
            weight = tf.constant(self.encoder_weights_[layer], dtype=tf.float32)
            bias = tf.constant(self.encoder_biases_[layer], dtype=tf.float32)
            layer = tf.matmul(X_cur, weight) + bias
            X_cur = self.activation_fun(layer)
        return X_cur.eval(session=sess)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)

    def transform_and_back(self, X):
        return self.inverse_transform(self.transform(X))

    def inverse_transform(self, X):
        X = check_array(X, ensure_2d=True, copy=True)
        tf.reset_default_graph()
        sess = tf.Session()
        X_cur = tf.constant(X, dtype=tf.float32)
        for layer in range(self.n_feature_layers):
            weight = tf.constant(self.decoder_weights_[layer], dtype=tf.float32)
            bias = tf.constant(self.decoder_biases_[layer], dtype=tf.float32)
            layer = tf.matmul(X_cur, weight) + bias
            X_cur = self.activation_fun(layer)
        return X_cur.eval(session=sess)

    def _init_weights_and_biases(self, n_input):
        """Init weights and biases with the correct number of neurons."""
        input_X = tf.placeholder("float", [None, n_input])

        weights_encoder = []
        biases_encoder = []
        weights_decoder = []
        biases_decoder = []

        # Encoder weights / bias
        weights_encoder.append(tf.Variable(tf.random_normal([n_input, self.feature_layer_sizes_[0]])))
        for layer in range(1, self.n_feature_layers):
            weights_encoder.append(tf.Variable(tf.random_normal([self.feature_layer_sizes_[layer - 1],
                                                                 self.feature_layer_sizes_[layer]])))
        biases_encoder.append(tf.Variable(tf.random_normal([self.feature_layer_sizes_[0]])))
        for layer in range(1, self.n_feature_layers):
            biases_encoder.append(tf.Variable(tf.random_normal([self.feature_layer_sizes_[layer]])))

        # Decoder weights / bias
        for layer in range(self.n_feature_layers - 1, 0, -1):
            weights_decoder.append(tf.Variable(tf.random_normal([self.feature_layer_sizes_[layer],
                                                                 self.feature_layer_sizes_[layer - 1]])))
        weights_decoder.append(tf.Variable(tf.random_normal([self.feature_layer_sizes_[0], n_input])))
        for layer in range(self.n_feature_layers - 1, 0, -1):
            biases_decoder.append(tf.Variable(tf.random_normal([self.feature_layer_sizes_[layer - 1]])))
        biases_decoder.append(tf.Variable(tf.random_normal([n_input])))

        encoder = [{"weights": weights_encoder[layer], "biases": biases_encoder[layer]}
                   for layer in range(self.n_feature_layers)]
        decoder = [{"weights": weights_decoder[layer], "biases": biases_decoder[layer]}
                   for layer in range(self.n_feature_layers)]

        return input_X, encoder, decoder

    def _build_encoder(self, encoder, input_X):
        encoder_layers_ = []

        linear_part = tf.add(tf.matmul(input_X, encoder[0]["weights"]), encoder[0]["biases"])
        encoder_layers_.append(self.activation_fun(linear_part))

        for layer in range(1, self.n_feature_layers):
            linear_part = tf.add(tf.matmul(encoder_layers_[layer - 1], encoder[layer]["weights"]),
                                 encoder[layer]["biases"])
            encoder_layers_.append(self.activation_fun(linear_part))
        return encoder_layers_[-1]

    def _build_decoder(self, decoder, tr_X):
        decoder_layers_ = []

        linear_part = tf.add(tf.matmul(tr_X, decoder[0]["weights"]), decoder[0]["biases"])
        decoder_layers_.append(self.activation_fun(linear_part))

        for layer in range(1, self.n_feature_layers):
            linear_part = tf.add(tf.matmul(decoder_layers_[layer - 1], decoder[layer]["weights"]),
                                 decoder[layer]["biases"])
            decoder_layers_.append(self.activation_fun(linear_part))
        return decoder_layers_[-1]

    def __str__(self):
        """Represent the NN as a string.

        Example:
        --------
        >>> m = AutoEncoder(feature_layer_sizes=(5, 6), n_iter=0)
        >>> m.fit(numpy.zeros((2, 10)))
        Optimization finished after 0 iterations
        >>> print(m)
        W:10x5, b:5 -> W:5x6, b:6 -> W:6x5, b:5 -> W:5x10, b:10
        """
        s = ""
        # Encoder
        for layer in range(self.n_feature_layers):
            w_i, w_o = self.encoder_weights_[layer].shape
            b = self.encoder_biases_[layer].shape[0]
            s += "W:%dx%d, b:%d -> " % (w_i, w_o, b)
        # Decoder
        for layer in range(self.n_feature_layers):
            w_i, w_o = self.decoder_weights_[layer].shape
            b = self.decoder_biases_[layer].shape[0]
            if layer < self.n_feature_layers - 1:
                s += "W:%dx%d, b:%d -> " % (w_i, w_o, b)
            else:
                s += "W:%dx%d, b:%d" % (w_i, w_o, b)
        return s


class StackedAutoEncoder(AutoEncoder):
    """Stacked Auto-Encoder model for feature extraction.

    Implement a stacked auto-encoder model to perform feature extraction using TensorFlow.
    In this model, feature layers are learned greedily.

    Attributes
    ----------
    feature_layer_sizes: tuple of ints, default (100, )
        Number of neurons in each layer of the auto-encoder.
        Decoder layers should not be included in the tuple since their size is inferred from those of the encoder.
    activation: {'relu', 'logistic'}, default 'relu'
        Activation function for the hidden layers
        * 'relu', the rectified linear unit function, returns :math:`f(x) = max(0, x)`
        * 'logistic', the logistic sigmoid function, returns :math:`f(x) = 1 / (1 + exp(-x))`.
    learning_rate: float, default: 0.01
        The learning rate used. It controls the step-size in updating the weights.
    n_iter: int, default: 20
        The number of iterations in the learning process.
    batch_size: int, default: 256
        The number of training samples in each batch of the batch stochatic gradient descent.
    radom_seed: int, default: None
        The seed used to initialize TensorFlow at each fit call.
        If None, no seed is forced at fit time.


    References
    ----------
        Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
        learning applied to document recognition." Proceedings of the IEEE,
        86(11):2278-2324, November 1998.

    Links
    -----
        [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
    """
    def fit(self, X):
        if self.random_seed is not None:
            tf.set_random_seed(self.random_seed)
            numpy.random.seed(self.random_seed)
        X = check_array(X, ensure_2d=True, copy=True)
        n_batches = int(X.shape[0] / self.batch_size)

        X_cur = X
        for layer in range(self.n_feature_layers):
            n_input = X_cur.shape[1]
            n_output = self.feature_layer_sizes_[layer]
            sess, cost, opt, input_X, encoded, decoded, encoder, decoder = self.prepare_session_for_layer(n_input,
                                                                                                          n_output)
            for iteration in range(self.n_iter):
                for batch in range(n_batches):
                    indices_batch = numpy.random.choice(X_cur.shape[0], size=self.batch_size)
                    sess.run(opt, feed_dict={input_X: X_cur[indices_batch]})
                c = sess.run(cost, feed_dict={input_X: X_cur})
                print("Layer %d, Epoch: %04d, cost=%.9f" % (layer + 1, iteration + 1, c))
            X_cur = sess.run(encoded, feed_dict={input_X: X_cur})
            self._save_weights_biases([encoder], [decoder], sess)
            sess.close()

        print("Optimization finished after %d iterations per layer" % self.n_iter)

        return self

    def prepare_session_for_layer(self, n_input, n_output):
        tf.reset_default_graph()

        weights_encoder = tf.Variable(tf.random_normal([n_input, n_output]))
        biases_encoder = tf.Variable(tf.random_normal([n_output]))
        weights_decoder = tf.Variable(tf.random_normal([n_output, n_input]))
        biases_decoder = tf.Variable(tf.random_normal([n_input]))

        encoder_wb = {"weights": weights_encoder, "biases": biases_encoder}
        decoder_wb = {"weights": weights_decoder, "biases": biases_decoder}

        input_X = tf.placeholder("float", [None, n_input])
        linear_part_enc = tf.add(tf.matmul(input_X, encoder_wb["weights"]), encoder_wb["biases"])
        encoded = self.activation_fun(linear_part_enc)
        linear_part_dec = tf.add(tf.matmul(encoded, decoder_wb["weights"]), decoder_wb["biases"])
        decoded = self.activation_fun(linear_part_dec)

        cost = tf.reduce_mean(tf.pow(input_X - decoded, 2))
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(cost)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        return sess, cost, optimizer, input_X, encoded, decoded, encoder_wb, decoder_wb

    def _save_weights_biases(self, encoder, decoder, sess):
        for enc, dec in zip(encoder, decoder):
            self.encoder_weights_.append(enc["weights"].eval(session=sess))
            self.encoder_biases_.append(enc["biases"].eval(session=sess))
            self.decoder_weights_.insert(0, dec["weights"].eval(session=sess))
            self.decoder_biases_.insert(0, dec["biases"].eval(session=sess))
