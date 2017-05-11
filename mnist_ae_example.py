from sklearn.datasets import fetch_mldata
import pylab
import numpy

from models import AutoEncoder

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


print("Fetching MNIST data")
mnist = fetch_mldata("MNIST original")
X, y = mnist.data / 255., mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

examples_to_show = 10

ae = AutoEncoder(feature_layer_sizes=(256, 128), random_seed=0)
ae.fit(X_train)

X_test_reconstruct = ae.inverse_transform(ae.transform(X_test))
f, a = pylab.subplots(2, examples_to_show, figsize=(examples_to_show, 2))
for i in range(examples_to_show):
    a[0][i].imshow(numpy.reshape(X_test[i], (28, 28)))
    a[1][i].imshow(numpy.reshape(X_test_reconstruct[i], (28, 28)))
pylab.show()
del ae
