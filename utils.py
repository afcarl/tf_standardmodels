import tensorflow as tf

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def tf_shape(tensor):
    """Return the shape of a tensor as a tuple

    Parameters
    ----------
    tensor: tf.Tensor

    Returns
    -------
    tuple
        A numpy-style shape tuple
    """
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])
