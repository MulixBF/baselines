from __future__ import division
from __future__ import absolute_import
import os

import tensorflow as tf

# ================================================================
# Saving variables
# ================================================================

def load_state(fname):
    saver = tf.train.Saver()
    saver.restore(tf.get_default_session(), fname)

def save_state(fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    saver = tf.train.Saver()
    saver.save(tf.get_default_session(), fname)

# ================================================================
# Placeholders
# ================================================================

class TfInput(object):
    def __init__(self, name=u"(unnamed)"):
        u"""Generalized Tensorflow placeholder. The main differences are:
            - possibly uses multiple placeholders internally and returns multiple values
            - can apply light postprocessing to the value feed to placeholder.
        """
        self.name = name

    def get(self):
        u"""Return the tf variable(s) representing the possibly postprocessed value
        of placeholder(s).
        """
        raise NotImplemented()

    def make_feed_dict(data):
        u"""Given data input it to the placeholder(s)."""
        raise NotImplemented()


class PlaceholderTfInput(TfInput):
    def __init__(self, placeholder):
        u"""Wrapper for regular tensorflow placeholder."""
        super(PlaceholderTfInput, self).__init__(placeholder.name)
        self._placeholder = placeholder

    def get(self):
        return self._placeholder

    def make_feed_dict(self, data):
        return {self._placeholder: data}

class BatchInput(PlaceholderTfInput):
    def __init__(self, shape, dtype=tf.float32, name=None):
        u"""Creates a placeholder for a batch of tensors of a given shape and dtype

        Parameters
        ----------
        shape: [int]
            shape of a single elemenet of the batch
        dtype: tf.dtype
            number representation used for tensor contents
        name: str
            name of the underlying placeholder
        """
        super(BatchInput, self).__init__(tf.placeholder(dtype, [None] + list(shape), name=name))

class Uint8Input(PlaceholderTfInput):
    def __init__(self, shape, name=None):
        u"""Takes input in uint8 format which is cast to float32 and divided by 255
        before passing it to the model.

        On GPU this ensures lower data transfer times.

        Parameters
        ----------
        shape: [int]
            shape of the tensor.
        name: str
            name of the underlying placeholder
        """

        super(Uint8Input, self).__init__(tf.placeholder(tf.uint8, [None] + list(shape), name=name))
        self._shape = shape
        self._output = tf.cast(super(Uint8Input, self).get(), tf.float32) / 255.0

    def get(self):
        return self._output