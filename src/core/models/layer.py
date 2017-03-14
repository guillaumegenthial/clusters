import tensorflow as tf
from math import ceil
from model import Model
from core.utils.tf import xavier_weight_init, conv2d, \
        max_pool_2x2, weight_variable, bias_variable
import copy


class Layer(object):
    def __init__(self, name=None, input_names=[]):
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        self.input_names = input_names

    def set_param(self, **kwargs):
        for key, value in kwargs.iteritems():
            if key == "dropout" and self.__class__.__name__ == "Dropout":
                self.dropout = value
            elif key == "input_shape":
                self.input_shape = value

        self.update_param()

    def update_param(self):
        """
        Computes operations
        Could compute self.output_shape for inst
        """
        self.output_shape = self.input_shape

    def __call__(self, inputs, mask=None):
        """
        Takes a tensor and outputs a tensor
        Args:
            inputs: tensor of shape (None, ...)
            mask: a tensor for masking
        Returns:
            outpouts: tensorf of shape (None, ...)
        """
        raise NotImplementedError

class FullyConnected(Layer):
    def __init__(self, output_size, bias=True, name=None, input_names=[]):
        Layer.__init__(self, name, input_names)
        self.output_size = output_size
        self.bias = bias

    def __call__(self, inputs, mask=None):
        """
        Args:
            inputs: tensor of shape [, .., d]
        Return:
            tensor of shape [, ..., output_size] = inputs \cdot W
        """
        with tf.variable_scope(self.name):
            W = weight_variable('W', [self.input_shape[-1], self.output_size])
            
            if len(self.input_shape) == 3:
                inputs = tf.reshape(inputs, [-1, self.input_shape[-1]])
            
            result = tf.matmul(inputs, W)

            if len(self.input_shape) == 3:
                result = tf.reshape(result, [-1, self.input_shape[-2], self.output_size])

            if self.bias == True:
                b = bias_variable('bias', [self.output_size])
                result += b
            return result

    def update_param(self):
        self.output_shape = self.input_shape[:-1] + [self.output_size]

class Dropout(Layer):
    def __init__(self, name=None, input_names=[]):
        Layer.__init__(self, name, input_names)

    def __call__(self, inputs, mask=None):
        with tf.variable_scope(self.name):
            return tf.nn.dropout(inputs, self.dropout)


class ReLu(Layer):
    def __call__(self, inputs, mask=None):
        with tf.variable_scope(self.name):
            return tf.nn.relu(inputs)

class Flatten(Layer):
    def __call__(self, inputs, mask=None):
        with tf.variable_scope(self.name):
            return tf.reshape(inputs, [-1, self.shape_flat])

    def update_param(self):
        self.shape_flat = reduce(lambda x, y: int(x*y), self.input_shape[1:])
        self.output_shape = self.input_shape[:1] + [self.shape_flat]


class Conv2d(Layer):
    def __init__(self, filter_height, filter_width, 
                in_channels, out_channels, name=None, input_names=[]):

        Layer.__init__(self, name, input_names)
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.in_channels = in_channels
        self.out_channels = out_channels

        # default for now
        self.strides=[1, 1, 1, 1]
        self.padding='SAME'

    def __call__(self, inputs, mask=None):
        with tf.variable_scope(self.name):
            W = weight_variable('W', [self.filter_height, self.filter_width, 
                                      self.in_channels, self.out_channels])

            b = bias_variable('b', [self.out_channels])
        
            return tf.nn.conv2d(inputs, W, self.strides, self.padding) + b

    def update_param(self):
        in_height, in_width = self.input_shape[1:3]
        if self.padding == 'SAME':
            out_height = ceil(float(in_height) / float(self.strides[1]))
            out_width  = ceil(float(in_width) / float(self.strides[2]))

        elif self.padding == 'VALID':
            out_height = ceil(float(in_height - self.filter_height + 1) / float(self.strides[1]))
            out_width  = ceil(float(in_width - self.filter_width + 1) / float(self.strides[2]))

        self.output_shape = self.input_shape[:1] + [out_height, out_width] + [self.out_channels]

class MaxPool(Layer):
    def __init__(self, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                padding='SAME', name=None, input_names=[]):

        Layer.__init__(self, name, input_names)
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    def __call__(self, inputs, mask=None):
        with tf.variable_scope(self.name):  
            return tf.nn.max_pool(inputs, self.ksize, self.strides, self.padding)

    def update_param(self):
        in_height, in_width = self.input_shape[1:3]

        if self.padding == 'SAME':
            out_height = ceil(float(in_height) / float(self.strides[1]))
            out_width  = ceil(float(in_width) / float(self.strides[2]))

        elif self.padding == 'VALID':
            out_height = ceil(float(in_height - self.ksize[1] + 1) / float(self.strides[1]))
            out_width  = ceil(float(in_width - self.ksize[2] + 1) / float(self.strides[2]))

        self.output_shape = self.input_shape[:1] + [out_height, out_width] + self.input_shape[3:]


class Embedding(Layer):
    def __init__(self, vocab_size, embedding_size, name=None, input_names=[]):
        Layer.__init__(self, name, input_names)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

    def __call__(self, inputs, mask=None):
        with tf.variable_scope(self.name):
            L = weight_variable('L', [self.vocab_size, self.embedding_size])
            return tf.nn.embedding_lookup(L, inputs, name="embeddings")

    def update_param(self):
        self.output_shape = self.input_shape + [self.embedding_size]


class Reduce(Layer):
    def __init__(self, axis, op="max", keep_dims=False, name=None, input_names=[]):
        Layer.__init__(self, name, input_names)
        self.axis = axis
        self.op = op
        self.keep_dims = keep_dims

    def __call__(self, inputs, mask=None):
        with tf.variable_scope(self.name):
            if self.op == "max":
                return tf.reduce_max(inputs, axis=self.axis, keep_dims=self.keep_dims)
            elif self.op == "min":
                return tf.reduce_min(inputs, axis=self.axis, keep_dims=self.keep_dims)
            elif self.op == "mean":
                return tf.reduce_max(inputs, axis=self.axis, keep_dims=self.keep_dims)
            elif self.op == "sum":
                return tf.reduce_sum(inputs, axis=self.axis, keep_dims=self.keep_dims)

    def update_param(self):
        if self.keep_dims:
            self.output_shape = self.input_shape[:self.axis]+ [1] + self.input_shape[self.axis+1:]
        else:
            self.output_shape = self.input_shape[:self.axis]+ self.input_shape[self.axis+1:]


class Combine(Layer):
    def __call__(self, inputs, mask=None):
        """
        Args:
            inputs: list or tuple of length 2
                    (embedding, features)
        """
        with tf.variable_scope(self.name):
            # shape = (None, max_n_cells, embedding_size, 1)
            input0 = tf.expand_dims(inputs[0], axis=3)
            # shape = (None, max_n_cells, 1, n_features)
            input1 = tf.expand_dims(inputs[1], axis=2)
            # shape = (None, max_n_cells, embedding_size, n_features)
            return tf.matmul(input0, input1)

    def update_param(self):
        shape0 = self.input_shape[0]
        shape1 = self.input_shape[1]
        self.output_shape = shape0 + [shape1[-1]]

class Expand(Layer):
    def __init__(self, axis, name=None, input_names=[]):
        Layer.__init__(self, name, input_names)
        self.axis = axis

    def __call__(self, inputs, mask=None):
        """
        Args:
            inputs: shape [batch_size, embed_size]
        Return:
            tensor of shape [batch_size, 1, embed_size]
        """
        with tf.variable_scope(self.name):
            return tf.expand_dims(inputs, axis=1)

    def update_param(self):
        self.output_shape = self.input_shape[:self.axis] + [1] + self.input_shape[self.axis:]

class Squeeze(Layer):

    def __call__(self, inputs, mask=None):
        with tf.variable_scope(self.name):
            return tf.squeeze(inputs)

    def update_param(self):
        self.output_shape = [s for s in self.input_shape if s != 1]

class Concat(Layer):
    def __init__(self, axis, name=None, input_names=[]):
        Layer.__init__(self, name, input_names)
        self.axis = axis

    def __call__(self, inputs, mask=None):
        """
        Args:
            inputs: list of tensors
        Return:
            a tensor that concat inputs along the given axis
        """
        assert type(inputs) == list
        with tf.variable_scope(self.name):
            return tf.concat(inputs, axis=self.axis)

    def update_param(self):
        shape0 = self.input_shape[0]
        self.output_shape = copy.deepcopy(shape0)
        self.output_shape[self.axis] = sum(s[self.axis] for s in self.input_shape)

class LastConcat(Concat):
    """
    Custom Concat
    """
    def __call__(self, inputs, mask=None):
        """
        Args:
            inputs: list of tensors
        Return:
            a tensor that concat inputs on the last axis
        """
        assert type(inputs) == list
        with tf.variable_scope(self.name):
            s = self.input_shape[0]
            inputs[1] = tf.tile(inputs[1], multiples=[1, s[1], 1])
            return tf.concat(inputs, axis=self.axis)


    def update_param(self):
        shape0 = self.input_shape[0]
        self.output_shape = copy.deepcopy(shape0)
        self.output_shape[self.axis] = sum(s[self.axis] for s in self.input_shape)

class Mask(Layer):
    def __init__(self, val=0, name=None, input_names=[]):
        Layer.__init__(self, name, input_names)
        self.val = val


    def __call__(self, inputs, mask=None):
        mask = tf.reshape(mask, shape=[-1])
        inputs = tf.reshape(inputs, shape=[-1, self.input_shape[-1]])
        masked = tf.ones_like(inputs) * self.val
        result = tf.where(mask, inputs, masked)

        return tf.reshape(result, shape=[-1, self.input_shape[-2], self.input_shape[-1]])






