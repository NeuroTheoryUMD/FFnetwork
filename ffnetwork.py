"""Basic network-building tools"""

from __future__ import print_function
from __future__ import division

import tensorflow as tf
from .layer import Layer


class FFNetwork(object):
    """Implementation of simple fully connected feedforward neural network

    Attributes:
        scope (str): name scope for network
        layers (list of Layer objects): layers of network
        num_layers (int): number of layers in network (not including input)
        log (bool): use tf summary writers in layer activations

    """

    def __init__(
            self,
            scope=None,
            inputs=None,
            layer_sizes=None,
            activation_funcs='relu',
            weights_initializer='trunc_normal',
            biases_initializer='zeros',
            reg_initializer=None,
            num_inh=0,
            pos_constraint=False,
            log_activations=False):
        """Constructor for Network class

        Args:
            scope (str): name scope for network
            inputs (tf Tensor or placeholder): input to network
            layer_sizes (list of ints): list of layer sizes, including input
                and output
            activation_funcs (str or list of strs, optional): pointwise 
                function for each layer; replicated if a single element. 
                See Layer class for options.
            weights_initializer (str or list of strs, optional): initializer  
                for the weights in each layer; replicated if a single element.
                See Layer class for options.
            biases_initializer (str or list of strs, optional): initializer for
                the biases in each layer; replicated if a single element.
                See Layer class for options.
            reg_initializer (dict): reg_type/vals as key-value pairs to 
                uniformly initialize layer regularization
            num_inh (None, int or list of ints, optional)
            pos_constraint (bool or list of bools, optional):
            log_activations (bool, optional): True to use tf.summary on layer 
                activations

        Raises:
            TypeError: If `scope` is not specified
            TypeError: If `inputs` is not specified
            TypeError: If `layer_sizes` is not specified
            ValueError: If `activation_funcs` is not a properly-sized list
            ValueError: If `weights_initializer` is not a properly-sized list
            ValueError: If `biases_initializer` is not a properly-sized list

        """

        # check for required inputs
        if scope is None:
            raise TypeError('Must specify network scope')
        if inputs is None:
            raise TypeError('Must specify network input')
        if layer_sizes is None:
            raise TypeError('Must specify layer sizes')

        self.scope = scope
        self.num_layers = len(layer_sizes) - 1

        # expand layer options
        if type(activation_funcs) is not list:
            activation_funcs = [activation_funcs] * self.num_layers
        elif len(activation_funcs) != self.num_layers:
            raise ValueError('Invalid number of activation_funcs')

        if type(weights_initializer) is not list:
            weights_initializer = [weights_initializer] * self.num_layers
        elif len(weights_initializer) != self.num_layers:
            raise ValueError('Invalid number of weights_initializer')

        if type(biases_initializer) is not list:
            biases_initializer = [biases_initializer] * self.num_layers
        elif len(biases_initializer) != self.num_layers:
            raise ValueError('Invalid number of biases_initializer')

        if type(num_inh) is not list:
            num_inh = [num_inh] * self.num_layers
        elif len(num_inh) != self.num_layers:
            raise ValueError('Invalid number of num_inh')

        if type(pos_constraint) is not list:
            pos_constraint = [pos_constraint] * self.num_layers
        elif len(pos_constraint) != self.num_layers:
            raise ValueError('Invalid number of pos_con')

        self.layers = []
        with tf.name_scope(self.scope):
            for layer in range(self.num_layers):
                self.layers.append(Layer(
                    scope='layer_%i' % layer,
                    inputs=inputs,
                    num_inputs=layer_sizes[layer],
                    num_outputs=layer_sizes[layer + 1],
                    activation_func=activation_funcs[layer],
                    weights_initializer=weights_initializer[layer],
                    biases_initializer=biases_initializer[layer],
                    reg_initializer=reg_initializer,
                    num_inh=num_inh[layer],
                    pos_constraint=pos_constraint[layer],
                    log_activations=log_activations))
                inputs = self.layers[-1].outputs

        if log_activations:
            self.log = True
        else:
            self.log = False
    # END __init__

    def assign_model_params(self, sess):
        """Read weights/biases in numpy arrays into tf Variables"""
        with tf.name_scope(self.scope):
            for layer in range(self.num_layers):
                self.layers[layer].assign_layer_params(sess)

    def write_model_params(self, sess):
        """Write weights/biases in tf Variables to numpy arrays"""
        for layer in range(self.num_layers):
            self.layers[layer].write_layer_params(sess)

    def assign_reg_vals(self, sess):
        """Update default tf Graph with new regularization penalties"""
        with tf.name_scope(self.scope):
            for layer in range(self.num_layers):
                self.layers[layer].assign_reg_vals(sess)

    def define_regularization_loss(self):
        """Build regularization loss portion of default tf graph"""
        with tf.name_scope(self.scope):
            # define regularization loss for each layer separately...
            reg_ops = [None for _ in range(self.num_layers)]
            for layer in range(self.num_layers):
                reg_ops[layer] = \
                    self.layers[layer].define_regularization_loss()
            # ...then sum over all layers
            reg_loss = tf.add_n(reg_ops)
        return reg_loss
