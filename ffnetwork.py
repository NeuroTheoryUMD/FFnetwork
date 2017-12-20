"""Basic network-building tools"""

from __future__ import print_function
from __future__ import division

import tensorflow as tf
from .layer import Layer


class FFNetwork(object):
    """Implementation of simple fully connected feedforward neural network

    Attributes:
        scope (str): name scope for network
        layers (list of `Layer` objects): layers of network
        num_layers (int): number of layers in network (not including input)
        log (bool): use tf summary writers in layer activations

    """

    def __init__(self,
                 scope=None,
                 params_dict=None):
        """Constructor for FFNetwork class

        Args:
            scope (str): name scope for network
            params_dict (dict): contains parameters about details of FFnetwork:
            -> layer_sizes (list of ints): list of layer sizes, including input 
                and output. All arguments (input size) can be up to a 
                3-dimensional list. REQUIRED (NO DEFAULT)
            -> num_inh: list or single number denoting number of inhibitory units in each
                layer. This specifies the output of that number of units multiplied by -1
                DEFAULT = 0 (and having any single value will be used for all layers)
            -> activation_funcs (str or list of strs, optional): pointwise
                function for each layer; replicated if a single element. 
                DEFAULT = 'relu'. See Layer class for other options.
            -> pos_constraints (bool or list of bools, optional): constrains all weights to be positive
                DEFAULTS = False.
            -> reg_initializer (dict): a list of dictionaries: one for each layer. Within the
                dictionary, reg_type/vals as key-value pairs.
                DEFAULT = None
            -> weights_initializer (str or list of strs, optional): initializer
                for the weights in each layer; replicated if a single element.
                DEFAULT = 'trunc_normal'. See Layer class for other options.
            -> biases_initializer (str or list of strs, optional): initializer for
                the biases in each layer; replicated if a single element.
                DEFAULT = 'zeros'. See Layer class for other options.
            -> log_activations (bool, optional): True to use tf.summary on layer activations
                DEFAULT = False

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
        if params_dict is None:
            raise TypeError('Must specify parameters dictionary.')

        self.scope = scope

        # Check information in params_dict and set defaults
        if 'layer_sizes' not in params_dict:
            params_dict['layer_sizes'] = None

        self.num_layers = len(params_dict['layer_sizes']) - 1

        if 'activation_funcs' not in params_dict:
            params_dict['activation_funcs'] = 'relu'
        if type(params_dict['activation_funcs']) is not list:
            params_dict['activation_funcs'] = \
                [params_dict['activation_funcs']] * self.num_layers
        elif len(params_dict['activation_funcs']) != self.num_layers:
            raise ValueError('Invalid number of activation_funcs')

        if 'weights_initializers' not in params_dict:
            params_dict['weights_initializers'] = 'trunc_normal'
        if type(params_dict['weights_initializers']) is not list:
            params_dict['weights_initializers'] = \
                [params_dict['weights_initializers']] * self.num_layers
        elif len(params_dict['weights_initializers']) != self.num_layers:
            raise ValueError('Invalid number of weights_initializer')

        if 'biases_initializers' not in params_dict:
            params_dict['biases_initializers'] = 'zeros'
        if type(params_dict['biases_initializers']) is not list:
            params_dict['biases_initializers'] = \
                [params_dict['biases_initializers']] * self.num_layers
        elif len(params_dict['biases_initializers']) != self.num_layers:
            raise ValueError('Invalid number of biases_initializer')

        if 'num_inh' not in params_dict:
            params_dict['num_inh'] = 0
        if type(params_dict['num_inh']) is not list:
            params_dict['num_inh'] = [params_dict['num_inh']] * self.num_layers
        elif len(params_dict['num_inh']) != self.num_layers:
            raise ValueError('Invalid number of num_inh')

        if 'pos_constraints' not in params_dict:
            params_dict['pos_constraints'] = False
        if type(params_dict['pos_constraints']) is not list:
            params_dict['pos_constraints'] = \
                [params_dict['pos_constraints']] * self.num_layers
        elif len(params_dict['pos_constraints']) != self.num_layers:
            raise ValueError('Invalid number of pos_con')

        if 'log_activations' not in params_dict:
            params_dict['log_activations'] = False

        if 'reg_initializers' not in params_dict:
            params_dict['reg_initializers'] = [None]*self.num_layers

        # Define network
        with tf.name_scope(self.scope):
            self._define_network(params_dict)

        if params_dict['log_activations']:
            self.log = True
        else:
            self.log = False
    # END FFNetwork.__init__

    def _define_network(self, network_params):

        layer_sizes = network_params['layer_sizes']

        self.layers = []
        for layer in range(self.num_layers):
            self.layers.append(Layer(
                scope='layer_%i' % layer,
                input_dims=layer_sizes[layer],
                output_dims=layer_sizes[layer + 1],
                activation_func=network_params['activation_funcs'][layer],
                weights_initializer=network_params['weights_initializers'][layer],
                biases_initializer=network_params['biases_initializers'][layer],
                reg_initializer=network_params['reg_initializers'][layer],
                num_inh=network_params['num_inh'][layer],
                pos_constraint=network_params['pos_constraints'][layer],
                log_activations=network_params['log_activations']))
    # END FFNetwork._define_network

    def _build_fit_variable_list(self, fit_parameter_list):
        """makes a list of variables of this network that will be fit given argument"""

        var_list = []
        for layer in range(self.num_layers):
            if fit_parameter_list[layer]["weights"]:
                var_list.append( self.layers[layer].weights_var )
            if fit_parameter_list[layer]["biases"]:
                var_list.append( self.layers[layer].biases_var )
        return var_list
    # END FFNetwork._build_fit_variable_list

    def build_graph(self, inputs, params_dict=None):

        with tf.name_scope(self.scope):
            for layer in range(self.num_layers):
                self.layers[layer].build_graph(inputs, params_dict)
                inputs = self.layers[layer].outputs
    # END FFNetwork._build_graph

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
