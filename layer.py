"""Basic network-building tools"""

from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
from .regularization import Regularization


class Layer(object):
    """Implementation of fully connected neural network layer

    Attributes:
        scope (str): name scope for variables and operations in layer
        input_dims (list): inputs to layer
        output_dims (list): outputs of layer
        outputs (tf Tensor): output of layer
        num_inh (int): number of inhibitory units in layer
        weights_ph (tf placeholder): placeholder for weights in layer
        biases_ph (tf placeholder): placeholder for biases in layer
        weights_var (tf Tensor): weights in layer
        biases_var (tf Tensor): biases in layer
        weights (numpy array): shadow variable of `weights_var` that allows for 
            easier manipulation outside of tf sessions
        biases (numpy array): shadow variable of `biases_var` that allows for 
            easier manipulation outside of tf sessions
        activation_func (tf activation function): activation function in layer
        reg (Regularization object): holds regularizations values and matrices
            (as tf constants) for layer
        ei_mask_var (tf constant): mask of +/-1s to multiply output of layer
        ei_mask (list): mask of +/-1s to multiply output of layer; shadows 
            `ei_mask_tf` for easier manipulation outside of tf sessions
        pos_constraint (bool): positivity constraint on weights in layer
        log (bool): use tf summary writers on layer output

    """

    def __init__(
            self,
            scope=None,
            input_dims=None,  # this can be a list up to 3-dimensions
            filter_dims=None,
            output_dims=None,
            activation_func='relu',
            weights_initializer='trunc_normal',
            biases_initializer='zeros',
            reg_initializer=None,
            num_inh=0,
            pos_constraint=False,
            log_activations=False):
        """Constructor for Layer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int or list): dimensions of input data
            output_dims (int or list): dimensions of output data
            activation_func (str, optional): pointwise function applied to  
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' | 
                'elu' | 'quad'
            weights_initializer (str, optional): initializer for the weights
                ['trunc_normal'] | 'normal' | 'zeros'
            biases_initializer (str, optional): initializer for the biases
                'trunc_normal' | 'normal' | ['zeros']
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (bool, optional): True to constrain layer weights to 
                be positive
            log_activations (bool, optional): True to use tf.summary on layer 
                activations

        Raises:
            TypeError: If `variable_scope` is not specified
            TypeError: If `inputs` is not specified
            TypeError: If `num_inputs` or `num_outputs` is not specified
            ValueError: If `num_inh` is greater than total number of units
            ValueError: If `activation_func` is not a valid string
            ValueError: If `weights_initializer` is not a valid string
            ValueError: If `biases_initializer` is not a valid string

        """

        # check for required inputs
        if scope is None:
            raise TypeError('Must specify layer scope')
        if input_dims is None or output_dims is None:
            raise TypeError('Must specify both input and output dimensions')

        self.scope = scope

        # Make input, output, and filter sizes explicit
        if isinstance(input_dims,list):
            while len(input_dims) < 3:
                input_dims.append(1)
        else:
            input_dims = [input_dims,1,1]
        if isinstance(output_dims,list):
            while len(output_dims) < 3:
                output_dims.append(1)
            num_outputs = output_dims[0]*output_dims[1]*output_dims[2]
        else:
            num_outputs = output_dims
            output_dims = [output_dims,1,1]

        self.input_dims = input_dims
        self.output_dims = output_dims
        # default to have N filts for N outputs in base layer class
        self.num_filters = num_outputs
        if filter_dims is None:
            filter_dims = input_dims
        num_inputs = filter_dims[0] * filter_dims[1] * filter_dims[2]
        self.filter_dims = filter_dims

        # resolve activation function string
        if activation_func == 'relu':
            self.activation_func = tf.nn.relu
        elif activation_func == 'sigmoid':
            self.activation_func = tf.sigmoid
        elif activation_func == 'tanh':
            self.activation_func = tf.tanh
        elif activation_func == 'lin':
            self.activation_func = tf.identity
        elif activation_func == 'linear':
            self.activation_func = tf.identity
        elif activation_func == 'softplus':
            self.activation_func = tf.nn.softplus
        elif activation_func == 'quad':
            self.activation_func = tf.square
        elif activation_func == 'elu':
            self.activation_func = tf.nn.elu
        elif activation_func == 'exp':
            self.activation_func = tf.exp
        else:
            raise ValueError('Invalid activation function ''%s''' %
                             activation_func)

        # create excitatory/inhibitory mask
        if num_inh > num_outputs:
            raise ValueError('Too many inhibitory units designated')
        self.ei_mask = [1] * (num_outputs - num_inh) + [-1] * num_inh

        # save positivity constraint on weights
        self.pos_constraint = pos_constraint

        # use tf's summary writer to save layer activation histograms
        if log_activations:
            self.log = True
        else:
            self.log = False

        # Set up layer regularization
        self.reg = Regularization(
            input_dims=filter_dims,
            num_outputs=num_outputs,
            vals=reg_initializer)

        # Initialize weight values
        weight_dims = (num_inputs, num_outputs)
        if weights_initializer == 'trunc_normal':
            init_weights = np.random.normal(size=weight_dims, scale=0.1)
        elif weights_initializer == 'normal':
            init_weights = np.random.normal(size=weight_dims, scale=0.1)
        elif weights_initializer == 'zeros':
            init_weights = np.zeros(shape=weight_dims, dtype='float32')
        else:
            raise ValueError('Invalid weights_initializer ''%s''' %
                             weights_initializer)
        if pos_constraint is True:
            init_weights = np.maximum(init_weights, 0)
        # Initialize numpy array that will feed placeholder
        self.weights = init_weights.astype('float32')

        # Initialize bias values
        bias_dims = (1, num_outputs)
        if biases_initializer == 'trunc_normal':
            init_biases = np.random.normal(size=bias_dims, scale=0.1)
        elif biases_initializer == 'normal':
            init_biases = np.random.normal(size=bias_dims, scale=0.1)
        elif biases_initializer == 'zeros':
            init_biases = np.zeros(shape=bias_dims, dtype='float32')
        else:
            raise ValueError('Invalid biases_initializer ''%s''' %
                             biases_initializer)
        # Initialize numpy array that will feed placeholder
        self.biases = init_biases.astype('float32')

        # Define tensorflow variables as placeholders
        self.weights_ph = None
        self.weights_var = None
        self.biases_ph = None
        self.biases_var = None
        self.outputs = None

    # END Layer.__init__

    def _define_layer_variables(self):
        # Define tensor-flow versions of variables (placeholder and variables)
        with tf.name_scope('weights_init'):
            self.weights_ph = tf.placeholder_with_default(
                self.weights,
                shape=self.weights.shape,
                name='weights_ph')
            self.weights_var = tf.Variable(
                self.weights_ph,
                dtype=tf.float32,
                name='weights_var')

        # Initialize biases placeholder/variable
        with tf.name_scope('biases_init'):
            self.biases_ph = tf.placeholder_with_default(
                self.biases,
                shape=self.biases.shape,
                name='biases_ph')
            self.biases_var = tf.Variable(
                self.biases_ph,
                dtype=tf.float32,
                name='biases_var')

        # Check for need of ei_mask
        if np.sum(self.ei_mask) < len(self.ei_mask):
            self.ei_mask_var = tf.constant(
                self.ei_mask, dtype=tf.float32, name='ei_mask')
        else:
            self.ei_mask_var = None

    # END Layer._define_layer_variables

    def build_graph(self, inputs, params_dict=None):

        with tf.name_scope(self.scope):
            self._define_layer_variables()

            # Define computation
            if self.pos_constraint:
                pre = tf.add(tf.matmul(
                    inputs,
                    tf.maximum(0.0, self.weights_var)), self.biases_var)
            else:
                pre = tf.add(
                    tf.matmul(inputs, self.weights_var), self.biases_var)

            if self.ei_mask_var is not None:
                post = tf.multiply(self.activation_func(pre), self.ei_mask_var)
            else:
                post = self.activation_func(pre)

            self.outputs = post

        if self.log:
            tf.summary.histogram('act_pre', pre)
            tf.summary.histogram('act_post', post)
    # END layer._define_network

    def assign_layer_params(self, sess):
        """Read weights/biases in numpy arrays into tf Variables"""
        sess.run(
            [self.weights_var.initializer, self.biases_var.initializer],
            feed_dict={self.weights_ph: self.weights,
                       self.biases_ph: self.biases})

    def write_layer_params(self, sess):
        """Write weights/biases in tf Variables to numpy arrays"""
        self.weights = sess.run(self.weights_var)
        if self.pos_constraint is True:
            self.weights = np.maximum(self.weights, 0)
        self.biases = sess.run(self.biases_var)

    def define_regularization_loss(self):
        """Wrapper function for building regularization portion of graph"""
        with tf.name_scope(self.scope):
            return self.reg.define_reg_loss(self.weights_var)

    def set_regularization(self, reg_type, reg_val):
        """Wrapper function for setting regularization"""
        return self.reg.set_reg_val(reg_type, reg_val)

    def assign_reg_vals(self, sess):
        """Wrapper function for assigning regularization values"""
        self.reg.assign_reg_vals(sess)

    def get_reg_pen(self, sess):
        """Wrapper function for returning regularization penalty struct"""
        return self.reg.get_reg_penalty(sess)
