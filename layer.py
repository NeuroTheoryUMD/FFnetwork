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
        num_inputs (int): number of inputs to layer
        num_outputs (int): number of outputs of layer
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
            inputs=None,
            num_inputs=None,
            num_outputs=None,
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
            inputs (tf Tensor or placeholder): input to layer
            num_inputs (int): dimension of input data
            num_outputs (int): dimension of output data
            activation_func (str, optional): pointwise function applied to  
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' | 'elu' | 'quad'
            weights_initializer (str, optional): initializer for the weights
                ['trunc_normal'] | 'normal' | 'zeros'
            biases_initializer (str, optional): initializer for the biases
                'trunc_normal' | 'normal' | ['zeros']
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (bool, optional): True to constrain layer weights to be 
                positive
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
        if inputs is None:
            raise TypeError('Must specify layer input')
        if num_inputs is None or num_outputs is None:
            raise TypeError('Must specify both input and output dimensions')

        self.scope = scope

        # make layer size explicit
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        # resolve activation function string
        if activation_func == 'relu':
            self.activation_func = tf.nn.relu
        elif activation_func == 'sigmoid':
            self.activation_func = tf.sigmoid
        elif activation_func == 'tanh':
            self.activation_func = tf.tanh
        elif activation_func == 'linear':
            self.activation_func = tf.identity
        elif activation_func == 'softplus':
            self.activation_func = tf.nn.softplus
        elif activation_func == 'quad':
            self.activation_func = tf.square
        elif activation_func == 'elu':
            self.activation_func = tf.nn.elu
        else:
            raise ValueError('Invalid activation function ''%s''' %
                             activation_func)

        # create excitatory/inhibitory mask
        if num_inh > num_outputs:
            raise ValueError('Too many inhibitory units designated')
        self.ei_mask = [1] * (num_outputs - num_inh) + [-1] * num_inh
        if num_inh > 0:
            self.ei_mask_var = tf.constant(
                self.ei_mask, dtype=tf.float32, name='ei_mask')
        else:
            self.ei_mask_var = None

        # save positivity constraint on weights
        self.pos_constraint = pos_constraint

        # use tf's summary writer to save layer activation histograms
        if log_activations:
            self.log = True
        else:
            self.log = False

        # build layer
        with tf.name_scope(self.scope):

            # resolve weights initializer string
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
            # initialize numpy array that will feed placeholder
            if pos_constraint is True:
                init_weights = np.maximum(init_weights,0)
            self.weights = init_weights.astype('float32')
            # initialize weights placeholder/variable
            with tf.name_scope('weights_init'):
                self.weights_ph = tf.placeholder_with_default(
                    self.weights,
                    shape=[num_inputs, num_outputs],
                    name='weights_ph')
            self.weights_var = tf.Variable(
                self.weights_ph,
                dtype=tf.float32,
                name='weights_var')

            # resolve biases initializer string
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
            # initialize numpy array that will feed placeholder
            self.biases = init_biases.astype('float32')
            # initialize biases placeholder/variable
            with tf.name_scope('biases_init'):
                self.biases_ph = tf.placeholder_with_default(
                    self.biases,
                    shape=[1, num_outputs],
                    name='biases_ph')
            self.biases_var = tf.Variable(
                self.biases_ph,
                dtype=tf.float32,
                name='biases_var')

            # save layer regularization info
            self.reg = Regularization(num_inputs=num_inputs,
                                      num_outputs=num_outputs,vals=reg_initializer)

            # push data through layer
            if self.pos_constraint:
                pre = tf.add(tf.matmul(inputs,
                                       tf.maximum(0.0, self.weights_var)),
                             self.biases_var)
            else:
                pre = tf.add(tf.matmul(inputs, self.weights_var),
                             self.biases_var)

            if self.ei_mask_var is not None:
                post = tf.multiply(self.activation_func(pre), self.ei_mask_var)
            else:
                post = self.activation_func(pre)

            self.outputs = post

            if self.log:
                tf.summary.histogram('act_pre', pre)
                tf.summary.histogram('act_post', post)
    # END __init__

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
