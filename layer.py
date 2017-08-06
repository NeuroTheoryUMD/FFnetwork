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
        weights_tf (tf Tensor): weights in layer
        biases_tf (tf Tensor): biases in layer
        weights (numpy array): shadow variable of `weights_tf` that allows for 
            easier manipulation outside of tf sessions
        biases (numpy array): shadow variable of `biases_tf` that allows for 
            easier manipulation outside of tf sessions
        activation_func (tf activation function): activation function in layer
        reg (Regularization object): holds regularizations values and matrices
            (as tf constants) for layer
        ei_mask_tf (tf constant): mask of +/-1s to multiply output of layer
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
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' | 'elu'                
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
        elif activation_func == 'elu':
            self.activation_func = tf.nn.elu
        else:
            raise ValueError('Invalid activation function')

        # create excitatory/inhibitory mask
        if num_inh > num_outputs:
            raise ValueError('Too many inhibitory units designated')
        self.ei_mask = [1] * (num_outputs - num_inh) + [-1] * num_inh
        if num_inh > 0:
            self.ei_mask_tf = tf.constant(self.ei_mask, dtype=tf.float32)
        else:
            self.ei_mask_tf = None

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
            weight_dims = [num_inputs, num_outputs]
            with tf.name_scope('weight_init'):
                if weights_initializer == 'trunc_normal':
                    init_weights = tf.truncated_normal(weight_dims, stddev=0.1)
                elif weights_initializer == 'normal':
                    init_weights = tf.random_normal(weight_dims, stddev=0.1)
                elif weights_initializer == 'zeros':
                    init_weights = tf.constant(0.0, shape=weight_dims)
                else:
                    raise ValueError('Invalid weights_initializer string')
            # initialize weights
            self.weights_tf = tf.Variable(
                initial_value=init_weights,
                dtype=tf.float32,
                name='weights')

            # resolve biases initializer string
            bias_dims = [1, num_outputs]
            if biases_initializer == 'trunc_normal':
                init_biases = tf.truncated_normal(bias_dims, stddev=0.1)
            elif biases_initializer == 'normal':
                init_biases = tf.random_normal(bias_dims, stddev=0.1)
            elif biases_initializer == 'zeros':
                init_biases = tf.constant(0.0, shape=bias_dims)
            else:
                raise ValueError('Invalid biases_initializer string')
            # initialize biases
            self.biases_tf = tf.Variable(
                initial_value=init_biases,
                dtype=tf.float32,
                name='biases')

            # save layer regularization info
            self.reg = Regularization(num_inputs=num_inputs,
                                      vals=reg_initializer)

            # push data through layer
            if self.pos_constraint:
                pre = tf.add(tf.matmul(inputs,
                                       tf.maximum(0.0, self.weights_tf)),
                             self.biases_tf)
            else:
                pre = tf.add(tf.matmul(inputs, self.weights_tf),
                             self.biases_tf)

            if self.ei_mask_tf is not None:
                post = tf.multiply(self.activation_func(pre), self.ei_mask_tf)
            else:
                post = self.activation_func(pre)

            self.outputs = post

            if self.log:
                tf.summary.histogram('act_pre', pre)
                tf.summary.histogram('act_post', post)

        # create "shadow variables" for easier manipulation outside tf session
        self.weights = np.zeros(shape=[num_inputs, num_outputs],
                                dtype='float32')
        self.biases = np.zeros(shape=[1, num_outputs], dtype='float32')

    # END __init__

    def read_weights(self, sess):
        """Read weights/biases in numpy arrays into tf Variables"""
        with tf.name_scope(self.scope):
            sess.run(self.weights_tf.assign(self.weights))
            sess.run(self.biases_tf.assign(self.biases))

    def write_weights(self, sess):
        """Write weights/biases in tf Variables to numpy arrays"""
        self.weights = sess.run(self.weights_tf)
        self.biases = sess.run(self.biases_tf)

    def define_regularization_loss(self):
        """Wrapper function for building regularization portion of graph"""
        with tf.name_scope(self.scope):
            return self.reg.define_loss(self.weights_tf)

    def set_regularization(self, reg_type, reg_val):
        """Wrapper function for setting regularization"""
        return self.reg.set_val(reg_type, reg_val)

    def assign_reg_vals(self, sess):
        """Wrapper function for assigning regularization values"""
        with tf.name_scope(self.scope):
            self.reg.assign_reg_vals(sess)

    def get_reg_pen(self, sess):
        """Wrapper function for returning regularization penalty struct"""
        return self.reg.get_reg_pen(sess, self.weights_tf)
