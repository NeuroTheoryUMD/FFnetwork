"""Basic network-building tools"""

from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf


class Regularization(object):
    """Class for handling layer-wise regularization
    
    Attributes:
        vals (dict): values for different types of regularization stored as
            floats
        vals_tf (dict): values for different types of regularization stored as
            tf.constants
        mats (dict): matrices for different types of regularization
        num_inputs (int): dimension of layer input size
        
    """

    allowed_reg_types = ['l1', 'l2', 'd2t']

    def __init__(self, num_inputs=None, vals=None):
        """Constructor for Regularization class
        
        Args:
            num_inputs (int): dimension of input size (for building reg mats)
            vals (dict, optional): key-value pairs specifying value for each
                type of regularization 

        Raises:
            TypeError: If `num_inputs` is not specified
            
        """

        # check input
        if num_inputs is None:
            raise TypeError('Must specify `num_inputs`')
        self.num_inputs = num_inputs

        # set all default values to zero
        self.vals = {'l1': None, 'l2': None, 'd2t': None}
        self.vals_tf = {'l1': None, 'l2': None, 'd2t': None}
        self.mats = {'l1': None, 'l2': None, 'd2t': None}

        # read user input
        if vals is not None:
            for key, val in vals:
                self.set_val(key, val)
    # END __init__

    def set_val(self, reg_type, reg_val):
        """Set regularization value in list (doesn't update tf Graph until a
        session is run; see `assign_reg_vals`)
        
        Args:
            reg_type (str): see `allowed_reg_types` for options
            reg_val (float): value of regularization parameter
            
        Returns:
            new_reg_type (bool): True if `reg_type` has not been previously set
            
        Raises:
            ValueError: If `reg_type` is not a valid regularization type
            ValueError: If `reg_val` is less than 0.0
            
        """

        # check inputs
        if reg_type not in self.allowed_reg_types:
            raise ValueError('Invalid regularization type')
        if reg_val < 0.0:
            raise ValueError('`reg_val` must be greater than or equal to zero')

        # determine if this is a new type of regularization
        if self.vals[reg_type] is None and reg_val > 0.0:
            new_reg_type = True
        else:
            new_reg_type = False

        self.vals[reg_type] = reg_val

        return new_reg_type

    def assign_reg_vals(self, sess):
        """Update regularization values in default tf graph"""
        # loop through all types of regularization
        for reg_type, reg_val in self.vals.iteritems():
            # only assign if applicable
            if reg_val is not None:
                with tf.name_scope(reg_type + '_loss'):
                    sess.run(
                        self.vals_tf[reg_type].assign(self.vals[reg_type]))
    # END set_val

    def define_loss(self, weights):
        """Define regularization loss in default tf graph"""
        reg_loss = []
        # loop through all types of regularization
        for reg_type, reg_val in self.vals.iteritems():
            # set up reg val variable if it doesn't already exist
            if reg_val is not None:
                with tf.name_scope(reg_type + '_loss'):
                    # use tf Variable (rather than constant) for easy
                    # reassignment later
                    self.vals_tf[reg_type] = tf.Variable(
                        reg_val,
                        trainable=False,
                        dtype=tf.float32,
                        name=reg_type + '_param')
                    self.mats[reg_type] = self._build_reg_mats(reg_type)
                    reg_loss.append(self._calc_penalty(reg_type, weights))

        # if no regularization, define regularization loss to be zero
        if len(reg_loss) == 0:
            reg_loss.append(tf.constant(0.0, tf.float32, name='zero'))

        return tf.add_n(reg_loss)
    # END define_loss

    def _build_reg_mats(self, reg_type):
        """Build regularization matrices

        Args:
            reg_type (str): see `allowed_reg_types` for options

        """

        if reg_type == 'd2t':
            n = self.num_inputs
            reg_mat = np.zeros(shape=[n, n], dtype='float32')
            reg_mat += np.diag([-1.0] * (n - 1), -1) \
                + np.diag([2.0] * n) \
                + np.diag([-1.0] * (n - 1), 1)
            # Add boundary conditions (none) at edges
            reg_mat[0, :] = 0
            reg_mat[-1, :] = 0
            name = 'd2t_laplacian'
        else:
            reg_mat = 0.0
            name = 'lp_placeholder'

        return tf.constant(reg_mat, dtype=tf.float32, name=name)
    # END build_reg_mats

    def _calc_penalty(self, reg_type, weights):
        """Calc reg penalty for various reg types"""
        if reg_type == 'l1':
            reg_pen = tf.multiply(
                self.vals_tf['l1'],
                tf.reduce_sum(tf.abs(weights)))
        elif reg_type == 'l2':
            reg_pen = tf.multiply(
                self.vals_tf['l2'],
                tf.nn.l2_loss(weights))
        elif reg_type == 'd2t':
            reg_pen = tf.multiply(
                self.vals_tf['d2t'],
                tf.reduce_sum(tf.square(
                    tf.matmul(self.mats['d2t'], weights))))
        else:
            reg_pen = tf.constant(0.0)
        return reg_pen

    def get_reg_pen(self, sess, weights):
        """Build structure that details reg penalty from each reg type"""

        reg_dict = {}
        for reg_type, reg_val in self.vals.iteritems():
            if reg_val is not None:
                with tf.name_scope(reg_type + '_loss'):
                    reg_pen = sess.run(self._calc_penalty(reg_type, weights))
            else:
                reg_pen = 0.0
            reg_dict[reg_type] = reg_pen

        return reg_dict
