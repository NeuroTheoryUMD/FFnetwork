"""Basic network-building tools"""

from __future__ import print_function
from __future__ import division

import os
import sys
import numpy as np
import tensorflow as tf


class Network(object):
    """Base class for neural networks"""

    _log_min = 1e-5  # constant to add to all arguments to logarithms

    def __init__(self):
        """Constructor for Network class; model architecture should be defined
        here"""

        # default: use cpu for training
        self.use_gpu = False
        self.sess_config = tf.ConfigProto(device_count={'GPU': 0})

    def _initialize_data_pipeline(self):
        """Define pipeline for feeding data into model"""

        # placeholders for data
        self.data_in_ph = tf.placeholder(
            dtype=tf.float32,
            shape=[self.num_examples, self.input_size],
            name='input_ph')
        self.data_out_ph = tf.placeholder(
            dtype=tf.float32,
            shape=[self.num_examples, self.output_size],
            name='output_ph')

        # turn placeholders into variables so they get put on GPU
        self.data_in_var = tf.Variable(
            self.data_in_ph,   # initializer for Variable
            trainable=False,   # no GraphKeys.TRAINABLE_VARS
            collections=[],    # no GraphKeys.GLOBAL_VARS
            name='input_var')
        self.data_out_var = tf.Variable(
            self.data_out_ph,  # initializer for Variable
            trainable=False,   # no GraphKeys.TRAINABLE_VARS
            collections=[],    # no GraphKeys.GLOBAL_VARS
            name='output_var')

        # use selected subset of data
        self.indices = tf.placeholder(
            dtype=tf.int32,
            shape=None,
            name='indices_ph')
        self.data_in_batch = tf.gather(
            self.data_in_var,
            self.indices,
            name='input_batch')
        self.data_out_batch = tf.gather(
            self.data_out_var,
            self.indices,
            name='output_batch')

    def _define_loss(self):
        """Loss function that will be used to optimize model parameters"""
        raise NotImplementedError

    def _define_optimizer(self, var_list=None):
        """Define one step of the optimization routine"""

        if self.learning_alg == 'adam':
            self.train_step = tf.train.AdamOptimizer(self.learning_rate). \
                minimize(self.cost_penalized)
        elif self.learning_alg == 'lbfgs':
            if var_list is None:
                self.train_step = tf.contrib.opt.ScipyOptimizerInterface(
                    self.cost_penalized,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 10000,
                        'disp': False})
            else:
                self.train_step = tf.contrib.opt.ScipyOptimizerInterface(
                    self.cost_penalized,
                    var_list=var_list,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 10000,
                        'disp': False})
    # END _define_optimizer

    def train(
            self,
            input_data=None,
            output_data=None,
            train_indxs=None,
            test_indxs=None,
            layers_to_skip=None,
            biases_const=False,
            batch_size=100,
            epochs_training=10000,
            epochs_disp=None,
            epochs_ckpt=None,
            epochs_early_stop=None,
            epochs_summary=None,
            early_stop=False,
            output_dir=None):
        """Network training function

        Args:
            input_data (time x input_dim numpy array): input to network
            output_data (time x output_dim numpy array): desired output of 
                network
            train_indxs (numpy array, optional): subset of data to use for 
                training
            test_indxs (numpy array, optional): subset of data to use for 
                testing; if available these are used when displaying updates,
                and are also the indices used for early stopping if enabled
            batch_size (int, optional): batch size used by the gradient
                descent-based optimizers (adam).
            epochs_training (int, optional): number of epochs for gradient 
                descent-based optimizers
            epochs_disp (int, optional): number of epochs between updates to 
                the console
            epochs_ckpt (int, optional): number of epochs between saving 
                checkpoint files
            epochs_early_stop (int, optional): number of epochs between checks
                for early stopping
            epochs_summary (int, optional): number of epochs between saving
                network summary information
            early_stop (bool, optional): if True, training exits when the
                cost function evaluated on test_indxs begins to increase 
            output_dir (str, optional): absolute path for saving checkpoint
                files and summary files; must be present if either epochs_ckpt  
                or epochs_summary is not 'None'. If `output_dir` is not 'None',
                the graph will automatically be saved.

        Returns:
            epoch (int): number of total training epochs

        Raises:
            ValueError: If `input_data` and `output_data` don't share time dim
            ValueError: If data time dim doesn't match that specified in model
            ValueError: If `epochs_ckpt` is not None and output_dir is 'None'
            ValueError: If `epochs_summary` is not 'None' and `output_dir` is 
                'None'
            ValueError: If `early_stop` is True and `test_indxs` is 'None'

        """

        # check input
        if input_data.shape[0] != output_data.shape[0]:
            raise ValueError('Input and output data must have matching ' +
                             'number of examples')
        if input_data.shape[0] != self.num_examples:
            raise ValueError('Input/output data dims must match model values')
        if epochs_ckpt is not None and output_dir is None:
            raise ValueError('output_dir must be specified to save model')
        if epochs_summary is not None and output_dir is None:
            raise ValueError('output_dir must be specified to save summaries')
        if early_stop and test_indxs is None:
            raise ValueError('test_indxs must be specified for early stopping')

        if train_indxs is None:
            train_indxs = np.arange(self.num_examples)

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:

            # handle output directories
            if output_dir is not None:

                # remake checkpoint directory
                if epochs_ckpt is not None:
                    ckpts_dir = os.path.join(output_dir, 'ckpts')
                    if os.path.isdir(ckpts_dir):
                        tf.gfile.DeleteRecursively(ckpts_dir)
                    os.makedirs(ckpts_dir)

                # remake training summary directories
                summary_dir_train = os.path.join(
                    output_dir, 'summaries', 'train')
                if os.path.isdir(summary_dir_train):
                    tf.gfile.DeleteRecursively(summary_dir_train)
                os.makedirs(summary_dir_train)
                train_writer = tf.summary.FileWriter(
                    summary_dir_train, sess.graph)

                # remake testing summary directories
                summary_dir_test = os.path.join(
                    output_dir, 'summaries', 'test')
                if test_indxs is not None:
                    if os.path.isdir(summary_dir_test):
                        tf.gfile.DeleteRecursively(summary_dir_test)
                    os.makedirs(summary_dir_test)
                    test_writer = tf.summary.FileWriter(
                        summary_dir_test, sess.graph)

            else:
                train_writer = None
                test_writer = None

            # redefine optimizer if necessary
            if layers_to_skip is not None:
                layers_included = list(set(range(self.network.num_layers))
                                       - set(layers_to_skip))
                var_list = []
                for layer in layers_included:
                    var_list.append(self.network.layers[layer].weights_var)
                    if not biases_const:
                        var_list.append(
                            self.network.layers[layer].biases_var)

                with tf.variable_scope('optimizer'):
                    self._define_optimizer(var_list)

            # overwrite initialized values of network with stored values
            self._restore_params(sess, input_data, output_data)

            # select learning algorithm
            if self.learning_alg == 'adam':
                epoch = self._train_adam(
                    sess=sess,
                    train_writer=train_writer,
                    test_writer=test_writer,
                    train_indxs=train_indxs,
                    test_indxs=test_indxs,
                    batch_size=batch_size,
                    epochs_training=epochs_training,
                    epochs_disp=epochs_disp,
                    epochs_ckpt=epochs_ckpt,
                    epochs_early_stop=epochs_early_stop,
                    epochs_summary=epochs_summary,
                    early_stop=early_stop,
                    output_dir=output_dir)
            elif self.learning_alg == 'lbfgs':
                self.train_step.minimize(
                    sess,
                    feed_dict={self.indices: train_indxs})
                epoch = float('NaN')
            else:
                raise ValueError('Invalid learning algorithm')

            # write out weights/biases to numpy arrays before session closes
            self.network.write_model_params(sess)

        return epoch
    # END train

    def _train_adam(
            self,
            sess=None,
            train_writer=None,
            test_writer=None,
            train_indxs=None,
            test_indxs=None,
            batch_size=None,
            epochs_training=None,
            epochs_disp=None,
            epochs_ckpt=None,
            epochs_early_stop=None,
            epochs_summary=None,
            early_stop=None,
            output_dir=None):
        """Training function for adam optimizer to clean up code in `train`"""

        if self.use_batches:
            num_batches = train_indxs.shape[0] // batch_size
        else:
            batch_size = train_indxs.shape[0]
            num_batches = 1

        if early_stop:
            prev_cost = float('Inf')

        # start training loop
        for epoch in range(epochs_training):

            # shuffle data before each pass
            train_indxs_perm = np.random.permutation(train_indxs)

            # pass through dataset once
            for batch in range(num_batches):
                # get training indices for this batch
                batch_indxs = train_indxs_perm[
                              batch * batch_size:
                              (batch + 1) * batch_size]
                # one step of optimization routine
                sess.run(
                    self.train_step,
                    feed_dict={self.indices: batch_indxs})

            # print training updates
            if epochs_disp is not None and \
                    (epoch % epochs_disp == epochs_disp - 1 or epoch == 0):

                cost = sess.run(
                    self.cost,
                    feed_dict={self.indices: train_indxs_perm})
                r2s, _ = self._get_r2s(sess, train_indxs_perm)
                print('\nEpoch %03d:' % epoch)
                print('   train cost = %2.5f' % cost)
                print('   train r2 = %1.4f' % np.mean(r2s))

                # print additional testing info
                if test_indxs is not None:
                    cost_test = sess.run(
                        self.cost,
                        feed_dict={self.indices: test_indxs})
                    r2s_test, _ = self._get_r2s(sess, test_indxs)
                    print('   test cost = %2.5f' % cost_test)
                    print('   test r2 = %1.4f' % np.mean(r2s_test))

            # save model checkpoints
            if epochs_ckpt is not None and \
                    (epoch % epochs_ckpt == epochs_ckpt - 1 or epoch == 0):
                save_file = os.path.join(
                    output_dir, 'ckpts',
                    str('epoch_%05g.ckpt' % epoch))
                self.checkpoint_model(sess, save_file)

            # save model summaries
            if epochs_summary is not None and \
                    (epoch % epochs_summary == epochs_summary - 1
                     or epoch == 0):
                summary = sess.run(
                    self.merge_summaries,
                    feed_dict={self.indices: train_indxs})
                train_writer.add_summary(summary, epoch)
                print('Writing train summary')
                if test_indxs is not None:
                    summary = sess.run(
                        self.merge_summaries,
                        feed_dict={self.indices: test_indxs})
                    test_writer.add_summary(summary, epoch)
                    print('Writing test summary')

            # check for early stopping
            if early_stop and \
                    epoch % epochs_early_stop == epochs_early_stop - 1:

                cost_test = sess.run(
                    self.cost,
                    feed_dict={self.indices: test_indxs})

                if cost_test >= prev_cost:

                    # save model checkpoint if desired and necessary
                    if epochs_ckpt is not None and \
                            epochs_ckpt != epochs_early_stop:
                        save_file = os.path.join(
                            output_dir, 'ckpts',
                            str('epoch_%05g.ckpt' % epoch))
                        self.checkpoint_model(sess, save_file)

                    # save model summaries if desired and necessary
                    if epochs_summary is not None and \
                            epochs_summary != epochs_early_stop:
                        summary = sess.run(
                            self.merge_summaries,
                            feed_dict={self.indices: train_indxs})
                        train_writer.add_summary(summary, epoch)
                        print('Writing train summary')
                        if test_indxs is not None:
                            summary = sess.run(
                                self.merge_summaries,
                                feed_dict={self.indices: test_indxs})
                            test_writer.add_summary(summary, epoch)
                            print('Writing test summary')

                    break  # out of epochs loop
                else:
                    prev_cost = cost_test

        return epoch
    # END _train_adam

    def _get_r2s(self, sess, data_indxs=None):
        """Transform a given input into its reconstruction 

        Args:
            sess (tf.Session object): current session object to run graph
            data_indxs (numpy array, optional): indexes of data to use in 
                calculating forward pass; if not supplied, all data is used             

        Returns:
            r2s (1 x num_cells numpy array): pseudo-r2 values for each cell
            lls (dict): contains log-likelihoods for fitted model, null model 
                (prediction is mean), and saturated model (prediction is true
                activity) for each cell

        Raises:
            ValueError: If both input and output data are not provided
            ValueError: If input/output time dims don't match

        """

        if data_indxs is None:
            data_indxs = np.arange(self.num_examples)

        data_in = sess.run(
            self.data_in_batch,
            feed_dict={self.indices: data_indxs})
        data_out = sess.run(
            self.data_out_batch,
            feed_dict={self.indices: data_indxs})
        pred = sess.run(
            self.network.layers[-1].outputs,
            feed_dict={self.indices: data_indxs})

        t, num_cells = data_in.shape
        mean_act = np.tile(np.mean(data_out, axis=0), (t, 1))

        if self.noise_dist == 'gaussian':

            ll = np.sum(np.square(data_out - pred), axis=0)
            ll_null = np.sum(np.square(data_out - mean_act), axis=0)
            ll_sat = 0.0

        elif self.noise_dist == 'poisson':

            ll = -np.sum(
                np.multiply(data_out, np.log(self._log_min + pred))
                - pred, axis=0)
            ll_null = -np.sum(
                np.multiply(data_out, np.log(self._log_min + mean_act))
                - mean_act, axis=0)
            ll_sat = np.multiply(data_out, np.log(self._log_min + data_out))
            ll_sat = -np.sum(ll_sat - data_out, axis=0)

        elif self.noise_dist == 'bernoulli':

            ll_sat = 1.0
            ll_null = 0.0
            ll = 0.0

        r2s = 1.0 - np.divide(ll_sat - ll, ll_sat - ll_null)
        r2s[ll_sat == ll_null] = 1.0

        lls = {
            'll': ll,
            'll_null': ll_null,
            'll_sat': ll_sat
        }

        return r2s, lls
    # END _get_r2s

    def _restore_params(self, sess, input_data, output_data):
        """Restore model parameters from numpy matrices and update
        regularization values from list. This function is called by any other 
        function that needs to initialize a new session to run parts of the 
        graph."""

        # initialize all parameters randomly
        sess.run(self.init)

        # initialize input/output data
        sess.run(self.data_in_var.initializer,
                 feed_dict={self.data_in_ph: input_data})
        sess.run(self.data_out_var.initializer,
                 feed_dict={self.data_out_ph: output_data})

        # overwrite randomly initialized values of model with stored values
        self._assign_model_params(sess)

        # update regularization parameter values
        self._assign_reg_vals(sess)

    def _assign_model_params(self, sess):
        """Assigns parameter values previously stored in numpy arrays to 
        tf Variables in model; function needs to be implemented by specific 
        model"""
        raise NotImplementedError()

    def _assign_reg_vals(self, sess):
        """Loops through all current regularization penalties and updates  
        parameter values in the tf Graph; needs to be implemented by specific 
        model"""
        raise NotImplementedError()

    def checkpoint_model(self, sess, save_file):
        """Checkpoint model parameters in tf Variables

        Args:
            sess (tf.Session object): current session object to run graph
            save_file (str): full path to output file

        """

        if not os.path.isdir(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))

        self.saver.save(sess, save_file)
        print('Model checkpointed to %s' % save_file)

    def restore_model(self, save_file, input_data, output_data):
        """Restore previously checkpointed model parameters in tf Variables 

        Args:
            sess (tf.Session object): current session object to run graph
            save_file (str): full path to saved model
            input_data (time x input_dim numpy array): input to network
            output_data (time x output_dim numpy array): desired output of 
                network

        Raises:
            ValueError: If `save_file` is not a valid filename

        """

        if not os.path.isfile(save_file + '.meta'):
            raise ValueError(str('%s is not a valid filename' % save_file))

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:

            # initialize tf params in new session
            self._restore_params(sess, input_data, output_data)
            # restore saved variables into tf Variables
            self.saver.restore(sess, save_file)
            # write out weights/biases to numpy arrays before session closes
            self.network.write_model_params(sess)

    def save_model(self, save_file):
        """Save full network object using dill (extension of pickle)

        Args:
            save_file (str): full path to output file

        """

        import dill

        sys.setrecursionlimit(10000)  # for dill calls to pickle

        if not os.path.isdir(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))

        with file(save_file, 'wb') as f:
            dill.dump(self, f)
        print('Model pickled to %s' % save_file)

    @classmethod
    def load_model(cls, save_file):
        """Restore previously saved network object 

        Args:
            save_file (str): full path to saved model

        Raises:
            ValueError: If `save_file` is not a valid filename

        """

        import dill

        if not os.path.isfile(save_file):
            raise ValueError(str('%s is not a valid filename' % save_file))

        with file(save_file, 'rb') as f:
            return dill.load(f)
