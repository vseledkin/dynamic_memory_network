import numpy as np
import tensorflow as tf


class DMN(object):
    """Dynamic Memory Network (Kumar 2015)."""

    def __init__(self, n_in_in, n_in_hid):

        ####################
        ### Input module ###
        ####################
        
        with tf.variable_scope("input"):
            with tf.variable_scope("z"):
                W_z = tf.get_variable("W", [n_in_in, n_in_hid],
                                      initializer=tf.random_normal_initializer())
                U_z = tf.get_variable("U", [n_in_hid, n_in_hid],
                                      initializer=tf.random_normal_initializer())
                b_z = tf.get_variable("b", [n_in_hid],
                                      initializer=tf.constant_initializer())
             

        

        #######################
        ### Question module ###
        #######################

        ##############################
        ### Episodic memory module ###
        ##############################

        #####################
        ### Answer module ###
        #####################

    def _GRU_words(self, X, H_0):
        """
        Computes the symbolic GRU given X, where each X has one sentence.
        :param X: Input matrix of size [number of examples, length of sentence, embedding dimension]
        :param H_0: Initial state of size [length of sentence, hidden dimension]
        :return: List of output for each word
        """

        n_in = X.shape[2]
        n_hid = H_0.shape[1]

        
        # create variables
        with tf.variable_scope("z"):
            W_z = tf.get_variable("W", [n_in, n_hid],
                                  initializer=tf.random_normal_initializer())
            U_z = tf.get_variable("U", [n_hid, n_hid],
                                  initializer=tf.random_normal_initializer())
            b_z = tf.get_variable("b", [n_hid],
                                  initializer=tf.constant_initializer())

        with tf.variable_scope("r"):
            W_r = tf.get_variable("W", [n_in, n_hid],
                                  initializer=tf.random_normal_initializer())
            U_r = tf.get_variable("U", [n_hid, n_hid],
                                  initializer=tf.random_normal_initializer())
            b_r = tf.get_variable("b", [n_hid],
                                  initializer=tf.constant_initializer())

        with tf.variable_scope("h_tilde"):
            W = tf.get_variable("W", [n_in, n_hid],
                                initializer=tf.random_normal_initializer())
            U = tf.get_variable("U", [n_hid, n_hid],
                                initializer=tf.random_normal_initializer())
            b = tf.get_variable("b", [n_hid],
                                initializer=tf.constant_initializer())

            
        # transpose X so computations can be vectorized
        X = tf.transpose(X, [1, 0, 2])

        # convert H_0 so it's useable for each example
        H_0 = tf.tile(tf.expand_dims(H_0, 1), tf.pack([1, X.shape[1], 1]))

        

        
            
            

        

