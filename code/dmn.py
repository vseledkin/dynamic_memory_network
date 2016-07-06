import numpy as np
import tensorflow as tf

from gru import Gru


class Dmn(object):
    """Dynamic Memory Network (Kumar 2015)."""

    def __init__(self, in_n_in, in_n_out, in_n_hid):

        ####################
        ### Input module ###
        ####################

        self._in_n_in = in_n_in
        self._in_n_out = in_n_out
        self._in_n_hid = in_n_hid

        self._in_inputs = tf.placeholder(tf.float32, [None, input_n_in], name="in_inputs")
        self._in_outputs = tf.placeholder(tf.float32, [None, input_n_out], name="in_outputs")

        self._in_gru = Gru(in_n_in, in_n_out, in_n_hid)

        #######################
        ### Question module ###
        #######################

        ##############################
        ### Episodic memory module ###
        ##############################

        #####################
        ### Answer module ###
        #####################

        pass

