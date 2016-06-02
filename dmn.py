import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell


class DMN(object):
    """Dynamic Memory Network (Kumar 2015)."""

    def __init__(self, glove, n_input, n_question, n_answer):

        ####################
        ### Placeholders ###
        ####################

        self._i = tf.placeholder(tf.int32, [None])  # input
        self._input_index = tf.placeholder(tf.int32, [None])  # index of input facts
        self._q = tf.placeholder(tf.int32, [None])  # question
        self._a = tf.placeholder(tf.int32, [None])  # answer

        #################
        ### Embedding ###
        #################

        embeddings = tf.Variable(glove, trainable=False)

        i = tf.nn.embedding_lookup(embeddings, self._i)
        q = tf.nn.embedding_lookup(embeddings, self._q)
        a = tf.nn.embedding_lookup(embeddings, self._a)
            
        ####################
        ### Input module ###
        ####################
                    
        input_gru = rnn_cell.GRUCell(n_input)
        input_out, _ = rnn.rnn(input_gru, i, dtype=tf.float32)
        input_facts = input_out[self._input_index]
        
        #######################
        ### Question module ###
        #######################

        question_gru = rnn_cell.GRUCell(n_question)
        question_out, _ = rnn.rnn(question_gru, q, dtype=tf.float32)
        question_vector = question_out

        ##############################
        ### Episodic memory module ###
        ##############################

        #####################
        ### Answer module ###
        #####################


        ############
        ### Cost ###
        ############

        

    def fit(self, input, question, answer):
        """Fits the model to the data."""

        

        

        

        
            
            

        

