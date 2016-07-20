import tensorflow as tf
import pandas as pd
import numpy as np


class DMN(object):
    """Dynamic Memory Network (Kumar 2015)"""

    def __init__(self, n_emb, n_hid, n_episodes):

        self._n_emb = n_emb
        self._n_hid = n_hid
        self._n_episodes = n_episodes

        self._inputs = tf.placeholder(tf.int32, [None], name="inputs")
        self._fact_indices = tf.placeholder(tf.int32, [None], name="fact_indices")
        self._question = tf.placeholder(tf.int32, [None], name="question")
        self._outputs = tf.placeholder(tf.int32, [None], name="outputs")

        self._embeddings, self._embeddings_indices = self._get_embeddings()

        ####################
        ### Input module ###
        ####################

        with tf.variable_scope("facts"):
            self._facts = self._compute_facts()
            self._n_facts = tf.gather(tf.shape(self._facts), 0)

        #######################
        ### Question module ###
        #######################

        with tf.variable_scope("question"):
            self._q = self._compute_question()

        ##############################
        ### Episodic memory module ###
        ##############################

        with tf.variable_scope("episodic_module"):
            self._memory = self._compute_memory()

        #####################
        ### Answer module ###
        #####################

        with tf.variable_scope("answer_module"):
            self._answer, self._logits = self._compute_answer()

        ####################
        ### For training ###
        ###################

        with tf.variable_scope("loss"):
            self._loss = self._compute_loss()

    
    def _get_embeddings(self):
        """
        Gets the glove embeddings.
        """

        location = os.path.join("src", "glove.6B.%dd.txt" % self.n_emb)

        words_embeddings = pd.read_table(location, header=None)

        embeddings = words_embeddings[words_embeddings.columns[1:]].as_matrix().astype(np.float32)
        embeddings = np.vstack((np.zeros([self._n_emb]),
                                      embeddings,
                                      np.zeros([self._n_emb])))

        embeddings_indices = {w: i for i, w in enumerate(words_embeddings[0])}
        embeddings_indices["__end__"] = embeddings.shape[0] - 1

        return embeddings, embeddings_indices
 

    def _fact_step(self, h_prev, x):
        """
        Computes a single fact step.
        """

        e = tf.nn.embedding_lookup(self._embeddings, x)

        initializer = tf.random_uniform_initializer(-0.01, 0.01)

        with tf.variable("gru_block"):

            W_z = tf.get_variable("W_z", [self._n_hid, self._n_emd],
                                  initializer=initializer)
            U_z = tf.get_variable("U_z", [self._n_hid, self._n_hid],
                                  initializer=initializer)
            b_z = tf.get_variable("b_z", [self._n_hid],
                                  initializer=tf.constant_initializer(0.0))
            W_r = tf.get_variable("W_r", [self._n_hid, self._n_emd],
                                  initializer=initializer)
            U_r = tf.get_variable("U_r", [self._n_hid, self._n_hid],
                                  initializer=initializer)
            b_r = tf.get_variable("b_r", [self._n_hid],
                                  initializer=tf.constant_initializer(0.0))
            W = tf.get_variable("W", [self._n_hid, self._n_emd],
                                initializer=initializer)
            U = tf.get_variable("U", [self._n_hid, self._n_hid],
                                initializer=initializer)
            b = tf.get_variable("b", [self._n_hid],
                                initializer=tf.constant_initializer(0.0))

            z = tf.sigmoid(tf.matmul(W_z, e) + tf.matmul(U_z, h_prev) + b_z)
            r = tf.sigmoid(tf.matmul(W_r, e) + tf.matmul(U_r, h_prev) + b_r)
            h_tilde = tf.tanh(tf.matmul(W, e) + r * tf.matmul(U, h_prev) + b)
            h = z * h_prev + (1 - z) * h_tilde

        return h

        
    def _compute_facts(self):
        """
        Computes the facts embeddings.
        """

        with tf.variable_scope("states"):
            initial_state = tf.zeros([self._n_hid])
            states = tf.scan(self._fact_step, self._inputs,
                             initializer=initial_state)

        facts = tf.gather(states, self._fact_indices)

        return facts


    def _compute_question(self):
        """
        Computes the representation of the question.
        """

        with tf.variable_scope("states"):
            initial_state = tf.zeros([self._n_hid])
            q = tf.foldl(self._gru_step, self._question,
                         initializer = initial_state)

        return q


    def _attention_gates(self, m):
        """
        Computes a single attention step.
        """

        initializer = tf.random_uniform_initializer(-0.01, 0.01)

        with tf.variable_scope("attention"):

            W_b = tf.get_variable("W_b", [self._n_hid, self._n_hid],
                                  initializer=initializer)
            b_b = tf.get_variable("b_b", [self._n_hid],
                                  initializer=tf.constant_initializer(0.0))
            W_1 = tf.get_variable("W_1", [self._n_hid, self._n_hid],
                                  initializer=initializer)
            b_1 = tf.get_variable("b_1", [self._n_hid],
                                  initializer=tf.constant_initializer(0.0))
            W_2 = tf.get_variable("W_2", [self._n_hid, 1],
                                  initializer=initializer)
            b_2 = tf.get_variable("b_2", [1],
                                  initializer=tf.constant_initializer(0.0))

            Z = tf.concat(1,
                          [self._facts,
                           tf.tile(m, [self._n_facts, 1]),
                           tf.tile(self._q, [self._n_facts, 1]),
                           self._facts * m,
                           self._facts * self._q,
                           tf.abs(self._facts - m),
                           tf.abs(self._facts - self._q),
                           tf.matmul(tf.matmul(self._facts, W_b), m),
                           tf.matmul(tf.matmul(self._facts, W_b), self._q)])

            g = tf.tanh(tf.matmul(tf.matmul(Z, W_1) + b_1, W_2) + b_2)

            g = tf.squeeze(g)

        return g


    def _memory_step(self, h_prev, c, g):
        """
        Single memory step.
        """

        with tf.variable_scope("memory_block"):

            W_z = tf.get_variable("W_z", [self._n_hid, self._n_hid],
                                  initializer=initializer)
            U_z = tf.get_variable("U_z", [self._n_hid, self._n_hid],
                                  initializer=initializer)
            b_z = tf.get_variable("b_z", [self._n_hid],
                                  initializer=tf.constant_initializer(0.0))
            W_r = tf.get_variable("W_r", [self._n_hid, self._n_hid],
                                  initializer=initializer)
            U_r = tf.get_variable("U_r", [self._n_hid, self._n_hid],
                                  initializer=initializer)
            b_r = tf.get_variable("b_r", [self._n_hid],
                                  initializer=tf.constant_initializer(0.0))
            W = tf.get_variable("W", [self._n_hid, self._n_hid],
                                initializer=initializer)
            U = tf.get_variable("U", [self._n_hid, self._n_hid],
                                initializer=initializer)
            b = tf.get_variable("b", [self._n_hid],
                                initializer=tf.constant_initializer(0.0))

            z = tf.sigmoid(tf.matmul(W_z, c) + tf.matmul(U_z, h_prev) + b_z)
            r = tf.sigmoid(tf.matmul(W_r, c) + tf.matmul(U_r, h_prev) + b_r)
            h_tilde = tf.tanh(tf.matmul(W, c) + r * tf.matmul(U, h_prev) + b)
            h = z * h_prev + (1 - z) * h_tilde
            h = g * h + (1 - g) * h_prev
            

    def _episode_step(self, m_prev, _):
        """
        Single episode step.
        """

        with tf.variable_scope("attention"):
            attention = self._attention_gates(m_prev)

        with tf.variable_scope("episode"):
            initial_state = tf.zeros([self._n_hid])
            e = tf.foldl(self._memory_step, [self._facts, attention],
                         initializer=initial_state)

        return e

    
    def _compute_memory(self):
        """
        Computes the memory.
        """

        with tf.variable_scope("memory"):
            initial_state = tf.zeros([self._n_hid])
            _ = tf.zeros([self._n_episodes])
            m = tf.foldl(self._episode_step, _,
                         initializer=initial_state)

        return m


    def _answer_step(self, y_prev, a_prev):
        """
        Computes a single answer step.
        """

        x = tf.concat(0, [y_prev, self._q])

        initializer = tf.random_uniform_initializer(-0.01, 0.01)

        with tf.variable_scope("gru_block"):
            
            W_z = tf.get_variable("W_z", [self._n_hid, self._n_hid * 2],
                                  initializer=initializer)
            U_z = tf.get_variable("U_z", [self._n_hid, self._n_hid],
                                  initializer=initializer)
            b_z = tf.get_variable("b_z", [self._n_hid],
                                  initializer=tf.constant_initializer(0.0))
            W_r = tf.get_variable("W_r", [self._n_hid, self._n_hid * 2],
                                  initializer=initializer)
            U_r = tf.get_variable("U_r", [self._n_hid, self._n_hid],
                                  initializer=initializer)
            b_r = tf.get_variable("b_r", [self._n_hid],
                                  initializer=tf.constant_initializer(0.0))
            W = tf.get_variable("W", [self._n_hid, self._n_hid * 2],
                                initializer=initializer)
            U = tf.get_variable("U", [self._n_hid, self._n_hid],
                                initializer=initializer)
            b = tf.get_variable("b", [self._n_hid],
                                initializer=tf.constant_initializer(0.0))

            z = tf.sigmoid(tf.matmul(W_z, x) + tf.matmul(U_z, a) + b_z)
            r = tf.sigmoid(tf.matmul(W_r, x) + tf.matmul(U_r, a) + b_r)
            a_tilde = tf.tanh(tf.matmul(W, x) + r * tf.matmul(U, a) + b)
            a = z * a + (1 - z) * a_tilde

        with tf.variable_scope("predictions"):

            W_a = tf.get_variable("W_a", [self._n_hid, self._n_hid],
                                  initializer=initializer)
            
            y = tf.softmax(tf.matmul(W_a, a))

        return y, a


    def _compute_answer(self):
        """
        Computes the answer.
        """

        with tf.variable_scope("predictions"):
            y = self._q
            a = tf.zeros([self._n_hid])
            _ = tf.zeros([tf.shape(self._outputs)[0]])
            logits, _ = tf.scan(self._answer_step, _)
            predictions = tf.argmax(logits, 1)

        return predictions, logits


    def _compute_loss(self):
        """
        Computes the cross-validation loss.
        """

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self._logits,
                                                              self._outputs)

        return loss

        
        
        
            

        
