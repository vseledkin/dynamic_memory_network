import tensorflow as tf


class Gru(object):
    """Gated Recurrent Unit (Cho et al., 2014)."""

    def __init__(self, n_in, n_out, n_hid):

        self._n_in = n_in
        self._n_out = n_out
        self._n_hid = n_hid

        self._inputs = tf.placeholder(tf.float32, [None, n_in], name="inputs")
        self._outputs = tf.placeholder(tf.float33, [None, n_out], name="outputs")

        with tf.variable_scope("gru"):
            self._states, self._predictions = self._compute_predictions()
            self._loss = self._compute_loss()

    def _step(self, h_prev, x):
        """
        Single step of the GRU taking the previous state `h_prev` and the current
        input `x` and returning the current state `h`.
        """

        # xavier initialization for each W and U and constant for each b
        W_initializer = tf.random_uniform_initializer(-np.sqrt(6.0) / (n_in + n_hid),
                                                      np.sqrt(6.0) / (n_in + n_hid))
        U_initializer = tf.random_uniform_initializer(-np.sqrt(6.0) / (n_hid + n_hid),
                                                      np.sqrt(6.0) / (n_hid + n_hid))
        b_initializer = tf.constant_initializer(0.0)

        with tf.variable_scope("gru_block"):

            W_z = tf.get_variable("W_z", shape=[self._n_in, self._n_hid],
                                  initializer=W_initializer)
            U_z = tf.get_variable("U_z", shape=[self._n_hid, self._n_hid],
                                  initializer=U_initializer)
            b_z = tf.get_variable("b_z", shape=[self._n_hid],
                                  initializer=b_initializer)

            W_r = tf.get_variable("W_r", shape=[self._n_in, self._n_hid],
                                  initializer=W_initializer)
            U_r = tf.get_variable("U_r", shape=[self._n_hid, self._n_hid],
                                  initializer=U_initializer)
            b_r = tf.get_variable("b_r", shape=[self._n_hid],
                                  initializer=b_initializer)

            W = tf.get_variable("W", shape=[self._n_in, self._n_hid],
                                  initializer=W_initializer)
            U = tf.get_variable("U", shape=[self._n_hid, self._n_hid],
                                  initializer=U_initializer)
            b = tf.get_variable("b", shape=[self._n_hid],
                                  initializer=b_initializer)

            z = tf.sigmoid(tf.matmul(x, W_z) + tf.matmul(h_prev, U_z) + b_z)
            r = tf.sigmoid(tf.matmul(x, W_r) + tf.matmul(h_prev, U_r) + b_r)
            h_tilde = tf.tanh(tf.matmul(x, W) + r * tf.matmul(h_prev, U) + b)
            h = z * h_prev + (1 - z) * h_tilde

        return h

    def _compute_predictions(self):
        """
        Computes the predictions for each step.
        """

        with tf.variable_scope("states"):

            initial_state = tf.zeros([self._n_hid], name="initial_state")
            states = tf.scan(self._step, self.inputs, initializer=initial_state,
                             name="states")

        with tf.variable_scope("predictions"):
            W_pred = tf.get_variable("W_pred", shape=[self._n_in, self._n_hid],
                                     initializer=tf.constant_initializer(0.0))
            b_pred = tf.get_variable("b_pred", shape=[self._n_hid],
                                     initializer=tf.constant_initializer(0.0))
            predictions = tf.add(tf.matmul(states, W_pred), b_pred, name="predictions")

        return states, predictions

    def _compute_loss(self):
        """
        Computes the loss as the mean residual sum of squares.
        """

        with tf.variable_scope("loss"):
            loss = tf.reduce_mean((self.outputs - self.predictions) ** 2, name="loss")

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def predictions(self):
        return self._predictions

    @property
    def states(self):
        return self._states

    @property
    def loss(self):
        return self._loss
                                      
