import tensorflow as tf


class RecurrentUnit:
    def __init__(self, input_n, state_n):
        self.input_weights = tf.Variable(tf.random.normal(shape=(state_n, input_n)))
        self.recurrent_weights = tf.Variable(tf.random.normal(shape=(state_n, state_n)))
        self.bias = tf.Variable(tf.zeros(shape=(state_n,)))
    def output(self, input_tensor, recurrent_state):
        preactivation = tf.linalg.matvec(self.input_weights, input_tensor) + tf.linalg.matvec(self.recurrent_weights, recurrent_state) + self.bias
        return tf.tanh(preactivation)
    
class RNN:
    def __init__(self, input_n, state_n):
        self.unit = RecurrentUnit(input_n, state_n)

    def output(self, input_sequence):
        if not input_sequence:
            raise ValueError("Input sequence cannot be empty.")
        states = []
        recurrent_state = tf.zeros(shape=(self.unit.state_n,), dtype=tf.float32)
        input_sequence = [tf.cast(input_tensor, tf.float32) if input_tensor.dtype != tf.float32 else input_tensor for input_tensor in input_sequence]
        for input_tensor in input_sequence:
            recurrent_state = self.unit.output(input_tensor, recurrent_state)
            states.append(recurrent_state)
        return states[-1]

class LSTMUnit:
    def __init__(self, input_tensor, hidden_state):
        self.input_tensor = tf.Variable(input_tensor, dtype=tf.float32)
        self.hidden_state = tf.Variable(hidden_state, dtype=tf.float32)
    def forget_gate(self, input_tensor, hidden_state):
        self.input_weights = tf.Variable(tf.random.normal(shape=(hidden_state, input_tensor)))
        self.latent_weights = tf.Variable(tf.random.normal(shape=(hidden_state, hidden_state)))
        return tf.sigmoid(tf.linalg.matvec(self.input_weights, input_tensor) + tf.linalg.matvec(self.latent_weights, hidden_state) + self.bias)
    def input_gate(self, input_tensor, hidden_state):
        self.input_weights = tf.Variable(tf.random.normal(shape=(hidden_state, input_tensor)))
        self.latent_weights = tf.Variable(tf.random.normal(shape=(hidden_state, hidden_state)))
        return tf.sigmoid(tf.linalg.matvec(self.input_weights, input_tensor) + tf.linalg.matvec(self.latent_weights, hidden_state) + self.bias)
    def candidate_cell_state(self, input_tensor, hidden_state):
        self.input_weights = tf.Variable(tf.random.normal(shape=(hidden_state, input_tensor)))
        self.latent_weights = tf.Variable(tf.random.normal(shape=(hidden_state, hidden_state)))
        return tf.tanh(tf.linalg.matvec(self.input_weights, input_tensor) + tf.linalg.matvec(self.latent_weights, hidden_state) + self.bias)
    def cellstate():
        return forget_gate * cell_state + input_gate * candidate_cell_state
        
