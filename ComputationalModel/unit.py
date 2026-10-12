import tensorflow as tf


class ComputationalUnit:
    def __init__(self, input_n):
        self.weight_tensor = tf.Variable(tf.random.normal(shape=(input_n,)))
        self.bias = tf.Variable(tf.random.normal(shape=()))

    def output(self, input_tensor):
        linear_preactivation = tf.tensordot(input_tensor, self.weight_tensor, axes=1) + self.bias
        #linear_preactivation = tf.add(tf.tensordot(input_tensor, self.weight_tensor, axes=1), self.bias)
        nonlinear_activation = tf.nn.relu(linear_preactivation)
        return nonlinear_activation


class ComputationalLayer:
    def __init__(self, units_num, input_n):
        self.units = [ComputationalUnit(input_n) for _ in range(units_num)]

    def activation(self, input_tensor):
        return tf.stack([unit.output(input_tensor) for unit in self.units])


class FFNN:
    def __init__(self, input_n, layer_activation_sizes):
        self.layers = []
        current_input_n = input_n
        for units_num in layer_activation_sizes:
            self.layers.append(ComputationalLayer(units_num, current_input_n))
            current_input_n = units_num
    def output(self, input_tensor):
        activation = input_tensor
        for layer in self.layers:
            activation = layer.activation(activation)
        return activation

    
    