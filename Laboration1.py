import matplotlib.pyplot as plt
import numpy as np
import random as random


class Layer:
    def __init__(self, layer_size, activation_function):
        self.nodes = []
        self.acivation_function = activation_function
        for i in range(layer_size):
            nodes.add(Node(layer_size))

class Node:

    def __init__(self, value):
        self.value = value

    def getValue(self):
        return self.value

class ComplexNode(Node):
    def __init__(self, no_weights):
        self.b = random(0, 1)
        self.weights = []
        for i in range(no_weights - 1):
            self.weights.add(random(0, 1))

    def forward(self):
        pass

    def backprop(self):
        pass

class MLP:
    def __init__(self, loss_function, no_layers, inputs):
        self.layers = []




        for i in range(no_layers - 1):
            self.layers.add(Layer(no_layerNodes, no_weights))

    def addInputLayer(self, inputs):
        for i in range(inputs -1):
            self.layers[0].nodes.append(Node(inputs[i]))


    def add_complexlayer(self, size, activation_function):
        self.layers.append(Layer(size, activation_function))

    def _backprop(self, x, y, d_loss, learning_rate):
        pass

    def train(self, x, y, learning_rate=0.01, n_epochs=100):
        pass

    def predict(self, x, y):
        sigmoid = lambda z: 1 / (1 + np.e ** (-z))

            # class Layer:
            #     def __init__(self, n_nodes, n_inputs):
            #         pass

            #     def forward(self, x):
            #         return lambda z: 1 / (1 + np.e ** (-x))

            #     def backprop(self, loss, learning_rate):
            #         pass

class LossFunction:
    @staticmethod
    def forward(z):
        pass
    @staticmethod
    def backward(z):
        pass
class ActivationFunction:
    @staticmethod
    def forward(z):
        pass
    @staticmethod
    def backward(z):
        pass