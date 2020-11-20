import matplotlib.pyplot as plt
import numpy as np
import random as random

class Layer:
    def __init__(self, weights, activationFunction):
        self.nodes = []
        self.activationFunction = activationFunction

        for i in range(weights):
            nodes.append(ComplexNode(weights[i]))

class InputLayer:
    def __init__(self, x):
        self.nodesList = []

        for i in range(x):
            nodes = []
            for j in range(len(x[i] -1)):
                nodes.append(Node(x[i][j]))

            nodesList.append(nodes)



class Node:

    def __init__(self, value, layerIndex):
        self.value = value
        self.layerIndex = layerIndex

    def getValue(self):
        return self.value


class ComplexNode(Node):
    def __init__(self, weight, layerIndex):
        super(ComplexNode, self).__init__(0)
        self.b = random.random(0, 1)
        self.weight = weight
        self.layerIndex = layerIndex

    def forward(self):
        pass

    def backprop(self):
        pass


class MLP:
    def __init__(self):
        self.layers = []
        self.layerIndex = 0

    def addInputLayer(self, inputs, activationFunction):
        nodes = []
        for i in range(len(inputs)):
            nodes.append(Node(inputs[i], self.layerIndex))

        self.layers.append(Layer(nodes, activationFunction))
        self.layerIndex+=1

    def addComplexLayer(self, noNodes, activationFunction):
        nodes = []
        for i in range(len(noNodes)):
            nodes.append(ComplexNode(random.random(0, 1), self.layerIndex))

        self.layers.append(Layer(Nodes, activationFunction))
        self.layerIndex += 1

    def addOutputLayer(self, classes, activationFunction):
        nodes = []
        for i in range(len(classes)):
            nodes.append(ComplexNode(random.random(0, 1), self.layerIndex))

        self.layers.append(Layer(Nodes, activationFunction))
        self.layerIndex += 1

    def _backprop(self, x, y, d_loss, learning_rate):
        pass

    def train(self, x, y, learning_rate=0.01, n_epochs=100, noHiddenLayers, noNodes):
        self.addInputLayer(x, activationFunction)
        for i in range(noHiddenLayers):
            self.addComplexLayer(noNodes, sig()) #SIG HÄR

        self.addOutputLayer(1, lin()) #LIN HÄR

        for i in range(n_epochs):
            for j in range(len(x)):
                self.layers[0].nodesList[j]


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