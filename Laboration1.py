import matplotlib.pyplot as plt
import numpy as np
import random as random
import csv
import os

class LossFunction:
    @staticmethod
    def forward(predictions, correct_outputs):
        return (predictions - correct_outputs) ** 2

    @staticmethod
    def backward(predictions, correct_outputs):
        return 2 * (predictions - correct_outputs)

class LinearActivationFunction:
    @staticmethod
    def forward(z): return z

    @staticmethod
    def backward(z): return np.ones(z.size)

class SigmoidActivationFunction:
    @staticmethod
    def forward(z): return 1 / (1 + np.e ** (-z))

    @staticmethod
    def backward(z): return z * (1 - z)

class Layer:
    def __init__(self, numberOfNodesPerHiddenLayer, activationFunction, numberOfWeights):
        self.nodes = []
        self.activationFunction = activationFunction

        for i in range(numberOfNodesPerHiddenLayer):
            self.nodes.append(ComplexNode(numberOfWeights))


class InputLayer:
    def __init__(self):
        self.nodes = []

    def addData(self, x):
        for i in range(x):
            nodes = []
            for j in range(len(x[i] -1)):
                nodes.append(Node(x[i][j]))

            self.nodes.append(nodes)

    def forward(self, z):
        LinearActivationFunction.forward(z)

    def backprop(self, z):
        LinearActivationFunction.backward(z)

class Node:

    def __init__(self, value):
        self.value = value

    def getValue(self):
        return self.value


class ComplexNode(Node):
    def __init__(self, numberOfWeights):
        super(ComplexNode, self).__init__(0)
        self.b = random.random(0, 1)
        self.weights = []
        for i in range(numberOfWeights):
            self.weights.append(random.random(0, 1))

    def forward(self, values):
        z = 0
        for k in range(values):
            z += (values[k] * self.weights[k])
        z += b
        self.value = SigmoidActivationFunction.forward(z)


    def backprop(self, z):
        pass


class MLP:
    def __init__(self, data, numberOfHiddenLayers, numberOfNodesPerHiddenLayer):
        self.data = data
        self.layers = []

        self.addInputLayer()
        self.addComplexLayer(numberOfNodesPerHiddenLayer, SigmoidActivationFunction, len(data[0]))
        for i in range(numberOfHiddenLayers -1):
            self.addComplexLayer(numberOfNodesPerHiddenLayer, SigmoidActivationFunction, numberOfNodesPerHiddenLayer)

        self.addOutputLayer(1, LinearActivationFunction, numberOfNodesPerHiddenLayer)

    def addInputLayer(self):
        self.layers.append(InputLayer())

    def addComplexLayer(self, numberOfNodesPerHiddenLayer, activationFunction, numberOfWeights):
        self.layers.append(Layer(numberOfNodesPerHiddenLayer, activationFunction, numberOfWeights))

    def addOutputLayer(self, classes, activationFunction, numberOfNodesPerHiddenLayer):
        self.layers.append(Layer(classes, activationFunction, numberOfNodesPerHiddenLayer))


    def _backprop(self, x, y, d_loss, learning_rate):
        pass

    def train(self, x, y, learning_rate=0.01, n_epochs=100):
        self.layers[0].addData(x)
        for i in range(n_epochs):
            #forward
            for j in range(1, layers):
                values = []
                for previousNode in layers[j - 1].nodes:
                    values.append(previousNode.getValue())

                for node in layers[j].nodes: #för varje nod i första hidden layer
                    node.forward(values)

            #loss
            layers[len(layers -1)]


            #backward
            for j in range(len(layers), 1, -1):



    def predict(self, x, y):
        sigmoid = lambda z: 1 / (1 + np.e ** (-z))

            # class Layer:
            #     def __init__(self, n_nodes, n_inputs):
            #         pass

            #     def forward(self, x):
            #         return lambda z: 1 / (1 + np.e ** (-x))

            #     def backprop(self, loss, learning_rate):
            #         pass



dataset = []
with open("C:\\Users\erik\PycharmProjects\ML2Labb1\\venv\datasets\\boston.csv", 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        dataset.append(row)

numberOfHiddenLayers = 2
numberOfNodesPerHiddenLayer = 4
mlp = MLP(dataset, numberOfHiddenLayers, numberOfNodesPerHiddenLayer)


