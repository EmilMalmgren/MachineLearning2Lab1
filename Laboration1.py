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

class ActivationFunction:
    @staticmethod
    def forward(z): pass

    @staticmethod
    def backward(z): pass

class LinearActivationFunction(ActivationFunction):
    @staticmethod
    def forward(z): return z

    @staticmethod
    def backward(z): return np.ones(z.size)

class SigmoidActivationFunction(ActivationFunction):
    @staticmethod
    def forward(z): return 1 / (1 + np.e ** (-z))

    @staticmethod
    def backward(z): return z * (1 - z)

class Layer:
    def __init__(self, numberOfNodesPerHiddenLayer, activationFunction, numberOfWeights):
        self.nodes = []
        self.values = []

        for i in range(numberOfNodesPerHiddenLayer):
            self.nodes.append(ComplexNode(numberOfWeights,activationFunction))


class InputLayer:
    def __init__(self):
        self.nodes = []

    def addData(self, x):
        for i in range(x):
            nodes = []
            for j in range(len(x[i] -1)):
                nodes.append(Node(x[i][j]))

            self.nodes.append(nodes)


class Node:

    def __init__(self, value):
        self.value = value

    def getValue(self):
        return self.value


class ComplexNode(Node):
    def __init__(self, numberOfWeights,activationFunction):
        self.activationFunction = activationFunction
        self.dLoss = 0
        self.dw = []
        self.db = 0
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
        self.value = self.activationFunction.forward(z)


    def backprop(self, values):
        for i in range(self.weights):
            self.dw.append(self.dLoss*self.activationFunction.backward(self.value)*values[i])
        self.db = self.dLoss*self.activationFunction.backward(self.value)

    def updateDLoss(self, dbs, weights):
        sum = 0
        for i in range(dbs):
            sum+= dbs[i]*weights[i]
        self.dLoss = sum

    def updateOutputDLoss(self,actualValue):
        self.dLoss = 2* (actualValue - self.value)

    def getDb(self):
        return self.db

    def getWeights(self):
        return self.weights

    def updateWeightsAndB(self, lRate):
        self.weights -= self.dw *lRate
        self.b -= self.db * lRate


class MLP:
    def __init__(self, data, numberOfHiddenLayers, numberOfNodesPerHiddenLayer):
        self.layers = []
        self.targets = []

        for i in range(len(data)):
            self.targets.append(data[i][0].pop())

        self.data = data
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
            for row in self.layers[0].nodes:
                for j in range(1, layers):
                    values = []
                    for previousNode in layers[j - 1].nodes:
                        values.append(previousNode.getValue())
                    layers[j].values = values
                    for node in layers[j].nodes: #för varje nod i första hidden layer
                        node.forward(values)

                layers[len(layers)-1][0].updateOutputDLoss(self.targets[row])
                layers[len(layers) - 1][0].backprop(values)

                for layer in range(len(layers)-2,1,-1):
                    dbs = []
                    weights = []
                    for previousNode in layers[layer - 1].nodes:
                        dbs.append(previousNode.getDb())
                        weights.append(previousNode.getWeights())
                        previousNode.updateWeightsAndB(learning_rate)

                    for node in range(layers[layer].nodes): #för varje nod i första hidden layer
                        node.updateDLoss(dbs[node],weights[node])
                        node.backprop(values)


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


