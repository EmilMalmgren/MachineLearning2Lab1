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
    def __init__(self):
        pass

    @staticmethod
    def forward(z): pass

    @staticmethod
    def backward(z): pass

class LinearActivationFunction(ActivationFunction):
    @staticmethod
    def forward(z): return z

    @staticmethod
    def backward(z): return np.ones(1)

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
            self.nodes.append(ComplexNode(numberOfWeights, activationFunction))


class InputLayer:
    def __init__(self):
        self.nodes = []

    def addData(self, x):
        for i in range(len(x)):
            nodes = []
            for j in range(len(x[i])):
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
        self.b = random.random()
        self.weights = []
        for i in range(numberOfWeights):
            self.weights.append(random.random())

    def forward(self, values):
        z = 0
        for k in range(len(values)):
            z += (float(values[k]) * self.weights[k])
        z += self.b
        self.value = self.activationFunction.forward(z)

    def backprop(self, values):
        dws = []
        for i in range(len(self.weights)):
            dws.append(self.dLoss * self.activationFunction.backward(self.value)*values[i])
        self.dw = dws
        self.db = self.dLoss * self.activationFunction.backward(self.value)

    def updateDLoss(self, dbs, weights):
        sum = 0
        for i in range(dbs):
            sum += dbs[i] * weights[i]
        self.dLoss = sum

    def updateOutputDLoss(self, actualValue):
        self.dLoss = 2 * (float(actualValue) - self.value)

    def getDb(self):
        return self.db

    def getWeights(self):
        return self.weights

    def updateWeightsAndB(self, lRate):
        self.weights -= float(self.dw) * lRate
        self.b -= self.db * lRate


class MLP:
    def __init__(self, data, numberOfHiddenLayers, numberOfNodesPerHiddenLayer):
        self.layers = []
        self.addInputLayer()
        self.addComplexLayer(numberOfNodesPerHiddenLayer, SigmoidActivationFunction, len(data[0])-1)
        for i in range(numberOfHiddenLayers -1):
            self.addComplexLayer(numberOfNodesPerHiddenLayer, SigmoidActivationFunction, numberOfNodesPerHiddenLayer)

        self.addOutputLayer(1, LinearActivationFunction, numberOfNodesPerHiddenLayer)

    def addInputLayer(self):
        self.layers.append(InputLayer())

    def addComplexLayer(self, numberOfNodesPerHiddenLayer, activationFunction, numberOfWeights):
        self.layers.append(Layer(numberOfNodesPerHiddenLayer, activationFunction, numberOfWeights))

    def addOutputLayer(self, classes, activationFunction, numberOfNodesPerHiddenLayer):
        self.layers.append(Layer(classes, activationFunction, numberOfNodesPerHiddenLayer))

    def train(self, data, learning_rate=0.01, n_epochs=100):
        targets = []
        for i in range(len(data)):
            targets.append(data[i].pop(0))

        self.layers[0].addData(data)
        for i in range(n_epochs):
            #forward
            for row in range(len(self.layers[0].nodes)):
                values = []
                for node in self.layers[0].nodes[row]:
                    values.append(node.value)
                for node in self.layers[1].nodes:  # för varje nod i första hidden layer
                    node.forward(values)
                values = []

                for j in range(2, len(self.layers)):
                    for previousNode in self.layers[j -1].nodes:
                        values.append(previousNode.getValue())
                    self.layers[j].values = values
                    values = []
                    for node in self.layers[j].nodes: #för varje nod i första hidden layer
                        node.forward(values)

                #Calculate output loss
                self.layers[len(self.layers)-1].nodes[0].updateOutputDLoss(targets[row])
                self.layers[len(self.layers) - 1].nodes[0].backprop(self.layers[len(self.layers) -1].values)

                #backprop
                for layer in range(len(self.layers)-2, 1, -1):
                    dbs = []
                    weights = []
                    for previousNode in self.layers[layer - 1].nodes:
                        dbs.append(previousNode.getDb())
                        weights.append(previousNode.getWeights())
                        previousNode.updateWeightsAndB(learning_rate)

                    for node in range(self.layers[layer].nodes): #för varje nod i första hidden layer
                        node.updateDLoss(dbs[node], weights[node])
                        node.backprop(self.layers[layer + 1])

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
with open("C:/Users/EmilMa/Skolarbete/repos/MachineLearning2Lab1/datasets/boston.csv", 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        dataset.append(row)
mlp = MLP(dataset, 2, 4)

numberOfHiddenLayers = 2
numberOfNodesPerHiddenLayer = 4
mlp = MLP(dataset, numberOfHiddenLayers, numberOfNodesPerHiddenLayer)
mlp.train(dataset)
x =5
