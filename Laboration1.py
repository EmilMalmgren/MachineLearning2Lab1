import matplotlib.pyplot as plt
import numpy as np
import random as r
import csv
import os
from math import exp

class Normalizer:
    def __init__(self):
        self.mean = []
        self.std = []

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        for i in range(x.shape[1]):
            self.mean.append(np.mean(x[:, i]))
            self.std.append(np.std(x[:, i]))

<<<<<<< HEAD
        self.mean.append(np.mean(y))
        self.std.append(np.std(y))
        self.mean = np.array(self.mean)
        self.std = np.array(self.std)
=======
class ActivationFunction:
    def __init__(self):
        pass

    @staticmethod
    def forward(z): pass
>>>>>>> b5182f381c72924ee9807330625107ab197fdbfd

    def normalize(self, x, y):
        return (x - self.mean[:-1]) / self.std[:-1], (y - self.mean[-1]) / self.std[-1]

    def renormalize(self, y):
        return y * self.std[-1] + self.mean[-1]

class LinearActivationFunction:
    @staticmethod
    def forward(z): return z

    @staticmethod
<<<<<<< HEAD
    def backward(z): return z * (1 - z) #np.ones(z)
=======
    def backward(z): return np.ones(1)
>>>>>>> b5182f381c72924ee9807330625107ab197fdbfd

class SigmoidActivationFunction:
    @staticmethod
    def forward(z): return 1 / (1 + np.e ** (-z))

    @staticmethod
    def backward(z): return z * (1 - z)

<<<<<<< HEAD

class SquaredError(object):
    @staticmethod
    def squared_error_forward(expected, actual):
        return (expected - actual)**2
=======
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
>>>>>>> b5182f381c72924ee9807330625107ab197fdbfd

    @staticmethod
    def squared_error_backward(expected, actual):
        return 2 * (expected - actual)

class MLP:
    def __init__(self, n_nodes, n_hidden_layers):
        self.layers = []
<<<<<<< HEAD
        self.n_nodes = n_nodes
        self.n_hidden_layers = n_hidden_layers
=======
        self.addInputLayer()
        self.addComplexLayer(numberOfNodesPerHiddenLayer, SigmoidActivationFunction, len(data[0])-1)
        for i in range(numberOfHiddenLayers -1):
            self.addComplexLayer(numberOfNodesPerHiddenLayer, SigmoidActivationFunction, numberOfNodesPerHiddenLayer)

        self.addOutputLayer(1, LinearActivationFunction, numberOfNodesPerHiddenLayer)
>>>>>>> b5182f381c72924ee9807330625107ab197fdbfd

    def add_layer(self, n_nodes, n_inputs, activation_function):
        self.layers.append(Layer(n_nodes, n_inputs, activation_function))

    def train(self, data, learning_rate=0.01, n_epochs=100):
        targets = []
        for i in range(len(data)):
            targets.append(data[i].pop(0))

    def _backprop(self, x, y, d_loss, learning_rate):
        pass

    def initialize_network(self, x):
        for i in range(self.n_hidden_layers):
            if i == 0:
                self.add_layer(self.n_nodes, x, SigmoidActivationFunction())
            else:
                self.add_layer(self.n_nodes, self.n_nodes, SigmoidActivationFunction())
        self.add_layer(1, self.n_nodes, LinearActivationFunction())

    def train(self, x, y, learning_rate=0.01, n_epochs=100):
        self.initialize_network(len(x[0]))
        for i in range(n_epochs):
            sum_error = 0
            for j in range(len(x)):
                #forward
                output = self.forward(x[j]) #kan va fel med ettorna? första

                #backward
                #um_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
                sum_error += (y[j] - output[0])**2
                self.backward(y[j])
                self.update_weights(x[j], learning_rate)
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (i,  learning_rate, sum_error))
    def update_weights(self, row, l_rate):
        for i in range(len(self.layers)):
            inputs = row
            if i != 0:
                inputs = self.layers[i - 1].outputs
            for j in range(len(self.layers[i].nodes)):
                for k in range(len(inputs)):
                    self.layers[i].nodes[j][k] += l_rate *  self.layers[i].dloss[j] * inputs[k]
                self.layers[i].nodes[j][-1] += l_rate * self.layers[i].dloss[j]

    def forward(self, row):
        inputs = row
        for layer in self.layers:
            output = []
            for i in range(len(layer.nodes)):
                output.append(layer.calculate(layer.nodes[i], inputs))
            inputs = output
            layer.outputs = output
        return inputs

    def backward(self, expected):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            layer.dloss = []
            errors = list()
            if i != len(self.layers) - 1:
                for j in range(len(layer.nodes)):
                    error = 0.0
                    for k in range(len(self.layers[i + 1].nodes)):
                        for l in range(len(self.layers[i + 1].nodes[k])-1):
                            error += (self.layers[i + 1].nodes[k][l] * self.layers[i + 1].dloss[k])
                    errors.append(error)
            else:
                for j in range(len(layer.outputs)):
                    errors.append(SquaredError.squared_error_backward(expected, layer.outputs[j]))
            for j in range(len(layer.outputs)):
                layer.dloss.append(errors[j] * layer.activation_function.backward(layer.outputs[j]))

    def predict(self, x, y):
        pass


class Layer:
    def __init__(self, n_nodes, n_inputs, activation_function):
        self.activation_function = activation_function
        self.nodes = []
        #[1, 2, 3], [1, 2, 3]
        self.outputs = []
        #[len(nodes) lång
        self.dloss = []
        for i in range(n_nodes):
            weights = []
            for j in range(n_inputs + 1):
                weights.append(r.random())
            self.nodes.append(weights)


    def forward(self, row):
        pass


    def backprop(self, loss, learning_rate):
        pass

<<<<<<< HEAD
    def calculate(self, weights, inputs):
        output = weights[-1]
        for i in range(len(weights)-1):
            output += weights[i] * inputs[i]
        return self.activation_function.forward(output)



x = []
y = []
with open("C:\\Users\erik\PycharmProjects\mllabb22\\venv\datasets\\boston.csv", 'r') as file:
    reader = csv.reader(file)
    for rowa in reader:
        datasetRow = []
        for value in range(len(rowa)):
            if value == 0:
                y.append(float(rowa[value]))
                continue
            else:
                datasetRow.append(float(rowa[value]))
        x.append(datasetRow)

normalizer = Normalizer()
normalizer.fit(x, y)
x, y = normalizer.normalize(x, y)
x = x.tolist()
y = y.tolist()
MLP = MLP(5, 2)
MLP.train(x, y, 0.1, 100)
a = 1

=======
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
>>>>>>> b5182f381c72924ee9807330625107ab197fdbfd
