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

        self.mean.append(np.mean(y))
        self.std.append(np.std(y))
        self.mean = np.array(self.mean)
        self.std = np.array(self.std)

    def normalize(self, x, y):
        return (x - self.mean[:-1]) / self.std[:-1], (y - self.mean[-1]) / self.std[-1]

    def renormalize(self, y):
        return y * self.std[-1] + self.mean[-1]

class LinearActivationFunction:
    @staticmethod
    def forward(z): return z

    @staticmethod
    def backward(z): return 1

class SigmoidActivationFunction:
    @staticmethod
    def forward(z): return 1 / (1 + np.e ** (-z))

    @staticmethod
    def backward(z): return z * (1 - z)


class SquaredError(object):
    @staticmethod
    def squared_error_forward(expected, actual):
        return (expected - actual)**2

    @staticmethod
    def squared_error_backward(expected, actual):
        return 2 * (expected - actual)

class MLP:
    def __init__(self, n_nodes, n_hidden_layers):
        self.layers = []
        self.n_nodes = n_nodes
        self.n_hidden_layers = n_hidden_layers

    def add_layer(self, n_nodes, n_inputs, activation_function):
        self.layers.append(Layer(n_nodes, n_inputs, activation_function))


    def _backprop(self, x, y, d_loss, learning_rate):
        pass

    def initialize_network(self, x):
        for i in range(self.n_hidden_layers):
            if i == 0:
                self.add_layer(self.n_nodes, x, SigmoidActivationFunction())
            else:
                self.add_layer(self.n_nodes, self.n_nodes, SigmoidActivationFunction())
        self.add_layer(1, self.n_nodes, LinearActivationFunction())

    def train(self, x, y, learning_rate=0.01, n_epochs=200, decay = 9):
        self.initialize_network(len(x[0]))
        for i in range(n_epochs):
            sum_error = 0
            for j in np.random.permutation(len(x)):

                #forward
                output = self.forward(x[j]) #kan va fel med ettorna? första

                #backward
                #um_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
                sum_error += (y[j] - output[0])**2
                self.backward(y[j])
                self.update_weights(x[j], learning_rate)
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (i,  learning_rate, sum_error / len(x)))
            learning_rate = learning_rate * (1 / (1 + decay))
            learning_rate = max(learning_rate, 0.01)
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

    def predict(self, x):
        results = []
        for i in x:
            results.append(normalizer.renormalize(self.forward(i)[0]))
        return results

#learning rate decay
#plotta
#loss över epoker, target mot prediktion x mot y, mse på testmängd för modell

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
                weights.append(np.random.uniform(-0.5, 0.5))
            self.nodes.append(weights)


    def forward(self, row):
        pass


    def backprop(self, loss, learning_rate):
        pass

    def calculate(self, weights, inputs):
        output = weights[-1]
        for i in range(len(weights)-1):
            output += weights[i] * inputs[i]
        return self.activation_function.forward(output)

def run():
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

    normalizer.fit(x, y)
    x, y = normalizer.normalize(x, y)
    x = x.tolist()
    y = y.tolist()
    mlp = MLP(5, 2)
    print(str(x[-1]))
    print(str(y[-1]))
    mlp.train(x[200:], y[200:], 0.01, 200, 0.05)
    results = mlp.predict(x[:199])
    a = 1

normalizer = Normalizer()
run()