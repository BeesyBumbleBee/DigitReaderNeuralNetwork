import random
import numpy as np
import json
from sys import modules


class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a-y) ** 2

    @staticmethod
    def delta(z, a, y):
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(np.multiply(-y, np.log(a)) - np.multiply((1-y), np.log(1-a))))

    @staticmethod
    def delta(z, a, y):
        return (a-y)


class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.layers = sizes
        self.sizes = sizes
        self.cost = cost
        self.initialize_weights()
    
    def initialize_weights(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,
            evaluation_data=None,
            monitor_eval_cost=False,
            monitor_eval_acc=False,
            monitor_train_cost=False,
            monitor_train_acc=False):
        self.batch_size = mini_batch_size
        n_eval = 0
        if evaluation_data:
            n_eval = len(evaluation_data)
        n = len(training_data)

        eval_cost, eval_acc = [], []
        train_cost, train_acc = [], []
        for j in range(epochs):
            
            random.shuffle(training_data)
            mini_batches = []
            inputs = training_data[0][0]
            outputs = training_data[0][1]

            for i in range(1, n):
                if i % mini_batch_size == 0:
                    mini_batches.append((inputs, outputs))
                    inputs = training_data[i][0]
                    outputs = training_data[i][1]
                else:
                    inputs = np.concatenate((inputs, training_data[i][0]), axis=1)
                    outputs = np.concatenate((outputs, training_data[i][1]), axis=1)
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            print("\nEpoch {0:4d} complete".format(j))
            
            if monitor_train_cost:
                cost = self.total_cost(training_data, lmbda)
                train_cost.append(cost)
                print("Cost on training data:       {:4.5f}".format(cost))
            if monitor_train_acc:
                acc = self.accuracy(training_data, y_vector=True)
                train_acc.append(acc)
                print("Accuracy on training data:   {:8d} / {:8d}".format(acc, n))
            if monitor_eval_cost:
                cost = self.total_cost(evaluation_data, lmbda, y_vector=True)
                eval_cost.append(cost)
                print("Cost on evaluation data:     {:4.5f}".format(cost))
            if monitor_eval_acc:
                acc = self.accuracy(evaluation_data)
                eval_acc.append(acc)
                print("Accuracy on evaluation data: {:8d} / {:8d}".format(acc, n_eval))

        return eval_cost, eval_acc, train_cost, train_acc



    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nabla_b, nabla_w = self.backpropagate(mini_batch[0], mini_batch[1])
        self.weights = [(1 - eta * (lmbda/n)) * w - (eta / len(mini_batch)) * nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases =  [b - (eta / len(mini_batch)) * nb 
                        for b, nb in zip(self.biases, nabla_b)]

    def backpropagate(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        activation = x
        activations = [x]
        zvectors = []
        for b, w in zip(self.biases, self.weights):
            b = np.repeat(b, self.batch_size, axis=1)
            z = np.dot(w, activation) + b
            zvectors.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = (self.cost).delta(zvectors[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for layer in range(2, self.num_layers):
            z = zvectors[-layer]
            sp = sigmoid_prime(z)
            delta = np.multiply(np.dot(self.weights[-layer + 1].transpose(), delta), sp)
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())
        for i, layer in enumerate(nabla_b):
            nabla_b[i] = np.sum(layer, axis=1)
        return (nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

    def accuracy(self, data, y_vector=False):
        if y_vector:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)
    
    def total_cost(self, data, lmbda, y_vector=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if y_vector: y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
        cost += 0.5*(lmbda / len(data)) * sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def load(self, weights, biases):
        self.biases = biases
        self.weights = weights

    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

def load_network(filename, layers) -> Network:
    try:
        f = open(filename, "r")
        print(f"Loading saved network: {filename}");
        data = json.load(f)
        f.close()
        cost = getattr(modules[__name__], data["cost"])
        net = Network(data["sizes"], cost=cost)
        net.load([np.array(w) for w in data["weights"]], 
                 [np.array(b) for b in data["biases"]])
    except FileNotFoundError:
        print(f"Creating network: {filename}");
        net = Network(layers)
        pass
    return net


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
        return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
