import warnings
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
        self.cost_history = []
        self.acc_history = []
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
            train_hyper_param=False,
            monitor_eval_cost=False,
            monitor_eval_acc=False,
            monitor_train_cost=False,
            monitor_train_acc=False):
        self.batch_size = mini_batch_size
        self.eta = eta
        self.lmbda = lmbda
        if train_hyper_param:
            self.init_hyper_param_training()
        n_eval = 0
        if evaluation_data:
            n_eval = len(evaluation_data)
        n = len(training_data)

        train_cost, train_acc = [], []
        for j in range(epochs*10):
            print(f"eta={self.eta}, lmbda={self.lmbda}, batch_size={self.batch_size}")

            random.shuffle(training_data)
            mini_batches = []
            inputs = training_data[0][0]
            outputs = training_data[0][1]

            for i in range(1, n):
                if i % self.batch_size == 0:
                    mini_batches.append((inputs, outputs))
                    inputs = training_data[i][0]
                    outputs = training_data[i][1]
                else:
                    inputs = np.concatenate((inputs, training_data[i][0]), axis=1)
                    outputs = np.concatenate((outputs, training_data[i][1]), axis=1)
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, self.eta, self.lmbda, len(training_data))
            print("\nEpoch {0:4d} complete".format(j))
            
            if monitor_train_cost:
                cost = self.total_cost(training_data, self.lmbda)
                train_cost.append(cost)
                print("Cost on training data:       {:4.5f}".format(cost))
            if monitor_train_acc:
                acc = self.accuracy(training_data, y_vector=True)
                print("Accuracy on training data:   {:8d} ({:+4d}) / {:8d}".format(acc, acc - max(train_acc) if len(train_acc) != 0 else 0, n))
                train_acc.append(acc)
            if monitor_eval_cost or train_hyper_param:
                cost = self.total_cost(evaluation_data, self.lmbda, y_vector=True)
                self.cost_history.append(cost)
                print("Cost on evaluation data:     {:4.5f}".format(cost))
            if monitor_eval_acc or train_hyper_param:
                acc = self.accuracy(evaluation_data)
                print("Accuracy on evaluation data: {:8d} ({:+4d}) / {:8d}".format(acc, acc - max(self.acc_history) if len(self.acc_history) != 0 else 0, n_eval))
                self.acc_history.append(acc)

            if train_hyper_param:
                if self.train_hyper_parameters(j, n_eval):
                    break

        return self.acc_history, self.cost_history, train_cost, train_acc



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

    def init_hyper_param_training(self):
        self.count = 0
        self.best_in = 10
        self.best_acc = 0
        self.eta = 0.001
        self.lmbda = 0.01
        self.batch_size = 10

    def train_hyper_parameters(self, epoch, n_eval):
        if epoch > 1 and self.acc_history[-2] - self.acc_history[-1] > 0.001 * n_eval:
            print(f"Lowered eta from {self.eta} to ", end='')
            self.eta /= 2
            print(self.eta)
            self.changed_eta = True
            return False

        if self.best_acc < self.acc_history[-1]:
            self.count = 0
            self.best_acc = self.acc_history[-1]
        else:
            self.count += 1
        if self.count == self.best_in / 2:
            self.lmbda /= 2
            self.batch_size = round(self.batch_size / 2, 0) 
        if self.count == self.best_in:
            return True

        return False
        
    
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

def nowarning(func):
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return wrapper

@nowarning
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

