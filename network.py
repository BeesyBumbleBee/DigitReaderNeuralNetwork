import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.float128(np.random.randn(y, 1)) for y in sizes[1:]]
        self.weights = [np.float128(np.random.randn(y, x)) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        self.batch_size = mini_batch_size
        n_test = 0
        if test_data:
            n_test = len(test_data)
        n = len(training_data)

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
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0:4d}: {1}\t/ {2}". format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0:4d} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b, nabla_w = self.backpropagate(mini_batch[0], mini_batch[1])
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

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
        delta = np.multiply(self.cost_derivative(activations[-1], y), sigmoid_prime(zvectors[-1]))
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

    def load_network(self, weights, biases):
        self.biases = biases
        self.weights = weights

    def get_network(self):
        return (self.weights, self.biases)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
        return np.multiply(sigmoid(z), (1 - sigmoid(z)))
