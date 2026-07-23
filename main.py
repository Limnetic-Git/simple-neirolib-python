import numpy as np
import json

class Neiro:
    def __init__(self, input_size, hidden_layers, output_size):
        self.weights = []
        self.biases = []
        prev = input_size
        for h in hidden_layers + [output_size]:
            w = np.random.randn(prev, h) * np.sqrt(2.0 / prev)
            b = np.zeros(h)
            self.weights.append(w)
            self.biases.append(b)
            prev = h

        self.activation_hidden = lambda x: np.maximum(x, 0)
        self.derivative_hidden = lambda x: (x > 0).astype(float)

        self.activation_output = lambda x: x
        self.derivative_output = lambda x: np.ones_like(x)

    def forward(self, x):
        self.a = [np.array(x, dtype=float)]

        for w, b in zip(self.weights, self.biases):
            z = np.dot(self.a[-1], w) + b
            x = self.activation_hidden(z)
            self.a.append(x)
        output = self.a[-1]
        final = self.activation_output(output)
        self.a[-1] = final
        return final

    def backward(self, target, lr=0.01):
        target = np.array(target, dtype=float)
        output = self.a[-1]
        delta = (output - target) * self.derivative_output(output)
        deltas = [delta]

        for layer in range(len(self.weights) - 1, -1, -1):
            w = self.weights[layer]
            if layer > 0:
                error = np.dot(w, deltas[-1])
                deltas.append(error * self.derivative_hidden(self.a[layer]))

        deltas = deltas[::-1]
        for layer in range(len(self.weights) - 1, -1, -1):
            self.weights[layer] -= lr * np.outer(self.a[layer], deltas[layer])
            self.biases[layer] -= lr * deltas[layer]


    def save_params(self, save_name="params0.json"):
        weights_list = [w.tolist() for w in self.weights]
        biases_list = [b.tolist() for b in self.biases]
        with open(save_name, 'w') as file:
            json.dump({
                'weights': weights_list,
                'biases': biases_list
            }, file)
        print('Сохранено')

    def load_params(self, load_name="params0.json"):
        with open(load_name, 'r') as file:
            data = json.load(file)
        self.weights = [np.array(w) for w in data['weights']]
        self.biases = [np.array(b) for b in data['biases']]
        print('Загружено')
