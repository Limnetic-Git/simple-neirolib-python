import math, random

class Neiro:
    def __init__(self, input_size, hidden_layers, output_size):
        self.weights = []
        self.biases = []
        prev = input_size
        for h in hidden_layers + [output_size]:
            w = [[random.gauss(0, math.sqrt(2.0/prev)) for _ in range(h)] for _ in range(prev)]
            b = [0.0] * h
            self.weights.append(w)
            self.biases.append(b)
            prev = h
        self.activation_hidden = lambda x: x if x > 0 else 0
        self.derivative_hidden = lambda x: 1 if x > 0 else 0

        self.activation_output = lambda x: x
        self.derivative_output = lambda x: 1

    def forward(self, x):
        self.a = [x]
        for w, b in zip(self.weights, self.biases):
            z = []
            for j in range(len(w[0])):
                s = b[j]
                for i in range(len(w)):
                    s += w[i][j] * x[i]
                z.append(s)
            x = [self.activation_hidden(v) for v in z]
            self.a.append(x)
        output = self.a[-1]
        final = [self.activation_output(v) for v in output]
        self.a[-1] = final
        return final

    def backward(self, target, lr=0.01):
        output = self.a[-1]
        delta = [(output[i] - target[i]) * self.derivative_output(output[i]) for i in range(len(output))]

        deltas = [None] * len(self.weights)
        deltas[-1] = delta

        for layer in range(len(self.weights) - 2, -1, -1):
            prev_activations = self.a[layer]
            w = self.weights[layer]
            next_delta = deltas[layer + 1]

            new_delta = [0.0] * len(prev_activations)
            for i in range(len(prev_activations)):
                error = 0.0
                for j in range(len(next_delta)):
                    error += next_delta[j] * w[i][j]
                new_delta[i] = error * self.derivative_hidden(prev_activations[i])
            deltas[layer] = new_delta

        for layer in range(len(self.weights) - 1, -1, -1):
            prev_activations = self.a[layer]
            w = self.weights[layer]
            delta = deltas[layer]

            for i in range(len(w)):
                for j in range(len(w[i])):
                    w[i][j] -= lr * delta[j] * prev_activations[i]

            for j in range(len(self.biases[layer])):
                self.biases[layer][j] -= lr * delta[j]
