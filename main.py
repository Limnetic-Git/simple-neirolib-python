import math, random

def relu(x):
    return x if x > 0 else 0

def relu_derivative(x):
    return 1 if x > 0 else 0

class Neiro:
    def __init__(self, input_size, hidden_layers, output_size):
        self.weights = []
        self.biases = []
        prev = input_size
        for i, h in enumerate(hidden_layers + [output_size]):
            w = [[random.gauss(0, math.sqrt(2.0/prev)) for _ in range(h)] for _ in range(prev)]
            b = [0.0] * h
            self.weights.append(w)
            self.biases.append(b)
            prev = h

    def forward(self, x):
        self.a = [x]
        for w, b in zip(self.weights, self.biases):
            z = []
            for j in range(len(w[0])):
                s = b[j]
                for i in range(len(w)):
                    s += w[i][j] * x[i]
                z.append(s)
            x = [relu(v) for v in z]
            self.a.append(x)
        return x

    def backward(self, target, lr=0.01):
        output = self.a[-1]
        delta = [output[i] - target[i] for i in range(len(output))]

        for layer in range(len(self.weights) - 1, -1, -1):
            prev_activations = self.a[layer]
            w = self.weights[layer]
            for i in range(len(w)):
                for j in range(len(w[i])):
                    w[i][j] -= lr * delta[j] * prev_activations[i]

            for j in range(len(self.biases[layer])):
                self.biases[layer][j] -= lr * delta[j]

            if layer > 0:
                new_delta = [0.0] * len(self.weights[layer-1][0])
                for i in range(len(new_delta)):
                    for j in range(len(delta)):
                        new_delta[i] += delta[j] * w[i][j]
                    new_delta[i] *= relu_derivative(self.a[layer][i])
                delta = new_delta

net = Neiro(2, [4], 1)

for epoch in range(10_000):
    x = [random.randint(0, 1), random.randint(0, 1)]
    y = [1 if x[0] != x[1] else 0]

    out = net.forward(x)
    net.backward(y, lr=0.1)

for test in [[0,0], [0,1], [1,0], [1,1]]:
    out = net.forward(test)
    print(f"{test} -> {out[0]:.4f}")
