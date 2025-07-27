import math, random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class Neiro:
    def __init__(self, input_layer, hidden_layers, output_layer_len):
        self.input_layer_len = len(input_layer)
        self.hidden_layers = hidden_layers
        self.output_layer_len = output_layer_len
        self.array = []
        
        self._init_array()
        self.generate_random_weights()
        self.array[0] = input_layer
        self.calculate_neirons()
            
        print(self.array)
        
    def _init_array(self):
        self._generate_layer(self.input_layer_len)
        for i in range(len(self.hidden_layers)):
            self._generate_weight_layer(self.hidden_layers[i])
            self._generate_layer(self.hidden_layers[i])
        self._generate_weight_layer(self.output_layer_len)
        self._generate_layer(self.output_layer_len)
        
    def _generate_layer(self, layer_len):
        self.array.append([0] * layer_len)
        
    def _generate_weight_layer(self, hidden_layer_len):
        new_layer = []
        for i in range(len(self.array[-1])):
            new_layer.append([])
        for i in range(len(new_layer)):
            for _ in range(hidden_layer_len):
                new_layer[i].append(0)
        self.array.append(new_layer)
        
    def generate_random_weights(self):
        for i in range(len(self.array)):
            if i % 2 != 0:
                for u in range(len(self.array[i])):
                    for j in range(len(self.array[i][u])):
                        self.array[i][u][j] = random.uniform(-1, 1)
                        
    def calculate_neirons(self):
        for layer_idx in range(2, len(self.array), 2): 
            current_layer = self.array[layer_idx]
            prev_layer = self.array[layer_idx - 2]
            weights = self.array[layer_idx - 1]
            
            for i in range(len(current_layer)): current_layer[i] = 0

            for prev_neuron_idx in range(len(prev_layer)):
                for current_neuron_idx in range(len(current_layer)):
                    current_layer[current_neuron_idx] += prev_layer[prev_neuron_idx] * weights[prev_neuron_idx][current_neuron_idx]

            for current_neuron_idx in range(len(current_layer)):
                current_layer[current_neuron_idx] = sigmoid(current_layer[current_neuron_idx])
                
    def backpropagate(self, target_output, learning_rate=0.1):
        if not isinstance(target_output, list):
            target_output = [target_output]

        deltas = [0] * len(self.array)
        
        output_layer_idx = len(self.array) - 1
        output_layer = self.array[output_layer_idx]
        deltas[output_layer_idx] = [0] * len(output_layer)
        
        for i in range(len(output_layer)):
            error = target_output[i] - output_layer[i]
            deltas[output_layer_idx][i] = error * sigmoid_derivative(output_layer[i])
        
        for layer_idx in range(output_layer_idx - 2, 0, -2):
            current_layer = self.array[layer_idx]
            next_layer = self.array[layer_idx + 2]
            weights = self.array[layer_idx + 1]
            deltas[layer_idx] = [0] * len(current_layer)
            
            for i in range(len(current_layer)):
                error = 0.0
                for j in range(len(next_layer)):
                    error += deltas[layer_idx + 2][j] * weights[i][j]
                deltas[layer_idx][i] = error * sigmoid_derivative(current_layer[i])
        
        for layer_idx in range(1, len(self.array), 2):
            prev_layer = self.array[layer_idx - 1]
            current_layer = self.array[layer_idx + 1]
            
            for i in range(len(prev_layer)):
                for j in range(len(current_layer)):
                    delta_weight = learning_rate * deltas[layer_idx + 1][j] * prev_layer[i]
                    self.array[layer_idx][i][j] += delta_weight
                    
#a = Neiro([2, 4], [728, 512, 728], 10)
