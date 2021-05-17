import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        weight_shapes = [(a, b) for a, b in zip(layer_sizes[1:], layer_sizes[:-1])]
        self.weights = [np.random.standard_normal(s) / np.sqrt(s[1]) for s in weight_shapes]
        self.bias = [np.zeros((s, 1)) for s in layer_sizes[1:]]

        self.num_layers = len(layer_sizes)
        self.activations = np.asarray([np.zeros(size) for size in layer_sizes])




    def feedforward(self, sample):
        def activate(z):
            return 1 / (1 + np.exp(-z))
        
        self.activations[0] = sample

        for i in range(self.num_layers - 1):
            z = np.matmul(self.weights[i], self.activations[i]) + self.bias[i]
            self.activations[i + 1] = activate(z)

        return self.activations[-1]




    def train(self, images, labels, epochs, batchSize, learningRate):

        data = [(x, y) for x, y in zip(images, labels)]

        print("start training")
        for j in range(epochs):
            batches = [data[k:k + batchSize] for k in range(0, len(data), batchSize)]

            for batch in batches:
                self.update_batch(batch, learningRate)

            print("epoch {0} complete".format(j))




    def update_batch(self, batch, learningRate):
        bias_gradients = [np.zeros(b.shape) for b in self.bias]
        weight_gradients = [np.zeros(w.shape) for w in self.weights]

        for sample, label in batch:
            bias_deltas, weight_deltas = self.back_propagation(sample, label)

            bias_gradients = [b_gradient + b_delta for b_gradient, b_delta in zip(bias_gradients, bias_deltas)]
            weight_gradients = [w_gradient + w_delta for w_gradient, w_delta in zip(weight_gradients, weight_deltas)]

        self.bias = [b - (learningRate / len(batch)) * b_gradient for b, b_gradient in
                       zip(self.bias, bias_gradients)]
        self.weights = [w - (learningRate / len(batch)) * w_gradient for w, w_gradient in
                        zip(self.weights, weight_gradients)]




    def back_propagation(self, sample, label):
        def activateDerivative(a):
            return a * (1 - a)
        

        # def costFunction(output, y):
            # return sum((a - b) ** 2 for a, b in zip(output, y))[0]
        def costFunctionDerivative(output, y):
            return 2 * (output - y)

        bias_deltas = [np.zeros(b.shape) for b in self.bias]
        weight_deltas = [np.zeros(w.shape) for w in self.weights]

        self.feedforward(sample)

        L = -1

        partial_deltas = costFunctionDerivative(self.activations[L], label) * \
                         activateDerivative(self.activations[L])

        bias_deltas[L] = partial_deltas  
        weight_deltas[L] = np.dot(partial_deltas, self.activations[L - 1].T)

        while L > -self.num_layers + 1:
            previous_layer_deltas = np.dot(self.weights[L].T, partial_deltas)

            partial_deltas = previous_layer_deltas * activateDerivative(self.activations[L - 1])
            bias_deltas[L - 1] = partial_deltas 
            weight_deltas[L - 1] = np.dot(partial_deltas, self.activations[L - 2].T)

            L -= 1

        return bias_deltas, weight_deltas




    def accuracy(self, samples, labels):
        predictions = [self.feedforward(sample) for sample in samples]
        num_correct = sum([np.argmax(p) == np.argmax(l) for p, l in zip(predictions, labels)])
        print(num_correct,"/", len(samples), "Genauigkeit:", 100 * num_correct / len(samples), "%")
