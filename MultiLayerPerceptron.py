#! /usr/bin/env python3

import numpy as np

class MultiLayerPerceptron:

    '''
    'layers' is a list describing how many neurons per layer the network
    contains.
    For example for a neural net with 2 input neurons, 4 hidden neurons and
    3 output neurons one would set layers = [2, 4, 3]
    '''
    def __init__(self, layers):
        # the weights of the network will be a list of weight matrices
        self.weights = []
        # the biases are represented as a list of vectors
        self.biases = []

        # initialize weights and biases
        for layerNum in range(len(layers)-1):
            self.weights.append(np.random.rand(layers[layerNum+1], layers[layerNum]))
            self.biases.append(np.random.rand(layers[layerNum+1]))

        # initialize the activation function as the sigmoid function
        self.sigmoid = lambda x: 1./(1.+np.exp(-x))

        # also initialize its derivative
        self.sigmoidDerivative = lambda x: self.sigmoid(x) * (1. - self.sigmoid(x))

    '''
    The feed-forward pass through the network
    '''
    def getOutput(self, input):
        # the current activation of the current layer
        curActivation = input
        # propagate the signal through the network
        for curBias, curWeight in zip(self.biases, self.weights):
            curActivation = self.sigmoid(np.dot(curWeight, curActivation) + curBias)
        return curActivation

    '''
    Implementation of the stochastic gradient descent algorithm.
    '''
    def stochasticGradientDescent(self, batchSize, numEpochs, learningRate, trainingData):
        # iterate over the amount of epochs
        for curEpoch in range(numEpochs):
            # shuffle the training data
            np.random.shuffle(trainingData)
            
            # initialize the batches
            batches = [trainingData[k:k+batchSize] for k in range(0, len(trainingData), batchSize)]

            # compute the gradient for each batch
            for curBatch in batches:
                # the gradient will be stored here
                gradientWeights = [np.zeros(w.shape) for w in self.weights]
                gradientBiases = [np.zeros(b.shape) for b in self.biases]

                # iterate over each example in the batch
                for input, output in curBatch:
                    # get the gradients for a single example
                    curGradientWeights, curGradientBiases = self.backpropagate(input, output)

                    # average them together
                    gradientWeights = [weight + 1./len(curBatch) * curWeight for weight, curWeight in zip(gradientWeights, curGradientWeights)]
                    gradientBiases = [bias + 1./len(curBatch) * curBias for bias, curBias in zip(gradientBiases, curGradientBiases)]

                # update the network parameters by taking one step of gradient descent
                self.updateParameters(gradientWeights, gradientBiases, learningRate)


    '''
    Computes the gradient of the loss function with respect to the weights as well as
    the biases on a single example using the backpropagation algorithm.
    Returns (gradientWeights, gradientBiases) where gradientWeights is a list of matrices
    representing the updates to each weight matrix and gradientBiases is a list of vectors
    representing the updates to each bias vector.

    'input': the input signal the network receives
    'output': the desired output
    '''
    def backpropagate(self, input, output):
        pass

    '''
    Updates the networks parameters when given the gradient as well as a learning rate by taking
    one step of gradient descent.
    '''
    def updateParameters(self, gradientWeights, gradientBiases, learingRate):
        # update the weight matrices
        for curWeight, curWeightGradient in zip(self.weights, gradientWeights):
            curWeight = curWeight - learningRate * curWeightGradient

        # update the biases
        for curBias, curBiasGradient in zip(self.biases, gradientBiases):
            curBias = curBias - learningRate * curBiasGradient
            

'''
The following code tests the network on the XOR-Problem
'''
if __name__ == "__main__":
    network = MultiLayerPerceptron([2, 4, 2])

    # the XOR problem
    trainingData = [([0, 0], [1, 0]),
                     ([0, 1], [0, 1]),
                     ([1, 0], [0, 1]),
                     ([1, 1], [1, 0])]
    # train the network
    network.stochasticGradientDescent(1, 1, 1, trainingData)

    # test the network
    for curExample in trainingData:
        prediction = network.getOutput(curExample[0])
        print("Input: "+str(curExample[0])+", Prediction: "+str(prediction))
            
