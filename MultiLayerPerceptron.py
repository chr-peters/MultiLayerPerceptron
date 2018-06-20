#! /usr/bin/env python3

import numpy as np

import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

class MultiLayerPerceptron:

    '''
    'layers' is a list describing how many neurons per layer the network
    contains.
    For example for a neural net with 2 input neurons, 4 hidden neurons and
    3 output neurons one would set layers = [2, 4, 3]
    '''
    def __init__(self, layers):
        self.layers = layers

        # the weights of the network will be a list of weight matrices
        self.weights = []
        # the biases are represented as a list of vectors
        self.biases = []

        # initialize weights and biases according to a standard normal distribution
        for layerNum in range(len(layers)-1):
            self.weights.append(np.random.randn(layers[layerNum+1], layers[layerNum]))
            self.biases.append(np.random.randn(layers[layerNum+1]))

        # initialize the activation function as the sigmoid function
        self.sigmoid = lambda x: 1./(1.+np.exp(-x))

        # also initialize its derivative
        self.sigmoidDerivative = lambda x: self.sigmoid(x) * (1. - self.sigmoid(x))

        # initialize the cost-function for a concrete example
        self.cost = lambda out, outTarget: 1./2 * (out-outTarget)**2

        # the derivative of the cost-function for a concrete example
        self.costDerivative = lambda out, outTarget: (out-outTarget)

    '''
    The feed-forward pass through the network
    '''
    def getOutput(self, netInput):
        # the current activation of the current layer
        curActivation = netInput
        # propagate the signal through the network
        for curBias, curWeight in zip(self.biases, self.weights):
            curActivation = self.sigmoid(np.dot(curWeight, curActivation) + curBias)
        return curActivation

    '''
    Implementation of the stochastic gradient descent algorithm.
    '''
    def stochasticGradientDescent(self, batchSize, numEpochs, learningRate, trainingData, costListeners=[]):
        # count the iterations
        iterations = 1
        
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

                # the cost of the batch
                batchCost = 0

                # iterate over each example in the batch
                for netInput, output in curBatch:
                    # calculate the cost
                    if len(costListeners) > 0:
                        batchCost += sum(self.cost(self.getOutput(netInput), output))
                    
                    # get the gradients for a single example
                    curGradientWeights, curGradientBiases = self.backpropagate(netInput, output)

                    # add them together
                    gradientWeights = [weight + curWeight for weight, curWeight in zip(gradientWeights, curGradientWeights)]
                    gradientBiases = [bias + curBias for bias, curBias in zip(gradientBiases, curGradientBiases)]

                # average the batchCost
                batchCost = batchCost / len(curBatch)

                # print the cost to the listeners
                for curListener in costListeners:
                    curListener.putValues(iterations, batchCost)

                # average the gradients
                gradientWeights = [1./len(curBatch) * curWeight for curWeight in gradientWeights]
                gradientBiases = [1./len(curBatch) * curBias for curBias in gradientBiases]

                # update the network parameters by taking one step of gradient descent
                self.updateParameters(gradientWeights, gradientBiases, learningRate)

                iterations+=1


    '''
    Computes the gradient of the loss function with respect to the weights as well as
    the biases on a single example using the backpropagation algorithm.
    Returns (gradientWeights, gradientBiases) where gradientWeights is a list of matrices
    representing the updates to each weight matrix and gradientBiases is a list of vectors
    representing the updates to each bias vector.

    'netInput': the input signal the network receives
    'output': the desired output
    '''
    def backpropagate(self, netInput, output):
        # initialize the matrices where the gradient will be stored
        gradientWeights = [np.zeros(curWeight.shape) for curWeight in self.weights]
        gradientBiases = [np.zeros(curBias.shape) for curBias in self.biases]
        
        # compute the activations and netInputs (denoted as z) for each layer
        activations = [np.array(netInput)]
        z = []

        # forward pass
        for curWeight, curBias in zip(self.weights, self.biases):
            curZ = np.dot(curWeight, activations[-1]) + curBias
            activations.append(self.sigmoid(curZ))
            z.append(curZ)

        # compute the delta of the output layer
        delta = self.costDerivative(activations[-1], output)*self.sigmoidDerivative(z[-1])

        # compute the weight and bias changes for the output layer
        gradientWeights[-1] = np.outer(delta, activations[-2])
        gradientBiases[-1] = delta

        # propagate through the rest of the network
        for curLayer in range(2, len(self.layers)):
            # compute delta
            delta = np.dot(self.weights[-curLayer+1].transpose(), delta) * self.sigmoidDerivative(z[-curLayer])

            # update weight changes and biases
            gradientWeights[-curLayer] = np.outer(delta, activations[-curLayer-1])
            gradientBiases[-curLayer] = delta

        return gradientWeights, gradientBiases
        

    '''
    Updates the networks parameters when given the gradient as well as a learning rate by taking
    one step of gradient descent.
    '''
    def updateParameters(self, gradientWeights, gradientBiases, learningRate):
        # update the weight matrices
        self.weights = [weight - learningRate * gradient for weight, gradient in zip(self.weights, gradientWeights)]

        # update the biases
        self.biases = [bias - learningRate * gradient for bias, gradient in zip(self.biases, gradientBiases)]

'''
This listener will print the current score to stdout in each iteration
'''
class StandardOutListener:
    def putValues(self, iterations, cost):
        print("Cost after {} iterations: {}".format(iterations, cost))

'''
This listener will create a real-time plot from the cost values
'''
class RealtimePlotListener:
    def __init__(self):
        self.iterationValues = []
        self.costValues = []
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Network Cost Function")

    def putValues(self, iterations, cost):
        self.iterationValues.append(iterations)
        self.costValues.append(cost)
        plt.plot(self.iterationValues, self.costValues)
        plt.pause(0.00001)

    def createPlot(self):
        plt.show()

'''
This listener will a static plot from the cost values
'''
class PlotListener:
    def __init__(self):
        self.iterationValues = []
        self.costValues = []

    def putValues(self, iterations, cost):
        self.iterationValues.append(iterations)
        self.costValues.append(cost)

    def createPlot(self):
        plt.plot(self.iterationValues, self.costValues)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Network Cost Function")
        plt.show()
    
'''
The following code tests the network on the XOR-Problem
'''
if __name__ == "__main__":
    network = MultiLayerPerceptron([2, 10, 10, 2])

    # the XOR problem
    trainingData = [([0, 0], [1, 0]),
                     ([0, 1], [0, 1]),
                     ([1, 0], [0, 1]),
                     ([1, 1], [1, 0])]

    # use this to generate a plot of the networks progress
    pl = PlotListener()
    
    # train the network
    network.stochasticGradientDescent(4, 2000, 0.5, trainingData, costListeners=[StandardOutListener(), pl])

    # test the network
    for curExample in trainingData:
        prediction = network.getOutput(curExample[0])
        print("Input: "+str(curExample[0])+", Solution: "+str(curExample[1])+", Prediction: "+str(prediction))

    pl.createPlot()
            
