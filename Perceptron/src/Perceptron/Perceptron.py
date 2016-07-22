import math
import random
import matplotlib.pyplot as plt

"""Multi Layer Perceptron"""
class Perceptron:
    def __init__(self, neuronsPerLayer, actFunctPerLayer, derivativePerLayer):
        self.neuronsPerLayer = neuronsPerLayer
        self.actFunctPerLayer = actFunctPerLayer
        self.derivativePerLayer = derivativePerLayer
        neuronsPerLayer[0] += 1  # add the bias neuron
        histogram = [neuronsPerLayer[0]]  # create histogram for efficient computations
        for i in list(range(1, len(neuronsPerLayer))):
            histogram += [histogram[i - 1] + neuronsPerLayer[i]]
        self.histogram = histogram
        totalWeights = histogram[len(neuronsPerLayer) - 1]
        self.weights = [[None for _ in range(totalWeights)] for _ in range(totalWeights)]
        curLayer = 0
        for i in range(totalWeights):  # iterate over the weight matrix
            if (i >= histogram[curLayer]):
                curLayer += 1
            for j in range(totalWeights):
                if(i == 0):  # init bias neuron
                    if(j >= neuronsPerLayer[0]):
                        self.weights[i][j] = random.random()*0.5+0.1
                else:
                    # init the weight if j is a member of the next layer
                    if (j >= histogram[curLayer] and j < histogram[curLayer + 1]):
                        self.weights[i][j] = random.random()*0.5+0.1  # a small arbitrary number
                    else:
                        self.weights[i][j] = None
        print(self.weights)
        
    def getOutput(self, netInput):
        activation = [0 for _ in range(self.histogram[len(self.histogram) - 1])]
        weightedSums = [0 for _ in range(self.histogram[len(self.histogram) - 1])]
        activation[0] = 1  # bias neuron
        curLayer = 0
        for i in list(range(1, self.histogram[-1])):
            if(i >= self.histogram[curLayer]):
                curLayer += 1
            if(curLayer == 0):
                activation[i] = netInput[i - 1]  # initialize the input neurons, i-1 because of the bias neuron
            else:
                weightedSum = 0  # calculate the weighted sum of the inputs of the preceding neurons
                for j in range(self.histogram[-1]):
                    if (self.weights[j][i] is not None):
                        weightedSum += self.weights[j][i] * activation[j]
                weightedSums[i] = weightedSum
                activation[i] = self.actFunctPerLayer[curLayer](weightedSum)
        self.weightedSums = weightedSums
        self.activation = activation
        return [activation[x] for x in list(range(self.histogram[curLayer - 1], self.histogram[curLayer]))]
    
    def backprop(self, learningRate, netInputs, teachingInputs, plotBool):
        plot=[]
        for i in range(len(netInputs)):
            output = self.getOutput(netInputs[i])
            curLayer = len(self.histogram) - 1
            deltaValues=[0 for _ in range(self.histogram[-1])]
            for h in list(range(self.histogram[0], self.histogram[-1]))[::-1]:
                if (h < self.histogram[curLayer - 1]):
                    curLayer -= 1
                for k in range(self.histogram[-1]):
                    if(self.weights[k][h] is not None):
                        delta = 0
                        if (curLayer == len(self.histogram) - 1):  # h is output neuron
                            outputTransformation = h - self.histogram[curLayer - 1]
                            delta = self.derivativePerLayer[curLayer](self.weightedSums[h]) * \
                                (teachingInputs[i][outputTransformation] - output[outputTransformation])
                        else:
                            deltaSum=0
                            for l in range(self.histogram[-1]):
                                if(self.weights[h][l] is not None):
                                    deltaSum+=deltaValues[l]*self.weights[h][l]
                            delta = self.derivativePerLayer[curLayer](self.weightedSums[h])*deltaSum
                        deltaValues[h]=delta    
                        self.weights[k][h] += learningRate * self.activation[k] * delta
            if(plotBool):#TODO Compute actual output sum
                plot+=[abs(output[x]-teachingInputs[i][x]) for x in range(len(output))]
        if(plotBool):
            plt.plot(plot)
            plt.ylabel("Sum of the euclidean error.")
            plt.show();
                        
def fermi(T):
    return lambda x: 1/(1+math.exp(-x/T))
    
def fermiDerivative(T):
    return lambda x: 1/T*fermi(T)(x)*(1-fermi(T)(x))
