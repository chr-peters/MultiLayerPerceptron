'''
Created on Jul 22, 2016

@author: peters
'''
import Perceptron as p
from random import randint

netInputs = []
teachingInputs = []
for i in range(100000):
    a=randint(0, 1)
    b=randint(0, 1)
    netInputs+=[[a, b]]
    teachingInputs+=[[a^b]]
    
network=p.Perceptron([2, 4, 4, 1], [lambda x: x, p.fermi(0.3), p.fermi(0.3), p.fermi(0.3)], 
                     [lambda x:1, p.fermiDerivative(0.3), p.fermiDerivative(0.3), p.fermiDerivative(0.3)])
network.backprop(0.3, netInputs, teachingInputs, True)
print(network.weights)
print(network.getOutput([1, 1]))
print(network.getOutput([0, 1]))
print(network.getOutput([1, 0]))
print(network.getOutput([0, 0]))
