import numpy as np
import sys
import math
import random
class neuralNet:
    def __init__(self,layerSizes,epochs=20,batchSize=20,learningRate=0.1,biasWeights=None,weights=None):

        self.sizes = layerSizes
        self.epochs = epochs
        self.batchSize = batchSize
        self.learningRate = learningRate

        if biasWeights and weights:
            self.biasWeights = biasWeights
            self.weights = weights
        else:
            self.weights = [np.random.randn(y,x) for x,y in zip(layerSizes[:-1],layerSizes[1:])]#for
            self.biasWeights = [np.random.randn(y, 1) for y in layerSizes[1:]] #for num hidden, then num output

            print(self.weights)

    def forwardPass(self,value):
        output = value
        netInValues = []
        activationValues = [value]

        for b,w in zip(self.biasWeights,self.weights):
            netIn = np.dot(w,output)
            activationValues.append(output)
        return netInValues,activationValues



    def sigmoid(self,x):
        return 1/(1+math.exp(-x))

    def sigmoidDerivative(self,x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))


numInputs = sys.argv[1]
numHidden = sys.argv[2]
numOutpus = sys.argv[3]
biasWeights = [
        np.array([[0.1], [0.1]]),
        np.array([[0.1], [0.1]])
    ]

weights = [
        np.array([[0.1, 0.1], [0.2, 0.1]]),
        np.array([[0.1, 0.1], [0.1, 0.2]])
    ]
layerSizes = [int(numInputs),int(numHidden),int(numOutpus)]
network = neuralNet(layerSizes,biasWeights,weights)
