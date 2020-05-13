import numpy as np
import np_utils
import sys
import gzip
import matplotlib.pyplot as plt
import load_data as ld
def meanSquareError(target,prediction):
    return np.mean(np.power(target-prediction,2))

def meanSquareErrorDerivative(target,prediction):
    return 2*(prediction-target)/target.size

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoidDerivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forwardPropagation(self,input):
        raise NotImplementedError

    def backwardPropagation(self,outputError,learningRate):
        raise NotImplementedError

class fullyConectedLayer(Layer):
    def __init__(self,inputSize,outputSize):
        self.weights = np.random.rand(inputSize,outputSize) - 0.5
        self.bias = np.random.rand(1,outputSize) - 0.5

    def forwardPropagation(self,inputData):
        self.input = inputData
        self.output = np.dot(self.input,self.weights) + self.bias
        return self.output

    def backwardPropagation(self,outputError,learningRate):
        #print("shapes=> ")
        #print("outputError = " ,outputError.shape)
        #print("weights = " ,self.weights.transpose().shape," weights[1] =", self.weights.transpose().shape[1])
        #print("input = " ,self.input.shape)

        #a = self.weights.transpose().shape[1]
        #b = outputError.shape[0]
        #print(a,b," Proposed shape")
        self.input = self.input.reshape((outputError.shape[0],self.weights.transpose().shape[1],))
        inputError = np.dot(outputError,self.weights.transpose())
        weighsError = np.dot(self.input.transpose(),outputError)
        self.weights -= learningRate*weighsError
        self.bias -= learningRate*outputError
        return inputError

class activationLayer(Layer):
    def __init__(self,activation,activationDerivative):
        self.activation = activation
        self.activationDerivative = activationDerivative

    def forwardPropagation(self,inputData):
        self.input = inputData
        self.output = self.activation(self.input)
        return self.output

    def backwardPropagation(self,outputError,learningRate):
        return self.activationDerivative(self.input) * outputError

class neuralNetwork:
    def __init__(self):
        self.layers = []
        self.lossFunction = None
        self.lossFunctionDerivative = None

    def addLayer(self,layer):
        self.layers.append(layer)

    def setLossFuntion(self,loss,lossDerivative):
        self.lossFunction = loss
        self.lossFunctionDerivative = lossDerivative

    def predictLabel(self,inputData):
        samples = len(inputData)
        result = []

        for i in range(samples):
            output = inputData[i]
            for layer in self.layers:
                output = layer.forwardPropagation(output)
            result.append(output)
        return result

    def trainNetwork(self,x,y,numEpochs,learningRate):
        samples = len(x)
        for i in range(numEpochs):
            err = 0
            for j in range(samples):
                output = x[j]
                for layer in self.layers:
                    output = layer.forwardPropagation(output)

                err += self.lossFunction(y[j],output)

                error = self.lossFunctionDerivative(y[j],output)
                for layer in reversed(self.layers):
                    error = layer.backwardPropagation(error,learningRate)
            err /= samples

            print('epoch',i+1,'/',numEpochs," error=",err)

#for i in range(2):
    #img = training_data[i][0].reshape((28,28))
    #print(training_data[i][1])
    #plt.imshow(img, cmap="Greys")
    #plt.show()
    #print(training_data[i][1])

#print(training_data[1].shape)

x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

print("Loading Data...")
"""
trainX = np.loadtxt("data/TrainDigitX.csv.gz", delimiter=',')
trainY = np.loadtxt("data/TrainDigitY.csv.gz", delimiter=',')
testX = np.loadtxt("data/TestDigitX.csv.gz", delimiter=',')
testY = np.loadtxt("data/TestDigitY.csv.gz", delimiter=',')
"""

trainX, trainY, testX, testY = ld.load_data("data/TrainDigitX.csv.gz","data/TrainDigitY.csv.gz","data/TestDigitX.csv.gz","data/TestDigitY.csv.gz")

"""------------------TESTING-----------------------"""


"""----------------TESTING END--------------------"""

print('x train shape',trainX.shape)


print("Data Loaded...")

network = neuralNetwork()
network.addLayer(fullyConectedLayer(28*28,100))
network.addLayer(activationLayer(sigmoid,sigmoidDerivative))
network.addLayer(fullyConectedLayer(100,50))
network.addLayer(activationLayer(sigmoid,sigmoidDerivative))
network.addLayer(fullyConectedLayer(50,10))
network.addLayer(activationLayer(sigmoid,sigmoidDerivative))
network.setLossFuntion(meanSquareError, meanSquareErrorDerivative)
network.trainNetwork(trainX[0:1],trainY[0:1],1,3)

out = network.predictLabel(testX[0:10])


for i in range(10):
    exp = testY[i][np.amax(testY[i])]
    act = np.where(out[i] == np.amax(out[i]))
    print(exp,act)

#print(testY[0:10])