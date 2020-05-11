import numpy as np
import gzip
import matplotlib.pyplot as plt

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
        inputError = np.dot(outputError,self.weights.T)
        weighsError = np.dot(self.input.T,outputError)

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


network = neuralNetwork()
network.addLayer(fullyConectedLayer(2,3))
network.addLayer(activationLayer(tanh,tanh_prime))
network.addLayer(fullyConectedLayer(3,1))
network.addLayer(activationLayer(tanh,tanh_prime))

network.setLossFuntion(meanSquareError, meanSquareErrorDerivative)

network.trainNetwork(x_train,y_train, 2000, 0.1)


