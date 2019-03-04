
#####################################################################################################################
#   CS 6375.003 - Assignment 3, Neural Network Programming
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from collections import defaultdict

class NeuralNet:
    def __init__(self, dataset, header = True, h1 = 4, h2 = 2):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        #raw_input = pd.read_csv(train)
        # TODO: Remember to implement the preprocess method
        raw = pd.read_csv(dataset)
        raw.replace('?',np.NaN)
        raw.replace('',np.NaN)
        raw.dropna(axis=0, how='any')
        raw.drop_duplicates(keep='first', subset=None, inplace = False)
        _dataset = self.preprocess(raw)
        scaler = MinMaxScaler()
        scaledData = scaler.fit_transform(_dataset)
        _dataset = pd.DataFrame(scaledData[:,:])
        ncols = len(_dataset.columns)
        nrows = len(_dataset.index)
        self.attributeValues= _dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.classValues = _dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)        
        self.attributeValues_train, self.attributeValues_test, self.classValues_train, self.classValues_test = train_test_split(self.attributeValues, self.classValues, test_size=0.25)
        self.classValues_train = pd.DataFrame(self.classValues_train[:,:])
        self.classValues_test = pd.DataFrame(self.classValues_test[:,:])
        self.attributeValues_train = pd.DataFrame(self.attributeValues_train[:,:])
        self.attributeValues_test = pd.DataFrame(self.attributeValues_test[:,:])
        ncols_TRAIN = len(self.attributeValues_train.columns)
        nrows_TRAIN = len(self.attributeValues_train.index)
        self.X = self.attributeValues_train.values.reshape(nrows_TRAIN, ncols_TRAIN)
        self.y= self.classValues_train.iloc[:,0].values.reshape(nrows_TRAIN,1)
        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        # assign random weights to matrices in network in the range -1 to 1 
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))
    #
    # TODO: I have coded the sigmoid activation function, you need to do the same for tanh and ReLu -- DONE
    #

    def __activation(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        elif activation == "tanh":
        	self.__tanh(self,x)
        elif activation == "ReLu":
        	self.__relu(self,x)


    #
    # TODO: Define the function for tanh, ReLu and their derivatives 	-- DONE
    #

    def __activation_derivative(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        elif activation == "tanh":
        	self.__tanh_derivative(self,x)
        elif activation == "ReLu":
        	self.__relu_derivative(self,x)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __tanh(self,x):
    	return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    def __relu(self,x):
        return x * (x > 0)

    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __tanh_derivative(self,x):
    	return 1 - (x*x)

    def __relu_derivative(self,x):
        return 1 * (x > 0)
        
    #
    # TODO: Write code for pre-processing the dataset, which would include standardization, normalization,
    #   categorical to numerical, etc
    #

    def preprocess(self, X):
        dict = defaultdict(LabelEncoder)
        encoded = X.apply(lambda l: dict[l.name].fit_transform(l))
        return encoded

    # Below is the training function

    def train(self,activation, max_iterations = 1000, learning_rate = 0.005):
        for iteration in range(max_iterations):
            out = self.forward_pass(activation)
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out, activation)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)

    def forward_pass(self,activation):
        # pass our inputs through our neural network
        in1 = np.dot(self.X, self.w01 )
        if activation == "sigmoid":
            self.X12 = self.__sigmoid(in1)
        elif activation == "tanh":
        	self.X12 = self.__tanh(in1)
        elif activation == "ReLu":
        	self.X12 = self.__relu(in1)
        
        in2 = np.dot(self.X12, self.w12)
        if activation == "sigmoid":
            self.X23 = self.__sigmoid(in2)
        elif activation == "tanh":
        	self.X23 = self.__tanh(in2)
        elif activation == "ReLu":
        	self.X23 = self.__relu(in2)
        
        in3 = np.dot(self.X23, self.w23)
        out= 0
        if activation == "sigmoid":
            out = self.__sigmoid(in3)
        elif activation == "tanh":
        	out = self.__tanh(in3)
        elif activation == "ReLu":
        	out = self.__relu(in3)
        return out
            
    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    # TODO: Implement other activation functions -- DONE

    def compute_output_delta(self, out, activation):
        delta_output=0
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        elif activation == "tanh":
        	delta_output  = (self.y - out) * (self.__tanh_derivative(out))
        elif activation == "ReLu":
        	delta_output  = (self.y - out) * (self.__relu_derivative(out))

        self.deltaOut = delta_output

    # TODO: Implement other activation functions -- DONE

    def compute_hidden_layer2_delta(self, activation):
        delta_hidden_layer2=0
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
        elif activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))
        elif activation == "ReLu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__relu_derivative(self.X23))
        self.delta23 = delta_hidden_layer2

    # TODO: Implement other activation functions -- DONE

    def compute_hidden_layer1_delta(self, activation):
        delta_hidden_layer1=0
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        elif activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))
        elif activation == "ReLu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__relu_derivative(self.X12))

        self.delta12 = delta_hidden_layer1

    # TODO: Implement other activation functions -- DONE

    def compute_input_layer_delta(self, activation):
        delta_input_layer=0
        if activation == "sigmoid":
            delta_input_layer = np.multiply(self.__sigmoid_derivative(self.X01), self.delta01.dot(self.w01.T))
        elif activation == "tanh":
            delta_input_layer = np.multiply(self.__tanh_derivative(self.X01), self.delta01.dot(self.w01.T))
        elif activation == "ReLu":
            delta_input_layer = np.multiply(self.__relu_derivative(self.X01), self.delta01.dot(self.w01.T))

        self.delta01 = delta_input_layer

    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    def predict(self, activation, header = True):
        ncols = len(self.attributeValues_test.columns)
        nrows = len(self.attributeValues_test.index)
        self.X = self.attributeValues_test.values.reshape(nrows, ncols)
        self.y = self.classValues_test.iloc[:,0].values.reshape(nrows,1)
        out = self.forward_pass(activation)
        error = 0.5 * np.power((out - self.y), 2)
        print ("Predicted error : ",np.sum(error))
        return np.sum(error)


if __name__ == "__main__":
    dataPath = input("Enter Path for data set : ")
    if(dataPath):
        print("-------ACTIVATION - SIGMOID-------")
        neural_network = NeuralNet(dataPath)
        neural_network.train("sigmoid")
        testError = neural_network.predict("sigmoid")
        
        print("-------ACTIVATION - TANH-------")
        neural_network = NeuralNet(dataPath)
        neural_network.train("tanh")
        testError = neural_network.predict("tanh")
        
        print("-------ACTIVATION - RELU-------")
        neural_network = NeuralNet(dataPath)
        neural_network.train("ReLu")
        testError = neural_network.predict("ReLu")
    else:
        print("No path entered. Try Again!")