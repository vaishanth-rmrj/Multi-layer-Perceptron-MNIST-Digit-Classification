#!/usr/bin/env python3

from _typeshed import Self
import numpy as np

class MultiLayerPerceptron:

    def __init__(self, layers_size, activation_func="sigmoid",
                        reg_lambda=0, bias_flag=True):
        '''
        Arguments:
            size_layers : List with the number of Units for:
                [Input, Hidden1, Hidden2, ... HiddenN, Output] Layers.
            act_funtc   : Activation function for all the Units in the MLP
                default = 'sigmoid'
            reg_lambda: Value of the regularization parameter Lambda
                default = 0, i.e. no regularization
            bias: Indicates is the bias element is added for each layer, but the output
        '''
        print("constructing mlp")
        self.layers_size = layers_size
        self.n_layers    = len(layers_size)
        self.activation_func       = activation_func
        self.lambda_r    = reg_lambda
        self.bias_flag   = bias_flag
 
        # Ramdomly initialize theta (MLP weights)
        self.layer_weights = self.initialize_layer_weights()

    def initialize_layer_weights(self):
        '''
        Initialize layer weights, initialization method depends
        on the Activation Function and the Number of Units in the current layer
        and the next layer.
        The weights for each layer as of the size [next_layer, current_layer + 1]
        '''

        layer_weights = []
        layer_sizes = self.layers_size
        # removing first layer size
        next_layer_sizes = self.layers_size.copy()
        next_layer_sizes.pop(0)

        for layer_size, nxt_layer_size in zip(layer_sizes, next_layer_sizes):

            if self.activation_func == "sigmoid":
                # Method presented "Understanding the difficulty of training deep feedforward neurla networks"
                # Xavier Glorot and Youshua Bengio, 2010
                epsilon = 4.0 * np.sqrt(6) / np.sqrt(layer_size + nxt_layer_size)
                # Weigts from a uniform distribution [-epsilon, epsion]
                # including bias term in the weights
                # each row in weights matrix corresponds to one perceptron node
                temp_weights = epsilon * ( (np.random.rand(nxt_layer_size, layer_size + 1) * 2.0 ) - 1)

            if self.activation_func == "relu":
                # Method presented in "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classfication"
                # He et Al. 2015
                epsilon = np.sqrt(2.0 / (layer_size * nxt_layer_size))
                # Weigts from Normal distribution mean = 0, std = epsion
                # including bias term in the weights
                # each row in weights matrix corresponds to one perceptron node
                temp_weights = epsilon * (np.random.randn(nxt_layer_size, layer_size + 1 ))

            layer_weights.append(temp_weights)
        return layer_weights
            



    def train(self, x, y, iters=400, reset_weights = False):
        '''
        Given X (feature matrix) and y (class vector)
        Updates the Theta Weights by running Backpropagation N tines
        Arguments:
            X          : Feature matrix [n_examples, n_features]
            Y          : Sparse class matrix [n_examples, classes]
            iterations : Number of times Backpropagation is performed
                default = 400
            reset      : If set, initialize Theta Weights before training
                default = False
        '''
        if reset_weights:
            self.initialize_layer_weights()

        for _ in range(iters):
            self.dw = self.backpropogation(x, y)
            # still constructing

    
    def backpropogation(self, x, y):
        '''
        Implementation of the Backpropagation algorithm with regularization
        '''
        if self.activation_func == "sigmoid":
            g_dz = lambda x: self.sigmoid_derivative(x)
        elif self.activation_func == "relu":
            g_dz = lambda x: self.relu_derivative(x)
        
        num_training = x.shape[0]
        # A -> list of inputs for corresponding layers
        # Z -> list of outputs for prev layer without activation 
        A, Z = self.feedforward(x)

        # Backpropogation
        # computing deltas (i.e change)
        deltas = [None] * self.n_layers
        # change for output layer is 
        # -> predicted class value - actual class val
        deltas[-1] = A[-1] - y

        # for layers other than last layer
        # looping backward i.e 2 -> 0 layer id
        for layer_id in np.arange(self.n_layers -1 -1, 0, -1):
            temp_weights = self.layer_weights[layer_id]

            # removing bias term from the weights
            temp_weights = np.delete(temp_weights, np.s_[0], 1)
            deltas[layer_id] = (np.matmul(temp_weights.T, deltas[layer_id+1].T).T)*g_dz(Z[layer_id])

        # still constructing
    
    def feedforward(self, x):
        if self.activation_func == "sigmoid":
            g = lambda x: self.sigmoid(x)
        elif self.activation_func == "relu":
            g = lambda x: self.relu(x)

        A, Z = [None]*self.n_layers, [None]*self.n_layers
        layer_input = x

        for layer_id in range(self.n_layers-1):

            num_train = layer_input.shape[0]
            # adding 1 to each input layer's front
            layer_input = np.concatenate((np.ones((num_train, 1)), layer_input), axis= 1)

            # adding layers to A list
            A[layer_id] = layer_input
            # multiplying layer input values with corresponding layer weights
            Z[layer_id + 1] = np.matmul(layer_input, self.layer_weights[layer_id].T)
            # passing output value through activation function
            layer_output = g(Z[layer_id + 1])

            # this output is the input for next layer
            layer_input = layer_output
        
        # adding last layer results to A list
        A[self.n_layers - 1] = layer_output

        return A, Z






    # activation functions
    def sigmoid(self, z):
        '''
        Sigmoid function
        z can be an numpy array or scalar
        '''
        return 1.0 / (1.0 + np.exp(-z))
    
    def relu(self, z):
        '''
        Rectified Linear function
        z can be an numpy array or scalar
        '''

        if np.isscalar(z):
            return np.max((0, z))
        
        else:
            # ???
            zero_aux = np.zeros(z.shape)
            meta_z = np.stack((z , zero_aux), axis = -1)
            return np.max(meta_z, axis = -1)

    def sigmoid_derivative(self, z):
        '''
        Derivative for Sigmoid function
        z can be an numpy array or scalar
        '''
        return self.sigmoid(z) * (1- self.sigmoid(z))

    def relu_derivative(self, z):
        '''
        Derivative for Rectified Linear function
        z can be an numpy array or scalar
        '''
        return 1 * (z > 0)
