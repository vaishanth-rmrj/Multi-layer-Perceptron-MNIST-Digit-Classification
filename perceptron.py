#!/usr/bin/env python3

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
        print(self.layer_weights[0].shape)

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
            



    def train(self):
        pass