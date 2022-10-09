#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 18:03:56 2022

@author: afzal-admin
"""

#Tensorflow-hello world

#importing the libraries
import tensorflow as tf
import numpy as np


#A. Building a Dense NN with 2 sequential, hidden layers with 3 neurons in first and 2 in second, output layer is 1

#1: input layer has neurons =  # of features, i.e., 4, in dense network
#2: hidden layers
layer_1 = tf.keras.layers.Dense(units=3, activation = "sigmoid")  #hidden layer 1 initialization #units = neurons
layer_2 = tf.keras.layers.Dense(units=2, activation = "sigmoid")  #hidden  layer 2 initialization
#3: output layer
output  = tf.keras.layers.Dense(units=1, activation = "sigmoid")  #output layer

model   = tf.keras.Sequential([layer_1,layer_2,output])  #model initializaed with sequential hidden layers

#B. Compiling the model: converting high level language to machine language
model.compile(loss='mse', optimizer='adam')  #mean squared error loss, OPtimization using ADAM gradient descent algorithm


#C. Identifying the training data
#features: # of training data by # of features matrix
x       = np.array([[1,1,1,1],\
                    [2,2,2,2],\
                    [3,3,3,3],\
                    [4,4,4,4],\
                    [6,6,6,6],\
                    [5,5,5,5]])


#target: # of training data array
y       = np.array([0,0,0,1,1,1])

#D. Model training
model.fit(x,y,epochs=1000)


#E. Prediction on test data
#loading the test features
x_new   = np.array([[2,1,4,2],\
                    [0,0,0,0],\
                    [10,10,10,10]])

#predicting the entry on test data
model.predict(x_new) 