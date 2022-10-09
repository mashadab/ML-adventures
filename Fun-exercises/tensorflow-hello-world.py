#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 18:03:56 2022

@author: afzal-admin
"""

#Tensorflow-hello world

import tensorflow as tf
import numpy as np


#Building a Dense NN with 2 sequential, hidden layers with 3 neurons in first and 2 in second
#units = neurons

layer_1 = tf.keras.layers.Dense(units=3, activation = "sigmoid")  #layer 1 initialization
layer_2 = tf.keras.layers.Dense(units=2, activation = "sigmoid")  #layer 2 initialization

model   = tf.keras.Sequential([layer_1,layer_2])  #model init with sequential hiddeen layers
model.compile()



