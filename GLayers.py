#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tensorflow as tf
import tensorflow.keras.layers as kerlayers


class GConvLayer(kerlayers.Layer):
    
    def __init__(self, outdim, activation=tf.keras.activations.relu, **kwargs):
        super().__init__(**kwargs)
        self.outdim = outdim
        self.activation = activation
        
    def build(self, inputsh):
        inputlen = int(inputsh[-1])
        self.w1 = self.add_weight("weight_1",shape=[inputlen,self.outdim], initializer=tf.keras.initializers.GlorotNormal())
        self.w2 = self.add_weight("weight_2",shape=[inputlen,self.outdim], initializer=tf.keras.initializers.GlorotNormal())
        self.b = self.add_weight("bias",shape=[self.outdim])
        
    def call(self, ins, adj):
        z1 = tf.matmul(ins, self.w1)
        z2 = tf.matmul(ins, self.w2)
        
        az1 = tf.matmul(adj, z1)
        
        az = az1 + z2
        digit = tf.nn.bias_add(az, self.b)
        return(self.activation(digit))
    


        




