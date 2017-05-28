# -*- coding: utf-8 -*-
"""
Created on Sun May 28 15:31:15 2017

@author: Robert
"""

import tensorflow as tf
from tensorflow.examples.tutorials.minist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE = 500
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        return tf.matmul(layer1,weights2)+biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weightsa))+avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)
        
def train(minist):
    x  = tf.placeholder(tf.float32,[None,INPUT_NODE], name = "x-inout")
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name = "y-input")
    
    weights = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev = 0.1))
    biases  = tf.Variable(tf.constant(0.1,shape = [LAYER1_NODE]))
    
    weights2 = tf.Variable(tf.truncate_normal(LAYER1_NODE,OUTPUT_NODE),stddev = 0.1)
    biases2 = tf.Variable(tf.constant(0.1,shape = [OUTPUT_NODE]))
    
    y = inference(x,None,weights1,weights1,biases1,weights2,biases2)
    
    global_step = tf.Variable(0,trainable = False)
    
    variable_averages = tf.train