import shutil

import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

from GradientDescentAlgorithmsComparison.HyperParameters import HyperParameters
from GradientDescentAlgorithmsComparison.NeuralNetwork import NeuralNetwork

# Prepare log dir
path = os.path.abspath("./logs")
if os.path.exists(path):
    shutil.rmtree(path)  # os.remove(path)

# Read input data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

# Create parameters for neural network
hp = HyperParameters()

for learning_rate in [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]:

    # Create all testing optimizers
    gd = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    adagrad = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
    momentum = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    rmsProp = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.9)

    # All optimizers in the dictionary
    optimizers = {"GradientDescent": gd, "Adagrad": adagrad, "Adam": adam, "momentum": momentum, "RMSProp": rmsProp}

    for optimizerName, optimizer in optimizers.items():

        # Construct log folder
        logPath = os.path.join(path, "LR-" + str(learning_rate) + "\\" + optimizerName + "_" + str(learning_rate))
        print("Path to log:", logPath)

        # Create neural network
        nn = NeuralNetwork(optimizer, hp, logPath)

        try:
            # Train neural network
            nn.train(mnist.train.images, mnist.train.labels)

            # Evaluate trained network on test set
            e = nn.evaluate(mnist.test.images, mnist.test.labels)

            # Print accuracy for now
            print("Testing Accuracy:", e['accuracy'], " algorithm:", optimizerName, " LearningRate:", str(learning_rate))
        except:
            print("Error during training", " algorithm:", optimizerName, " LearningRate:", str(learning_rate))

print("Finished")
