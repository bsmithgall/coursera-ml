# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# Import the relevant libraries
import scipy.io as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Image
import random

# <codecell>

# Load our two matlab input files
training = sp.loadmat('ex3data1.mat')
weights = sp.loadmat('ex3weights.mat')

# <codecell>

X = training['X']
y = training['y']
m = X[:,1].size

# <codecell>

Theta2 = weights['Theta2']
Theta1 = weights['Theta1']
num_labels = Theta2[:,1].size

# <codecell>

# Define the sigmoid function for use in the NN
def sigmoid(z):
    """
    Takes in a scalar, vector, or matrix Z and returns the sigmoid function for the particular scalar, vector, or matrix element.
    """
    g = np.zeros( (z.shape) )
    g = 1/(1 + np.exp(-z) )
    return g

# <codecell>

def predict(Theta1, Theta2, X):
    """
    Uses a neural network framework with one input layer, one hidden layer, and one output layer. Returns an array of 1x1 arrays containing the learned amount.
    Note that the result '10' equates to a guess of '0'.
    """
    p = np.zeros( (X[:,1].size, 1) )
    
    # Add a column of ones to X
    x0 = np.hstack( (np.ones( (X[:,1].size, 1) ), X))
    z1 = sigmoid(np.dot(Theta1,np.transpose(x0)))
    a2 = np.vstack( (np.ones( (1, z1[1,:].size) ), z1) )
    z2 = np.dot(Theta2,a2)
    a3 = sigmoid(z2)
    a3 = np.transpose(a3)
    
    for i in range(a3.shape[0]):
        x = max(a3[i])
        ix = np.where(a3[i] == x)[0] + 1
        
        p[i] = ix
    
    return p

# <codecell>

preds = list(predict(Theta1, Theta2, X))

# <codecell>

# Calculate accuracy
print "Matching the training set at", np.mean(preds == y)*100, "% accuracy"

# <codecell>

def plot_result(X):
    """
    Demostrates the results of one example of X
    """
    example = random.randint(0,X.shape[0])
    img = np.reshape(X[example,:], (20, 20), order = 'F')
    imgplot = plt.imshow(img,cmap="Greys", origin = 'upper')
    pred = int(preds[example])
    if pred == 10:
        pred = 0
    print imgplot, '\t\t\tThe Model prediction is',pred

# <codecell>

plot_result(X)

# <codecell>


