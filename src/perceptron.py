# importing the required MODULES
from sklearn import datasets
import numpy as np
from numpy import array, random, dot
from random import choice
from pylab import ylim, plot
import matplotlib.pyplot as plt

# methods used and other py notes:
# class object bundling data and functionality together.
# self. Self is always pointing to Current Object.
# _init_ init method or constructor, binds the attributes with the given arguments (assign)
# zip maps values
# coef_ coef_ gives you an array of weights estimated by linear regression.

# variables
# init
# perceptron

#-------------------------from train-perceptron--------------------------------------#
#from perceptron import *
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
#-------------------------from train-perceptron--------------------------------------#

class CustomPerceptron(object):
    def __init__(self, n_iterations=100, random_state=1, learning_rate=0.01):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.learning_rate = learning_rate
# #
# weights updated based in each training example
# learning of weights can continue for multiple iterations
# learning rate needs to be defined

    def fit(self,X,y):
        rgen = np.random.RandomState(self.random_state)
        self.coef_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        plt.figure()
        for _ in range(self.n_iterations):
            for xi, expected_value in zip(X, y):
                predicted_value = self.predict(xi)
                plt.plot(expected_value, predicted_value)
                self.coef_[1:] = self.coef_[1:] + self.learning_rate * (expected_value - predicted_value) * xi
                (self.coef_[0]) = self.coef_[0] + self.learning_rate * (expected_value - predicted_value) * 1
                #print(predicted_value, expected_value, _)
            plt.savefig("graph-breat-cancer-using-plt-fig.png")
    # 
    # sum of weights convetion net input
    def net_input(self, X):
        weighted_sum = np.dot(X, self.coef_[1:] + self.coef_[0])
        return weighted_sum
    #
    # activation function is fed net input 
    def activation_function(self,X):
        weighted_sum = self.net_input(X)
        return np.where(weighted_sum >= 0.0, 1, 0)
    #
    # unit step function is is ecacured ro determine the output
    #
    # predictions is made in the basus of ourput id activatuon fubction
    def predict(self,X):
        return self.activation_function(X)
    #
    # model score is calculared based in comparison of expected value and predicted value
    
    # Should fiz prob: assign vrar to np.array
    
    def score(self, X, y):
        misclassified_data_count = 0
        for xi, target in zip(X, y):
            output = self.predict(xi)
        if(target != output):
            misclassified_data_count += 1
        total_data_count = len(X)
        self.score_ = (total_data_count - misclassified_data_count)/total_data_count
        return self.score_

#-------------------------from train-perceptron--------------------------------------#


# Load the data set
# Load the data se
bc = datasets.load_breast_cancer()
X = bc.data
y = bc.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

#X, y = load_iris(as_frame=True, return_X_y=True)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Instantiate CustomPerceptron
#
prcptrn = CustomPerceptron()

# Fit the model
#
prcptrn.fit(X_train, y_train)

# Score the model
#
prcptrn.score(X_test, y_test), prcptrn.score(X_train, y_train)  
#print(X_train,y_train)

#-------------------------from train-perceptron--------------------------------------#