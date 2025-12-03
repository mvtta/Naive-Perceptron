import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from random import choice
from sklearn import datasets
from pylab import ylim, plot
from numpy import array, random, dot

#-------------------------from train-perceptron--------------------------------------#

from sklearn.datasets import load_iris
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer


class CustomPerceptron(object):
    
    def __init__(self, n_iterations=100, 
                 random_state=1, 
                 learning_rate=0.01):
                     
        self.n_iterations     = n_iterations
        self.random_state     = random_state
        self.learning_rate    = learning_rate

    

    def fit(self,X,y):
        
        rgen       = np.random.RandomState(self.random_state)
        self.coef_ = rgen.normal(loc=0.0, 
                                 scale=0.01, 
                                 size=1 + X.shape[1])
        plt.figure()
        
        for _ in range(self.n_iterations):
            for xi, expected_value in zip(X, y):
                predicted_value = self.predict(xi)
                
                plt.plot(expected_value, predicted_value)
                self.coef_[1:]  = self.coef_[1:] + self.learning_rate * (expected_value \
                                                                        - predicted_value\
                                                                       ) * xi
                (self.coef_[0]) = self.coef_[0] + self.learning_rate * (expected_value \
                                                                        - predicted_value\
             
                                                                       ) * 1

    # sum of weights convolutional net input
    def net_input(self, X):
        weighted_sum     = np.dot(X, self.coef_[1:] + self.coef_[0])
        return weighted_sum
        
    # activation function is fed net input 
    def activation_function(self,X):
        weighted_sum     = self.net_input(X)
        return np.where(weighted_sum >= 0.0, 1, 0)
    
    # unit step function 
    def predict(self,X):
        return self.activation_function(X)

    # model score 
    def score(self, X, y):
        misclassified_data_count = 0
        for xi, target in zip(X, y):
            output  = self.predict(xi)
            
        if(target != output):
            misclassified_data_count += 1
            
        total_data_count    = len(X)
        self.score_         = (total_data_count - misclassified_data_count)/total_data_count
        return self.score_



#-------------------------from train-perceptron--------------------------------------#
bc = datasets.load_breast_cancer()
X = bc.data
y = bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=42, 
                                                    stratify=y)
# Instantiate CustomPerceptron
prcptrn = CustomPerceptron()

# Fit
prcptrn.fit(X_train, y_train)

# Score
prcptrn.score(X_test, y_test), prcptrn.score(X_train, y_train)  
#-------------------------from train-perceptron--------------------------------------#
