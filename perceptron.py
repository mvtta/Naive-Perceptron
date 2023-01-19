# importing the required MODULES
from sklearn import datasets
import numpy as np
from numpy import array, random, dot
from random import choice
from pylab import ylim, plot
import matplotlib.pyplot as plt
import seaborn as sns

#########################################################################################
# methods used and other naive python notes:
# class object: bundling data and functionality together.
# self.: Self is always pointing to Current Object. (still didn't quite grasped it's utility yet makes
# python syntax slightly more organized, imo)
# _init_ :init method or constructor, binds the attributes with the given arguments (assign)
# zip : maps values
# coef_ : coef_ gives you an array of weights estimated by linear regression.

#############
# variables
# init
# perceptron

#-------------------------dependencies-----------------------------------------------#
#from perceptron import *
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier


# #######################################################
# weights updated based in each training example
# learning of weights can continue for multiple iterations
# learning rate needs to be defined
#-------------------------init model params-------------------------------------------------#
class CustomPerceptron(object):
    def __init__(self, n_iterations=100, random_state=1, learning_rate=0.01, w_eval=np.array(0)):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.w_eval = w_eval
#######################################################################################
#    
#------------------------- training specs --------------------------------------------#
    def fit(self,X,y):
        rgen = np.random.RandomState(self.random_state)
        self.coef_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        for _ in range(self.n_iterations):
            for xi, expected_value in zip(X, y):
                predicted_value = self.predict(xi)
                self.coef_[1:] = self.coef_[1:] + self.learning_rate * (expected_value - predicted_value) * xi
                (self.coef_[0]) = self.coef_[0] + self.learning_rate * (expected_value - predicted_value) * 1

    # ####################################
    # sum of weights convetion net input
    def net_input(self, X):
        weighted_sum = np.dot(X, self.coef_[1:] + self.coef_[0])
        return weighted_sum
    
    #######################################
    # activation function is fed net input 
    def activation_function(self,X):
        weighted_sum = self.net_input(X)
        self.w_eval = np.append(self.w_eval, weighted_sum)
        return np.where(weighted_sum >= 0.0, 1, 0)
    
    ########################################
    # unit step function is is ecacured ro determine the output
    # predictions is made in the basus of output id activatuon function
    def predict(self,X):
        return self.activation_function(X)
   
    ###########################################
    # model score is calculared based in comparison of expected value and predicted value 
    # Should fix prob: assign vrar to np.array
    def score(self, X, y):
        misclassified_data_count = 0
        for xi, target in zip(X, y):
            output = self.predict(xi)
        if(target != output):
            misclassified_data_count += 1
        total_data_count = len(X)
        self.score_ = (total_data_count - misclassified_data_count)/total_data_count
        return self.score_

#-------------------------end training module----------------------------#

################################################
# Load the data set
#-------------------------data load--------------------------------------#
bc = datasets.load_breast_cancer()
X = bc.data
y = bc.target

data_df = pd.DataFrame(data=bc.data,
                       columns=bc.feature_names)
#print(data_df.head().T)
print(data_df.head)
print(list(bc.target_names))
sns.set(style="darkgrid")

sns.relplot(x=bc.target_names, y=bc.data[1],
            data=bc)
exit()
#exit()
#print(bc.feature_names, len(bc.feature_names))

#print(list(bc.data))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=2, random_state=42, stratify=y)
# clf = RandomForestClassifier(n_estimators=100, random_state=42)

###################################################
# Instantiate CustomPerceptron
prcptrn = CustomPerceptron()


##################################################
# Fit the model
#-------------------------fit--------------------------------------#
prcptrn.fit(X_test, y_test)
prcptrn.fit(X_train, y_train)

result = permutation_importance(prcptrn, X_train, y_train, n_repeats=10, random_state=42)
perm_sorted_idx = result.importances_mean.argsort()
print("Accuracy on test data: {:.2f}\n".format(prcptrn.score(X_test, y_test)))
print(perm_sorted_idx, type(result), len(result))
print("\n")
print(result)
#exit()
#

##################################################
# visualize
#-------------------------viz--------------------------------------#
plt.figure(figsize=(30,10), layout='constrained')

plt.plot(y_train, X_train, 'r--')
plt.plot(y_test, X_test, 'b--')
plt.xlabel('iterations')
plt.ylabel('weighted sum')

plt.legend()
plt.show()
plt.savefig('Test-and-Training-Visualization2.png')

#-------------------------eof--------------------------------------#
