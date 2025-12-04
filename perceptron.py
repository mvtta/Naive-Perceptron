import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from random import choice
from sklearn import datasets
from pylab import ylim, plot
from numpy import array, random, dot

#-------------------------dependencies-----------------------------------------------#
from sklearn.datasets import load_iris
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

#-------------------------init model params-------------------------------------------------#

class CustomPerceptron(object):
    
    def __init__(self, n_iterations=100, random_state=1, 
                 learning_rate=0.01, w_eval=np.array(0)):
                     
        self.n_iterations     = n_iterations
        self.random_state     = random_state
        self.learning_rate    = learning_rate
        self.w_eval           = w_eval
        
#-------------------------- training specs --------------------------------------------#
    def fit(self,X,y):
        rgen = np.random.RandomState(self.random_state)
        self.coef_ = rgen.normal(loc=0.0, 
                                 scale=0.01, 
                                 size=1 + X.shape[1])
        
        for _ in range(self.n_iterations):
            for xi, expected_value in zip(X, y):
                predicted_value   = self.predict(xi)
                self.coef_[1:]    = self.coef_[1:] + self.learning_rate * (expected_value - \
                                                                        predicted_value) * xi
                (self.coef_[0])   = self.coef_[0] + self.learning_rate * (expected_value - \
                                                                        predicted_value) * 1

    def net_input(self, X):   
        weighted_sum    = np.dot(X, self.coef_[1:] + self.coef_[0])
        return weighted_sum
    

    def activation_function(self,X):
        weighted_sum     = self.net_input(X)
        self.w_eval      = np.append(self.w_eval, weighted_sum)  
        return np.where(weighted_sum >= 0.0, 1, 0)

    
    def predict(self,X):
        return self.activation_function(X)

    
    def score(self, X, y):
         misclassified_data_count = 0
        for xi, target in zip(X, y):
            output = self.predict(xi)
        if(target != output):
            misclassified_data_count += 1
            
        total_data_count    = len(X)
        self.score_ = (total_data_count - misclassified_data_count\
                                          )/total_data_count
        return self.score_


#-------------------------data load--------------------------------------#
bc    = datasets.load_breast_cancer()
X     = bc.data
y     = bc.target

data_df = pd.DataFrame(data=bc.data,
                       columns=bc.feature_names)


sns.set(style="darkgrid")
sns.relplot(x=bc.target_names, 
            y=bc.data[1],
            data=bc)

X_train, X_test, y_train, y_test       = train_test_split(
                                       X, y, test_size=2, 
                                       random_state=42,
                                       stratify=y)


prcptrn         = CustomPerceptron()

prcptrn.fit(X_test, y_test)
prcptrn.fit(X_train, y_train)

result          = permutation_importance(prcptrn, X_train, 
                                         y_train, n_repeats=10, 
                                         random_state=42)

perm_sorted_idx = result.importances_mean.argsort()
