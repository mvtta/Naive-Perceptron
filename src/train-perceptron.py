# import libs
# load dataset
# create traininf and test split
# instantiate costum perceptron
# fit the model
# score the model

from perceptron import *
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split




# Load the data set
#
X, y = load_iris(as_frame=True, return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
#
# Instantiate CustomPerceptron
#
prcptrn = CustomPerceptron()

# Fit the model
#
prcptrn.fit(X_train, y_train)


# Score the model
#
prcptrn.score(X_test, y_test), prcptrn.score(X_train, y_train)
