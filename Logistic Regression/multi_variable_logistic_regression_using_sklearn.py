# -*- coding: utf-8 -*-
"""Multi_variable_Logistic_regression_using_sklearn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iYj3OaPZ_L156QqlZu6YVz_xSUw3OqzH

Import Modules
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

"""Data"""

data=np.array([[2, 4, 0],
               [4, 11, 0],
               [6, 6, 0],
               [8, 5, 0],
               [10, 7, 1],
               [12, 16, 1],
               [14, 8, 1],
               [16, 3, 1],
               [18, 7, 1]])
x=data[:, 0:2].reshape((9, 2))
print(x)
y=data[:, 2:3].reshape((9, ))

"""Setting Model"""

model=LogisticRegression()
model.fit(x, y)

"""print weight"""

print(model.coef_)

"""Accuracy"""

print(model.score(x, y))

test=np.array([5, 8]).reshape(1, -1)
print(model.predict_proba(test))
print(model.predict(test))