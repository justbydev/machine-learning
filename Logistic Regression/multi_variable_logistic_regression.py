# -*- coding: utf-8 -*-
"""Multi_variable_Logistic_Regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EV3U37kTCZcWmn8rGbH3DV2TRfTyP0ot

Import Modules
"""

import numpy as np

"""Setting Data"""

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
y=data[:, 2:3].reshape((9, 1))

"""Setting Hyperparameter"""

learning_rate=0.01
epoch=50000
weight=np.random.rand(2, 1)
bias=np.random.rand(1)

"""Make sigmoid function"""

def sigmoid(x):
  return 1/(1+np.exp(-x))

"""Make Cost Function(Cross-Entropy cost function)"""

def error_function(W, b):
  y_pred=x.dot(W)+b
  y_pred=sigmoid(y_pred)
  return (-np.sum(y*np.log(y_pred)+(1-y)*np.log(1-y_pred)))

"""Derivative of Error(using numerical derivative)"""

def numerical_derivative(f, W, b):
  h=1e-4
  grad=np.zeros((1, W.shape[0]+1))
  for i in range(W.shape[0]):
    tmp=W[i, 0]
    W[i, 0]=float(W[i, 0])+h
    w_fx1=f(W, b)
    W[i, 0]=tmp

    tmp=W[i, 0]
    W[i, 0]=float(W[i, 0])-h
    w_fx2=f(W, b)
    W[i, 0]=tmp
    grad[0, i]=(w_fx1-w_fx2)/(2*h)
  
  b_fx1=f(W, float(b)+h)
  b_fx2=f(W, float(b)-h)
  grad[0, W.shape[0]]=(b_fx1-b_fx2)/(2*h)
  return grad

"""Predict"""

def predict(x):
  test=x.dot(weight)+bias
  test=sigmoid(test)
  print(test)
  if test>0.5:
    return 1
  else:
    return 0

"""Training"""

for i in range(epoch):
  grad=numerical_derivative(error_function, weight, bias)
  weight=weight-learning_rate*grad[0, :weight.shape[0]].reshape((weight.shape[0], 1))
  bias=bias-learning_rate*grad[0, weight.shape[0]]
  if i%5000==0:
     print('Epoch=', i, ' error_value=', error_function(weight, bias), "W=", weight, "b=", bias)

test=np.array([3, 17])
print(predict(test))