# -*- coding: utf-8 -*-
"""multi_class_logistic_regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1G9OIYaEYsWaFoyhq6-XJk98oOG6FZZ2y

Import Modules
"""

import numpy as np
import pandas as pd

"""Setting data"""

data=np.array([[6, 2, 8, 0],
               [8, 1, 7, 0],
               [7, 2, 9, 0],
               [5, 3, 3, 1],
               [4, 2, 3, 1],
               [4, 1, 2, 1],
               [5, 8, 3, 2],
               [4, 7, 2, 2],
               [8, 9, 3, 2],
               [1, 8, 3, 3],
               [0, 7, 2, 3],
               [2, 9, 1, 3]])
#label이 0=sunny, 1=cloudy, 2=rainy, 3=snowy
#x input [temparature, percentage of rain of snow, sunset]
x=data[:, :3]
y=data[:, 3:4]
print(x)
print(x.shape)
print(y)
print(y.shape)

"""one hot encoding for label"""

new_y=[]
for i in range(y.shape[0]):
  if y[i][0]==0:
    new_y.append([1, 0, 0, 0])
  elif y[i][0]==1:
    new_y.append([0, 1, 0, 0])
  elif y[i][0]==2:
    new_y.append([0, 0, 1, 0])
  elif y[i][0]==3:
    new_y.append([0, 0, 0, 1])
new_y=np.array(new_y)
print(new_y)
print(new_y.shape)

"""Setting Hyperparameter"""

learning_rate=0.01
epoch=50000
weight=np.random.rand(3, 4)#for each weight of classification
bias=np.random.rand(1, 4)
print(bias)
print(bias.shape)

"""Make softmax function"""

def softmax(x):
  max=np.max(x)
  exp_x=np.exp(x-max)
  sum_exp_x=np.sum(exp_x)
  act=exp_x/sum_exp_x
  return act

"""Make Cost Function"""

def error_function(W, b):
  y_pred=x.dot(W)+b
  for i in range(y_pred.shape[0]):
    y_pred[i]=softmax(y_pred[i])
  return -np.sum(new_y*np.log(y_pred)+(1-new_y)*np.log(1-y_pred))

"""Derivative of Error(using numerical derivative)"""

def numerical_derivative(f, W, b):
  h=1e-4
  grad=np.zeros((x.shape[1]+1, new_y.shape[1]))
  for i in range(new_y.shape[1]):
    for j in range(W.shape[0]):
      tmp=W[j, i]
      W[j, i]=float(W[j, i])+h
      w_fx1=f(W, b)
      W[j, i]=tmp

      tmp=W[j, i]
      W[j, i]=float(W[j, i])-h
      w_fx2=f(W, b)
      W[j, i]=tmp
      grad[j, i]=(w_fx1-w_fx2)/(2*h)
  
  for i in range(new_y.shape[1]):
    b_fx1=f(W, float(b[0, i])+h)
    b_fx2=f(W, float(b[0, i])-h)
    grad[x.shape[1], i]=(b_fx1-b_fx2)/(2*h)
  
  return grad

"""Predict"""

def predict(x):
  test=x.dot(weight)+bias
  test=softmax(test)
  print(test)
  max=test[0, 0]
  max_idx=0
  for i in range(1, weight.shape[1]):
    if test[0, i]>max:
      max=test[0, i]
      max_idx=i
  if max_idx==0:
    print("sunny")
  elif max_idx==1:
    print("cloudy")
  elif max_idx==2:
    print("rain")
  elif max_idx==3:
    print("snow")

"""Training"""

for i in range(epoch):
  grad=numerical_derivative(error_function, weight, bias)
  weight=weight-learning_rate*grad[:new_y.shape[1]-1, :]
  bias=bias-learning_rate*grad[x.shape[1]:, :]
  if i%5000==0:
    print('Epoch=', i, ' error_value=', error_function(weight, bias), "W=", weight, "b=", bias)

print(weight)
print(weight.shape)
print(bias)
print(bias.shape)

test=np.array([[8, 2, 2]])
predict(test)