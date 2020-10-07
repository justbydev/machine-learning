# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Jic3S9A6Zgsb9v7N4G9bWWhRTRTBCR8Z
"""

from google.colab import files
import pandas as pd
import lightgbm as lgbm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelEncoder

uploaded=files.upload()

test=pd.read_csv('test_x.csv', index_col=0)
train=pd.read_csv('train.csv', index_col=0)
submission=pd.read_csv('sample_submission.csv', index_col=0)

print(test.shape)
print(train.shape)

label=LabelEncoder()
train['gender'] = label.fit_transform(train['gender'])
train['age_group'] = label.fit_transform(train['age_group'])
train['race'] = label.fit_transform(train['race'])
train['religion'] = label.fit_transform(train['religion'])

test['gender'] = label.fit_transform(test['gender'])
test['age_group'] = label.fit_transform(test['age_group'])
test['race'] = label.fit_transform(test['race'])
test['religion'] = label.fit_transform(test['religion'])

plt.figure(figsize=(8, 6))
sns.countplot(data=train, x='age_group', hue=train['voted'])

plt.figure(figsize=(8, 6))
sns.countplot(data=train, x='education', hue=train['voted'])

plt.figure(figsize=(8, 6))
sns.countplot(data=train, x='engnat', hue=train['voted'])

plt.figure(figsize=(8, 6))
sns.countplot(data=train, x='familysize', hue=train['voted'])

plt.figure(figsize=(8, 6))
sns.countplot(data=train, x='gender', hue=train['voted'])

plt.figure(figsize=(8, 6))
sns.countplot(data=train, x='hand', hue=train['voted'])

plt.figure(figsize=(8, 6))
sns.countplot(data=train, x='married', hue=train['voted'])

plt.figure(figsize=(8, 6))
sns.countplot(data=train, x='race', hue=train['voted'])

plt.figure(figsize=(15, 6))
sns.countplot(data=train, x='religion', hue=train['voted'])

plt.figure(figsize=(8, 6))
sns.countplot(data=train, x='urban', hue=train['voted'])

plt.figure(figsize=(10, 6))
sns.countplot(data=train, x='wf_03', hue=train['voted'])

# drop_val = ['QaA', 'QaE', 'QbA', 'QbE', 'QcA', 'QcE', 'QdE', 'QeA','QeE',
#        'QfA', 'QfE', 'QgA', 'QgE', 'QhA', 'QhE', 'QiA', 'QiE', 'QjA', 'QjE',
#        'QkA', 'QkE', 'QlA', 'QlE', 'QmA', 'QmE', 'QnA', 'QnE', 'QoA', 'QoE',
#        'QpA', 'QpE', 'QqA', 'QqE', 'QrA', 'QrE', 'QsA', 'QsE', 'QtA', 'QtE','tp01', 'tp02', 'tp03', 'tp04', 'tp05',
#        'tp06', 'tp07', 'tp08', 'tp09', 'tp10', 'wf_01',
#        'wf_02', 'wf_03', 'wr_01', 'wr_02', 'wr_03', 'wr_04', 'wr_05', 'wr_06',
#        'wr_07', 'wr_08', 'wr_09', 'wr_10', 'wr_11', 'wr_12', 'wr_13']
drop_val=['engnat', 'familysize', 'gender', 'hand', 'race', 'religion', 'urban']
test_x=test.drop(drop_val, axis=1)
train_x=train.drop(drop_val, axis=1)
train_x=train_x.drop('voted', axis=1)
train_y=train['voted']
# model=LogisticRegression()
# model.fit(train_x, train_y)
model=lgbm.LGBMClassifier(n_estimators=500)
model.fit(train_x, train_y)

print(model.score(train_x, train_y))

pred_y=model.predict(test_x)

submission['voted']=pred_y
print(submission)
submission.to_csv('submission.csv')
files.download('submission.csv')