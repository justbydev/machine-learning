# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array

# %% [code]
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix

# %% [code]
from sklearn.model_selection import train_test_split

# %% [code]
epoch=5
learning_rate=0.05
batch_size=32
IMAGE_WIDTH=255
IMAGE_HEIGHT=255
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

# %% [code]
colnames=['filenames', 'plant_label', 'disease_label']
train_df=pd.read_csv("../input/input/train/train.tsv", sep='\t', names=colnames, header=None)
test_df=pd.read_csv("../input/input/test/test.tsv", sep='\t',header=None)
p_label=pd.DataFrame(keras.utils.to_categorical(train_df['plant_label'], 21)).astype('int64')
d_label=pd.DataFrame(keras.utils.to_categorical(train_df['disease_label'], 21)).astype('int64')
p_d_label=p_label+d_label

train_df=train_df.drop(['plant_label', 'disease_label'], axis=1)
merge_label=pd.concat([train_df, p_d_label], axis=1)

# %% [code]
merge_label.head()

# %% [code]
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# %% [code]
train_df, valid_df = train_test_split(merge_label, test_size = 0.15, random_state = 3)

train_df=train_df.reset_index(drop=True)
valid_df=valid_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_valid = valid_df.shape[0]

# %% [code]
columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
train_datagen = ImageDataGenerator(rescale=1./255)
train_gen = train_datagen.flow_from_dataframe(train_df,
                                              directory="../input/input/train",
                                              batch_size = batch_size,
                                              x_col = 'filenames',
                                              y_col = columns,
                                              class_mode='raw',
                                             )
validation_datagen = ImageDataGenerator(rescale=1./255)
valid_gen = validation_datagen.flow_from_dataframe(
    valid_df,
    directory="../input/input/train",
    x_col = 'filenames',
    y_col=columns,
    class_mode='raw',
    batch_size=batch_size,
)

# %% [code]
model=Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', strides=(2, 2),
                input_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.15))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='valid'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(21, activation='sigmoid'))
model.compile(loss='categorical_crossentropy',
             optimizer='nadam',
             metrics=['acc', f1_m])

# %% [code]
history = model.fit_generator(train_gen,
                    epochs = epoch,
                    validation_data = valid_gen,
                    validation_steps=total_valid//batch_size,
                    steps_per_epoch=total_train//batch_size)