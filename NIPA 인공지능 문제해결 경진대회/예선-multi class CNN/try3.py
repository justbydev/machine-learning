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
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Activation, BatchNormalization, MaxPooling2D, Dropout
from tensorflow.keras import backend as K

# %% [code]
IMAGE_WIDTH=255
IMAGE_HEIGHT=255
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
BATCH_SIZE=32
EPOCHS=5

# %% [code]
colnames=['filenames', 'plant_label', 'disease_label']
train_df = pd.read_csv("../input/input/train/train.tsv",sep='\t',names=colnames, header=None)
test_df = pd.read_csv("../input/input/test/test.tsv",sep='\t' ,header=None)

# %% [code]
train_df['clas'] = train_df[['plant_label', 'disease_label']].apply(tuple, axis=1)
train_df = train_df.drop('plant_label', 1)
train_df = train_df.drop('disease_label', 1)
train_df['category']=-1
train_df.head()

# %% [code]
for i in range(train_df.shape[0]):
    nw_clas=train_df.loc[i, 'clas']
    if nw_clas==(3, 5):
        train_df.loc[i, 'category']=0
    elif nw_clas==(3, 20):
        train_df.loc[i, 'category']=1
    elif nw_clas==(4, 2):
        train_df.loc[i, 'category']=2
    elif nw_clas==(4, 7):
        train_df.loc[i, 'category']=3
    elif nw_clas==(4, 11):
        train_df.loc[i, 'category']=4
    elif nw_clas==(5, 8):
        train_df.loc[i, 'category']=5
    elif nw_clas==(7, 1):
        train_df.loc[i, 'category']=6
    elif nw_clas==(7, 20):
        train_df.loc[i, 'category']=7
    elif nw_clas==(8, 6):
        train_df.loc[i, 'category']=8
    elif nw_clas==(8, 9):
        train_df.loc[i, 'category']=9
    elif nw_clas==(10, 20):
        train_df.loc[i, 'category']=10
    elif nw_clas==(11, 14):
        train_df.loc[i, 'category']=11
    elif nw_clas==(13, 1):
        train_df.loc[i, 'category']=12
    elif nw_clas==(13, 6):
        train_df.loc[i, 'category']=13
    elif nw_clas==(13, 9):
        train_df.loc[i, 'category']=14
    elif nw_clas==(13, 15):
        train_df.loc[i, 'category']=15
    elif nw_clas==(13, 16):
        train_df.loc[i, 'category']=16
    elif nw_clas==(13, 17):
        train_df.loc[i, 'category']=17
    elif nw_clas==(13, 18):
        train_df.loc[i, 'category']=18
    elif nw_clas==(13, 20):
        train_df.loc[i, 'category']=19

# %% [code]
print(train_df)

# %% [code]
import keras
cat_label=pd.DataFrame(keras.utils.to_categorical(train_df['category'], 20)).astype('int64')
train_df=pd.concat([train_df, cat_label], axis=1)
train_df

# %% [code]
print(train_df.isnull().sum())

# %% [code]
train_df, valid_df = train_test_split(train_df, test_size = 0.15, random_state = 3)


# %% [code]

total_train = train_df.shape[0]
total_valid = valid_df.shape[0]

# %% [code]
columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
train_datagen = ImageDataGenerator(rescale=1./255)
train_gen = train_datagen.flow_from_dataframe(train_df,
                                              directory="../input/input/train",
                                              batch_size = BATCH_SIZE,
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
    batch_size=BATCH_SIZE,
)

# %% [code]
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
model=Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                input_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(20, activation='softmax'))
model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['acc', f1_m])


# %% [code]
history = model.fit_generator(train_gen,
                    epochs = EPOCHS,
                    validation_data = valid_gen,
                    validation_steps=total_valid//BATCH_SIZE,
                    steps_per_epoch=total_train//BATCH_SIZE)