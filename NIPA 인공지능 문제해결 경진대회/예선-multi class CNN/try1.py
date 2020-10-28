# %% [code]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# %% [code]
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


# %% [code]
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Activation, BatchNormalization, MaxPooling2D, Dropout

# %% [code]
from tensorflow.keras import backend as K

# %% [code]
IMAGE_WIDTH=255
IMAGE_HEIGHT=255
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
BATCH_SIZE=32
EPOCHS=10
PATH1='../input/nipadataset/nipa/test/'
PATH2='../input/nipadataset/nipa/train/'

# %% [code]
# tsv 파일을 읽습니다.
colnames=['filenames', 'plant_label', 'disease_label']
train_df = pd.read_csv("../input/nipadataset/nipa/train.tsv",sep='\t',names=colnames, header=None)
test_df = pd.read_csv("../input/nipadataset/nipa/test.tsv",sep='\t' ,header=None)


# %% [code]
# clas 라는 이름의 column으로 plant label과 disease label을 합쳐서 (2,3) 과 같은 형태로 넣습니다.

train_df['clas'] = train_df[['plant_label', 'disease_label']].apply(tuple, axis=1)
print(train_df.info())

# %% [code]
print(train_df['plant_label'].value_counts())
print(train_df['disease_label'].value_counts())

# %% [code]
train_df = train_df.drop('plant_label', 1)
train_df = train_df.drop('disease_label', 1)

# %% [code]
print(train_df.head())
print(train_df['clas'].value_counts())

# %% [code]
# multi label classification 을 수행하기 위해 아래 코드를 수행합니다.
mlb = MultiLabelBinarizer()
train_df = train_df.join(pd.DataFrame(mlb.fit_transform(train_df.pop('clas')),
                          columns=mlb.classes_,
                          index=train_df.index))
print(train_df.columns.values)

# %% [code]
# MultiLabelBinarizer 를 거치면 아래와 같은 형태가 됩니다.
train_df.head()

# %% [code]
train_df, valid_df = train_test_split(train_df, test_size = 0.15, random_state = 3)

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_valid = valid_df.shape[0]

# %% [code]
columns=[1, 2, 3, 4, 5,6,7,8,9,10,11,13,14,15,16,17,18,20]
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.2)
train_gen = train_datagen.flow_from_dataframe(train_df,
                                              directory="../input/nipadataset/nipa/train",
                                              batch_size = BATCH_SIZE,
                                              x_col = 'filenames',
                                              y_col = columns,
                                              class_mode='raw',
                                             )
validation_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.4)
valid_gen = validation_datagen.flow_from_dataframe(
    valid_df,
    directory="../input/nipadataset/nipa/train",
    x_col = 'filenames',
    y_col=columns,
    class_mode='raw',
    batch_size=BATCH_SIZE,
)

# %% [code]
# f1 score를 위한 코드입니다 ?
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
def create_model():
    model = Sequential()
    model.add(Conv2D(64, (3,3), activation='relu', strides=(2,2), input_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(18, activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc',f1_m])
    return model

# %% [code]
model = create_model()

# %% [code]
model.summary()

# %% [code]
history = model.fit_generator(train_gen,
                    epochs = EPOCHS,
                    validation_data = valid_gen,
                    validation_steps=total_valid//BATCH_SIZE,
                    steps_per_epoch=total_train//BATCH_SIZE)