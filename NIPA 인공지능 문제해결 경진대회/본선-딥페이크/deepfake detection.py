import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from keras.utils import np_utils
import cv2
import gc
from keras import backend as bek
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from glob import glob
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU, ELU
from tensorflow.keras import optimizers
from keras import initializers
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D
from PIL import Image
from keras.applications.resnet50 import ResNet50

task_dir = '../../data/.train/.task146/data/train'
image_file_list = glob(task_dir + '/*.jpg', recursive=True)

df = pd.read_csv(task_dir + '/open_train_label.txt', sep=" ", header=None, names=['imagefiles', 'Truth'])


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
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


train_df, valid_df = train_test_split(df, test_size=0.2, random_state=3)

x_train = []
y_train = []
for i in range(train_df.shape[0]):
    imagefiles = train_df.iloc[i]['imagefiles']
    path = task_dir + '/' + imagefiles
    im = cv2.imread(path)
    im = cv2.resize(im, dsize=(32, 32), interpolation=Image.BILINEAR)
    #     im_gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #     im_gray=cv2.bilateralFilter(im_gray, 3, 100, 100)
    x_train.append(img_to_array(im))
    y_train.append(train_df.iloc[i]['Truth'])

x_valid = []
y_valid = []
for i in range(valid_df.shape[0]):
    imagefiles = valid_df.iloc[i]['imagefiles']
    path = task_dir + '/' + imagefiles
    im = cv2.imread(path)
    im = cv2.resize(im, dsize=(32, 32), interpolation=Image.BILINEAR)
    #     im_gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #     im_gray=cv2.bilateralFilter(im_gray, 3, 100, 100)
    x_valid.append(img_to_array(im))
    y_valid.append(valid_df.iloc[i]['Truth'])

x_train = np.array(x_train)
x_valid = np.array(x_valid)

y_train = np.array(y_train)
y_valid = np.array(y_valid)

y_train = to_categorical(y_train, 2)
y_valid = to_categorical(y_valid, 2)

x_train = np.where((x_train <= 20) & (x_train != 0), 0., x_train)
x_train = x_train / 255
x_train = x_train.astype('float32')

x_valid = np.where((x_valid <= 20) & (x_valid != 0), 0., x_valid)
x_valid = x_valid / 255
x_valid = x_valid.astype('float32')

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    zca_whitening=True,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.15,
    rotation_range=10,
    validation_split=0.2)
valgen = ImageDataGenerator(
    featurewise_center=True,
    zca_whitening=True,
)

initial_learningrate = 2e-3


def lr_decay(epoch):  # lrv
    return initial_learningrate * 0.99 ** epoch


from keras.callbacks import ModelCheckpoint, EarlyStopping


def create_model():
    #     mobileNetModel = MobileNet(weights='imagenet', include_top=False)
    #     model = Sequential()
    #     model.add(mobileNetModel)
    #     model.add(GlobalAveragePooling2D())
    #     model.add(Dense(units=2, activation='softmax', kernel_initializer='he_normal'))
    #     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1_m])

    resnet = ResNet50(weights=None, include_top=False, input_shape=(32, 32, 3))
    model_resnet = Sequential()
    model_resnet.add(resnet)
    model_resnet.add(GlobalAveragePooling2D())
    model_resnet.add(Dropout(0.3))
    model_resnet.add(Dense(512, activation='relu'))
    model_resnet.add(Dense(256, activation='relu'))
    model_resnet.add(Dropout(0.3))
    model_resnet.add(Dense(2, activation='softmax'))
    model_resnet.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy', f1_m])

    #     effnet = tf.keras.applications.EfficientNetB3(
    #           include_top=True,
    #           weights=None,
    #           input_shape=(32,32,3),
    #           classes=2,
    #           classifier_activation="sigmoid",)
    #     model = Sequential()
    #     model.add(effnet)
    #     model.add(GlobalAveragePooling2D())
    #     model.add(Flatten())
    #     model.add(Dense(256, activation='relu'))
    #     model.add(Dense(512, activation='relu'))
    #     model.add(Dense(2, activation='softmax'))
    #     model.compile(loss="categorical_crossentropy",
    #               optimizer=RMSprop(lr=initial_learningrate),
    #               metrics=['accuracy', f1_m])

    #     model=Sequential()
    #     model.add(Conv2D(256, (5, 5),
    #                      padding='same',
    #                      input_shape=(32,32,3),
    #                      kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    #     model.add(ELU(0.2))
    #     model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    #     model.add(Dropout(0.3))
    #     model.add(Conv2D(512, (5, 5), padding='same'))
    #     model.add(ELU(0.2))
    #     model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    #     model.add(Dropout(0.3))
    #     model.add(Flatten())
    #     model.add(Dense(256))
    #     model.add(ELU(0.2))
    #     model.add(Dropout(0.3))
    #     model.add(Dense(units=2, activation='softmax'))
    #     model.compile(loss='binary_crossentropy',
    #               optimizer=optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
    #               metrics=['acc', f1_m])
    return model_resnet


from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold

kfold = RepeatedKFold(n_splits=5, n_repeats=10, random_state=40)
cvscores = []
Fold = 1

from tensorflow.keras.callbacks import LearningRateScheduler

for train, val in kfold.split(x_train):
    if Fold >= 11:
        break

    es = EarlyStopping(monitor='val_loss', verbose=1, patience=10)
    filepath_val_acc = "/home/workspace/user-workspace/workspace/rhdnwns_project/models/model_148_final" + str(
        Fold) + ".ckpt"
    checkpoint_val_acc = ModelCheckpoint(filepath_val_acc, monitor='val_accuracy', verbose=1, save_best_only=True,
                                         mode='max', save_weights_only=True)

    gc.collect()
    bek.clear_session()
    print('Fold: ', Fold)

    X_train = x_train[train]
    X_val = x_train[val]
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    Y_train = y_train[train]
    Y_val = y_train[val]

    model = create_model()

    training_generator = datagen.flow(X_train, Y_train, batch_size=32, seed=5, shuffle=True)
    validation_generator = valgen.flow(X_val, Y_val, batch_size=32, seed=5, shuffle=True)
    model.fit(training_generator, epochs=50,
              shuffle=True,
              callbacks=[LearningRateScheduler(lr_decay), es, checkpoint_val_acc],
              validation_data=validation_generator,
              steps_per_epoch=len(X_train) // 32
              )
    del X_train
    del X_val
    del Y_train
    del Y_val

    gc.collect()
    bek.clear_session()

    Fold = Fold + 1




