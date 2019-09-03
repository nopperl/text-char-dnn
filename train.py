#!/usr/bin/env python3
from argparse import ArgumentParser
from os.path import join
from time import strftime

import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import BatchNormalization, Bidirectional, Conv1D, Dense, Dropout, GlobalAveragePooling1D, \
    GlobalMaxPooling1D, MaxPooling1D, \
    Lambda, CuDNNGRU, ReLU
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split

from common import sequence_length, tokens

parser = ArgumentParser()
parser.add_argument('-a', '--arch', choices=['cnn', 'vdcnn', 'gru'], default='vdcnn', help='Model architecture to use')
parser.add_argument('-o', '--optimizer', choices=['adam', 'rmsprop'], default='adam', help='Optimizer to use')
parser.add_argument('-d', '--data', nargs='+', choices=['blogs', 'pan13_tr_en', 'pan13_te_en', 'pan14_en', 'enron'], default=['blogs', 'pan13_tr_en'], help='Datasets to train the model on')
parser.add_argument('-c', '--checkpoints', default=5, help='Frequency of checkpoints')
parser.add_argument('-l', '--logs', default='logs', help='Log directory path')
args = parser.parse_args()
arch = args.arch

datasets = args.data
x_paths = ['data/proc/' + dataset + '/x.npy' for dataset in datasets]
y_paths = [x_path.replace('x.npy', 'y.npy') for x_path in x_paths]
checkpoint_frequency = args.checkpoints
logdir = args.logs
x = []
y = []
for i, x_path in enumerate(x_paths):
    x.append(np.load(x_path)[:, :sequence_length])
    y.append(np.load(y_paths[i]))
x = np.vstack(x)
y = np.hstack(y)
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=.3, stratify=y)
del x, y


def encode(ind, length=len(tokens)):
    return K.one_hot(ind, length)


def cnn():
    model = Sequential()
    model.add(Lambda(encode, input_shape=(None,), input_dtype='uint8', output_shape=(None, len(tokens))))
    model.add(Conv1D(filters=256, kernel_size=7))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling1D(pool_size=3))
    model.add(Conv1D(filters=256, kernel_size=7))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling1D(pool_size=3))
    model.add(Conv1D(filters=256, kernel_size=3))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv1D(filters=256, kernel_size=3))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv1D(filters=256, kernel_size=3))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv1D(filters=256, kernel_size=3))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(GlobalAveragePooling1D())
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dense(1, activation='sigmoid'))
    return model


def gru():
    model = Sequential()
    model.add(Lambda(encode, input_shape=(None,), input_dtype='uint8', output_shape=(None, len(tokens))))
    model.add(Bidirectional(CuDNNGRU(256, return_sequences=True), input_shape=(None, len(tokens))))
    model.add(CuDNNGRU(32))
    model.add(Dense(2048, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    return model


def vdcnn():
    model = Sequential()
    model.add(Lambda(encode, input_shape=(None,), input_dtype='uint8', output_shape=(None, len(tokens))))
    model.add(Conv1D(filters=64, kernel_size=3))
    model.add(BatchNormalization())
    blocks = [64, 64, 128, 128, 256, 256, 512, 512]
    pools = range(1, len(blocks), 2)
    for i in range(len(blocks)):
        filter_size = blocks[i]
        model.add(Conv1D(filters=filter_size, kernel_size=3))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Conv1D(filters=filter_size, kernel_size=3))
        model.add(BatchNormalization())
        model.add(ReLU())
        if i in pools and i < len(blocks) - 1:
            model.add(MaxPooling1D(pool_size=3, strides=2))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(2048, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(2048, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    return model


if arch == 'gru':
    model = gru()
elif arch == 'vdcnn':
    model = vdcnn()
else:
    model = cnn()
if args.optimizer == 'adam':
    optimizer = Adam(lr=5e-4, amsgrad=True)
else:
    optimizer = RMSprop(lr=1e-4, rho=.9)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(model.summary())
time = strftime('%Y-%m-%d-%H:%M:%S')
id = f'{time}_{arch}_{"_".join(datasets)}'
checkpoint = ModelCheckpoint('models/' + id + '.h5', save_best_only=True, period=checkpoint_frequency)
tensorboard = TensorBoard(log_dir=join(logdir, id), write_grads=False, write_images=False, write_graph=False)
model.fit(x_tr, y_tr, epochs=30, batch_size=250, validation_data=(x_te, y_te), callbacks=[checkpoint, tensorboard])
