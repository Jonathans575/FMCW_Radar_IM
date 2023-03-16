#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors: Runlong Li
#
# Main file for training the neural networks

# import the necessary packages
import os
import argparse
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras.backend as K
from keras.optimizers import Adam, RMSprop

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import complex-valued networks and the real-valued counterparts
from RV_FCN import RV_FCN
from CV_FCN import CV_FCN

# import the prior-guided loss function and performance metrics
from act_ComplexNN import relative_error, SINR, MSE

# set the gpu configurations
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# set the processing configurations
ap = argparse.ArgumentParser(description="processing configurations ...")
ap.add_argument("-d", "--dataset", type=str, default="../FMCW_data/dataset/",
                help="path to dataset for training")
ap.add_argument("-m", "--model", type=str, default="trained_model/",
                help="path to trained model")
ap.add_argument("-f", "--figure", type=str, default="figure/",
                help="path to loss/metrics plot")
args = vars(ap.parse_args())  # 'vars' returns the properties and property values of the object

# set the training parameters
MAX_EPOCHS = 100
INIT_LR = 0.001
BS = 32


# Prior-guided loss function: FMCW_loss
def make_fmcw_Loss(b):
    def fmcw_loss(y_true, y_pred):
        a = 1
        # loss1 : MSE
        loss1 = tf.reduce_mean(
            tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1), axis=-1), axis=-1))
        # loss2 : L2-1 norm
        y_pred_real = y_pred[:, :, :, 0]
        y_pred_imag = y_pred[:, :, :, 1]
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(y_pred_real) + tf.square(y_pred_imag), axis=1, keepdims=False))  # time dimension
        loss2 = tf.reduce_sum(norm2, axis=1) # frequency dimension
        loss2 = tf.reduce_mean(loss2)
        # check the shape of loss1 and loss2
        if (loss1.get_shape() != ()):
            raise Exception("There's not just one element in loss1")
        if (loss2.get_shape() != ()):
            raise Exception("There's not just one element in loss2")
        return a * loss1 + b * loss2
    return fmcw_loss


def load_data(train_file=None, scale_num=None):

    # load training data
    print("[INFO] Loading Training Data ...")
    from_filename = args["dataset"] + train_file + '.hdf5'
    f = h5py.File(from_filename, 'r')
    train_X_real = f['X_real']
    train_X_imag = f['X_imag']
    train_Y_real = f['Y_real']
    train_Y_imag = f['Y_imag']

    # distribute the real and imaginary part into two channels
    train_X = np.zeros(shape=(np.shape(train_X_real)[0], np.shape(train_X_real)[1], np.shape(train_X_real)[2], 2))
    train_X[:, :, :, 0] = train_X_real[:, :, :]
    train_X[:, :, :, 1] = train_X_imag[:, :, :]
    train_Y = np.zeros(shape=(np.shape(train_Y_real)[0], np.shape(train_Y_real)[1], np.shape(train_Y_real)[2], 2))
    train_Y[:, :, :, 0] = train_Y_real[:, :, :]
    train_Y[:, :, :, 1] = train_Y_imag[:, :, :]

    # clear
    f.close()

    # data normalization
    print("[INFO] Data Normalization ...")

    average = 0
    for i in range(np.shape(train_X)[0]):
        temp_X = np.sqrt(pow(train_X[i, :, :, 0], 2) + pow(train_X[i, :, :, 1], 2))
        temp_Y = np.sqrt(pow(train_Y[i, :, :, 0], 2) + pow(train_Y[i, :, :, 1], 2))
        max_X = np.max(temp_X)
        max_Y = np.max(temp_Y)
        max_complex = max(max_X, max_Y)
        train_X[i] = (train_X[i] / max_complex) * scale_num
        train_Y[i] = (train_Y[i] / max_complex) * scale_num
        average += max_complex
    print(average / np.shape(train_X)[0])

    # data shuffle
    print("[INFO] Data shuffle ...")
    index = [i for i in range(len(train_X))]
    np.random.seed(2020)
    np.random.shuffle(index)
    train_X = train_X[index, ...]
    train_Y = train_Y[index, ...]

    return train_X, train_Y


def train(mod_choose, depth, filters, c, saving_path, train_X_input, train_Y_input):
    # data split
    print("[INFO] Data Split ...")
    train_X, val_X, train_Y, val_Y = train_test_split(train_X_input, train_Y_input, test_size=0.25, random_state=5)
    print('Shape of train_X：' + str(np.shape(train_X)))
    print('Shape of train_Y：' + str(np.shape(train_Y)))
    print('Shape of val_X：' + str(np.shape(val_X)))
    print('Shape of val_Y：' + str(np.shape(val_Y)))

    # compiling model
    print("[INFO] Compiling Model ...")

    model = RV_FCN(depth=depth, filters=filters, kernel_size=3, use_bn=False,
                   mod='fix', sub_connect=False)
    model.summary()

    model_path = args["model"] + 'model_' + saving_path + '.h5'
    if os.path.exists(model_path):
        raise Exception("This model is already trained")
        # model.load_weights(model_path)

    opt = Adam(lr=INIT_LR, decay=INIT_LR / MAX_EPOCHS)
    model.compile(
                  # loss=make_fmcw_Loss(c),
                  loss = keras.losses.MSE,
                  optimizer=opt,
                  metrics=["accuracy", SINR, relative_error, MSE])

    # define the learning rate decay mechanism
    # def scheduler(epoch):
    #     if (int(epoch % 100) == 0) and epoch != 0:
    #         lr = K.get_value(model.optimizer.lr)
    #         K.set_value(model.optimizer.lr, lr * 0.5)
    #         print("lr changed to {} at epoch {}".format(lr * 0.5, epoch))
    #         return K.get_value(model.optimizer.lr)
    #     else:
    #         print("epoch {} lr is {}".format(epoch, K.get_value(model.optimizer.lr)))
    #         return K.get_value(model.optimizer.lr)

    # reduce_lr = keras.callbacks.LearningRateScheduler(scheduler)

    # start training
    print("[INFO] Training Network ...")
    starttime = datetime.datetime.now()
    H = model.fit(train_X,
                  train_Y,
                  batch_size=BS,
                  epochs=MAX_EPOCHS,
                  verbose=2,
                  validation_data=(val_X, val_Y),
                  shuffle=True,
                  callbacks=[
                      # reduce_lr,
                      keras.callbacks.ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, save_weights_only=True),
                      # keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
                  ]
                  )
    endtime = datetime.datetime.now()

    # plot the training loss and performance metrics
    print("[INFO] Ploting Figures ...")
    plt.style.use("ggplot")

    plt.figure()
    plt.plot(np.arange(0, len(H.history["loss"])), H.history["loss"], label="loss")
    plt.plot(np.arange(0, len(H.history["loss"])), H.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.savefig(args["figure"] + 'loss_' + saving_path + '.png')

    plt.figure()
    plt.plot(np.arange(0, len(H.history["SINR"])), H.history['SINR'], label="SINR")
    plt.plot(np.arange(0, len(H.history["SINR"])), H.history["val_SINR"], label="val_SINR")
    plt.title("Training SINR")
    plt.xlabel("Epoch")
    plt.ylabel("SINR")
    plt.legend(loc="best")
    plt.savefig(args["figure"] + 'SINR_' + saving_path + '.png')

    # calculate the training time
    print("Training start time:" + str(starttime))
    print("Training end time:" + str(endtime))
    # print("Training time(s):" + str((endtime - starttime).seconds))
    print("Training time(min):" + str(((endtime - starttime).seconds) / 60))
    print("Training time(h):" + str(((endtime - starttime).seconds) / 3600))
    print("[INFO] Completed ...")


if __name__ == '__main__':

    # load training data
    train_X, train_Y = load_data(train_file='FMCW_train', scale_num=1000)

    # train
    train(mod_choose=5, depth=None, filters=None, c=0, saving_path='model',
          train_X_input=train_X, train_Y_input=train_Y)
