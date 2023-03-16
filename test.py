#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors: Runlong Li
#
# Main file for testing the neural networks

# import the necessary packages
import h5py
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt

# import complex-valued networks and the real-valued counterparts
from keras.models import Model
from RV_FCN import RV_FCN
from CV_FCN import CV_FCN

from scipy.signal import istft, windows

def load_data(testfile='FMCW_test', signal_length=None, scale_num=None):
    # load data
    print("[INFO] Loading testing data ...")
    f = h5py.File('../FMCW_data/dataset/' + testfile + '.hdf5', 'r')
    test_X_real = f['X_real']
    test_X_imag = f['X_imag']

    print('Before reshaping: Shape of test_X (real part):' + str(np.shape(test_X_real)))

    # reshape
    lp = 4  # the overlap length between adjacent matries
    if signal_length != 256:
        N = signal_length // (256-2*lp) + 1
        test_X = np.zeros(shape=(np.shape(test_X_real)[0]*N, 256, 256, 2))
        test_X[0:np.shape(test_X_real)[0], :, :, 0] = test_X_real[:, 0:256, :]
        test_X[0:np.shape(test_X_real)[0], :, :, 1] = test_X_imag[:, 0:256, :]
        for i in range(1, N-1):
            test_X[i*np.shape(test_X_real)[0]:(i+1)*np.shape(test_X_real)[0], :, :, 0] = test_X_real[:,i*(256-2*lp):i*(256-2*lp)+256,:]
            test_X[i*np.shape(test_X_real)[0]:(i+1)*np.shape(test_X_real)[0], :, :, 1] = test_X_imag[:,i*(256-2*lp):i*(256-2*lp)+256,:]
        test_X[(N-1)*np.shape(test_X_real)[0]:N*np.shape(test_X_real)[0],0:signal_length-(N-1)*(256-2*lp), :, 0] = \
            test_X_real[:, (N-1)*(256-2*lp):signal_length, :]
        test_X[(N-1)*np.shape(test_X_real)[0]:N*np.shape(test_X_real)[0],0:signal_length-(N-1)*(256-2*lp), :, 1] = \
            test_X_imag[:, (N-1)*(256-2*lp):signal_length, :]
    else:
        test_X = np.zeros(shape=(np.shape(test_X_real)[0], 256, 256, 2))
        test_X[:, :, :, 0] = test_X_real[:, :, :]
        test_X[:, :, :, 1] = test_X_imag[:, :, :]

    print('After reshaping: Shape of test_X :' + str(np.shape(test_X)))

    f.close()

    # Data Normalization
    print("[INFO] Data Normalization ...")

    norm = []
    for i in range(np.shape(test_X)[0]):
        temp_X = np.sqrt(pow(test_X[i, :, :, 0], 2) + pow(test_X[i, :, :, 1], 2))
        max_X = np.max(temp_X)
        norm.append(max_X)
        test_X[i] = (test_X[i] / max_X) * scale_num

    return test_X, norm


def test(mod_choose, depth, filters, saving_path, signal_length, test_X_input, norm, scale_num):
    # load trained model
    model = RV_FCN(depth=depth, filters=filters, kernel_size=3, use_bn=False,
                   mod='fix', sub_connect=False)
    model.summary()

    # load weights
    para = saving_path
    if scale_num == 1000:
        model.load_weights('trained_model_mul1e3/' + para + '.h5')
    else:
        model.load_weights('trained_model_paper/' + para + '.h5')

    # View the output figure of the middle layer
    # intermediate_layer_model = Model(inputs=model.get_input_at(0),
    #                                  outputs=model.get_layer('subtract_10').output)
    # pred_Y = intermediate_layer_model.predict(test_X_input, batch_size=32, verbose=1)

    # predict
    starttime = datetime.datetime.now()
    pred_Y = model.predict(test_X_input, batch_size=64, verbose=0)
    endtime = datetime.datetime.now()

    # denorm
    for i in range(np.shape(pred_Y)[0]):
        pred_Y[i] = (pred_Y[i] * norm[i]) / scale_num

    # reshape
    lp = 4
    if signal_length != 256:
        N = signal_length // (256-2*lp) + 1
        length = int(np.shape(pred_Y)[0] / N)
        pred_Y_real = np.zeros((length, signal_length, 256))
        pred_Y_imag = np.zeros((length, signal_length, 256))
        pred_Y_real[:, :256-lp, :] = pred_Y[0:length, :256-lp, :, 0]
        pred_Y_imag[:, :256-lp, :] = pred_Y[0:length, :256-lp, :, 1]
        for i in range(1, N-1):
            pred_Y_real[:, i*(256-2*lp)+lp:i*(256-2*lp)+256-lp, :] = pred_Y[i*length:(i+1)*length,lp:256-lp, :, 0]
            pred_Y_imag[:, i*(256-2*lp)+lp:i*(256-2*lp)+256-lp, :] = pred_Y[i*length:(i+1)*length,lp:256-lp, :, 1]
        pred_Y_real[:, (N-1)*(256-2*lp)+lp:signal_length, :] = pred_Y[(N-1)*length:N*length,lp:signal_length-(N-1)*(256-2*lp), :, 0]
        pred_Y_imag[:, (N-1)*(256-2*lp)+lp:signal_length, :] = pred_Y[(N-1)*length:N*length,lp:signal_length-(N-1)*(256-2*lp), :, 1]
    else:
        pred_Y_real = pred_Y[:, :, :, 0]
        pred_Y_imag = pred_Y[:, :, :, 1]

    # create hdf5 file
    if scale_num == 1000:
        if signal_length == 16129:
            f = h5py.File('../FMCW_data/dataset/realdata/pred_chimney_resnet_sar.hdf5', 'w')
        else:
            f = h5py.File('../FMCW_data/dataset/' + saving_path + '_' + str(signal_length) + '.hdf5', 'w')
    else:
        f = h5py.File('../FMCW_data/dataset_1224/' + saving_path + '_' + str(signal_length) + '.hdf5', 'w')
    d1 = f.create_dataset('Y_real', (np.shape(pred_Y_real)[0], np.shape(pred_Y_real)[1], np.shape(pred_Y_real)[2]), 'f')
    d2 = f.create_dataset('Y_imag', (np.shape(pred_Y_imag)[0], np.shape(pred_Y_imag)[1], np.shape(pred_Y_imag)[2]), 'f')
    d1[...] = pred_Y_real[:, :, :]
    d2[...] = pred_Y_imag[:, :, :]
    f.close()

    print("Testing time(ms):" + str(((endtime-starttime).seconds * 1000 + (endtime-starttime).microseconds / 1000)))
    print("[INFO] Completed ...")


if __name__ == '__main__':

    # load data
    signal_length = 256
    scale_num = 1000

    if signal_length == 256:
        test_file = 'FMCW_test'
    elif signal_length == 3906:
        test_file = 'FMCW_test_3906_10'
    elif signal_length == 16129:
        test_file = 'realdata/chimney_small'
    else:
        raise Exception("The signal length is wrong!")

    test_X_input, norm = load_data(testfile=test_file, signal_length=signal_length, scale_num=scale_num)
    test(mod_choose=3, depth=11, filters=16, saving_path='model_CV_FCN_11_16_WholeData_MSE',
         signal_length=signal_length, test_X_input=test_X_input, norm=norm, scale_num=scale_num)
