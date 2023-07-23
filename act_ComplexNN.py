#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors: Jianping Wang
#
# Complex-valued activation functions
# Prior-guided loss function: FMCW_loss
# Metrics: SINR, relative_error, MSE

import numpy as np
import math
import tensorflow as tf
import keras
from keras import backend as K
from keras.engine import Layer


def modReLU(z, bias):
    """
    modReLU activation function
    modReLU(z) = relu(|z| + b) * z/|z|
        where b defines the "dead zone" in the complex-valued plane.
        In the case where b>=0, the whole complex-valued plane would
        preserve both amplitude and phase information.
    # Argument
        z:     complex-valued input tensor
        bias:  trainable parameter
    """
    # ndim = K.ndim(z)
    # input_shape = K.shape(z)  # Channel dimension
    # print(bias.get_shape())
    norm = K.abs(z)
    scale = K.relu(norm + bias) / (norm + 1e-6)
    output = tf.dtypes.complex(tf.math.real(z) * scale,
                               tf.math.imag(z) * scale)
    return output


def zReLU(z):
    """
    zReLU activation function
    zReLU(z) = z if 0 <= theta_z <= pi/2, otherwise 0
        which perserves the amplitude and phase information
        when z is located in the first quadrant.
    # Argument
        z:     complex-valued input tensor
    """
    # Compute the phase of input complex-valued number
    phase = tf.math.angle(z)
    # Check whether phase <= pi/2
    le = tf.math.less_equal(phase, math.pi/2)
    # if phase <= pi/2, keep it in comp
    # if phase > pi/2, throw it away and set comp equal to 0
    y = tf.zeros_like(z)
    z = tf.where(le, z, y)
    # Check whether phase >= 0
    ge = tf.math.greater_equal(phase, 0)
    # if phase >= 0, keep it in comp
    # if phase < 0, set output equal to 0
    output = tf.where(ge, z, y)
    return output


class modReLU_Layer(Layer):
    """docstring for modReLU_Layer"""

    def __init__(self, **kwargs):
        super(modReLU_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight(name="bias",
                                    shape=input_shape[3:],
                                    initializer=tf.constant_initializer(0.0),
                                    trainable=True)

    def call(self, inputs, **kwargs):
        return modReLU(inputs, self.bias)

    def compute_output_shape(self, input_shape):
        return input_shape


class zReLU_Layer(Layer):
    """docstring for zReLU_Layer"""

    def __init__(self, **kwargs):
        super(zReLU_Layer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return zReLU(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


def complex_to_channels(z, name="complex2channel", axis=-1):
    """
    Convert data from complex to channels.
    # Argument:
        z:     complex-valued tensor
    # Output:
        a real-valued tensor with the real and imaginary parts concatenated along the dimension of 'axis'
    """
    with tf.name_scope(name):
        z_ri = tf.concat([tf.math.real(z), tf.math.imag(z)], axis=axis)
    return z_ri


def channels_to_complex(z, name="channel2complex", axis=-1):
    """
    Convert data from channels to complex.
    # Argument:
        z:     tensor with the real and imaginary parts concatenated along the dimension of 'axis'
    # Output:
        complex-valued tensor
    """
    ndim = K.ndim(z)
    input_dim = z.get_shape()[axis] // 2

    if ndim == 2:
        z_real = z[:, :input_dim]
        z_imag = z[:, input_dim:]
    elif axis == 1:
        z_real = z[:, :input_dim, ...]
        z_imag = z[:, input_dim:, ...]
    elif axis == -1:
        z_real = z[..., :input_dim]
        z_imag = z[..., input_dim:]
    else:
        raise ValueError(
            'Incorrect axis or dimension of the tensor. axis should be'
            'either 1 or -1.'
        )
    with tf.name_scope(name):
        complex_out = tf.complex(z_real, z_imag)
    return complex_out


class Complex2Channel(Layer):
    """docstring for Complex2Channel"""

    def __init__(self, axis=-1, **kwargs):
        super(Complex2Channel, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        return complex_to_channels(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        out_shape = list(input_shape)
        out_shape[self.axis] = out_shape[self.axis] * 2
        return tuple(out_shape)


class Channel2Complex(Layer):
    """docstring for Channel2Complex"""

    def __init__(self, axis=-1, **kwargs):
        super(Channel2Complex, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        return channels_to_complex(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        out_shape = list(input_shape)
        out_shape[self.axis] = out_shape[self.axis] // 2
        return tuple(out_shape)


# Prior-guided loss function: FMCW_loss
def FMCW_Loss(y_true, y_pred):
    a = 1
    b = 0
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


# define relative error as one of the metrics
def relative_error(y_true, y_pred):
    # calculate relative error
    error = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1),axis=-1),axis=-1)
    # error.get_shape() = (batch_size,)
    signal = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.square(y_true), axis=-1),axis=-1),axis=-1)
    # signal.get_shape() = (batch_size,)
    relative_error = error / signal
    # relative_error.get_shape() = (batch_size,)
    relative_error = tf.reduce_mean(relative_error)
    return relative_error


# define SINR as one of the metrics
def SINR(y_true, y_pred):
    # calculate SINR
    signal = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.square(y_true), axis=-1),axis=-1),axis=-1)
    # signal.get_shape() = (batch_size,)
    interference_noise = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1),axis=-1),axis=-1)
    # interference_noise.get_shape() = (batch_size,)
    SINR = signal / interference_noise
    # SINR.get_shape() = (batch_size,)
    SINR = tf.reduce_mean(SINR)
    return SINR


# define MSE as one of the metrics
def MSE(y_true, y_pred):
    # calculate MSE
    MSE = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1),axis=-1),axis=-1),axis=-1)
    return MSE
