
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("always")


import sys
import os
from datetime import datetime
import ipdb
import traceback

from tensorflow.python.framework import ops

from l0norm import l0Dense, l0Conv

import functools

import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.keras import backend as K
l = tf.keras.layers
plt.ion()



tf.set_random_seed(0)
###################################################################################
num_classes = 10
class ModelBasicCNN(tf.keras.Model):
  def __init__(self, nb_classes, nb_filters, temp, **kwargs):
#   del kwargs
#       Model.__init__(self, scope, nb_classes, locals())
    super(ModelBasicCNN, self).__init__()
    self.nb_classes = nb_classes
    self.nb_filters = nb_filters

    self._temperature = temp
    my_conv = functools.partial(tf.keras.layers.Conv2D, activation=tf.nn.relu) #,
#                               kernel_initializer=HeReLuNormalInitializer)
#       with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):

    self.c0 = my_conv(self.nb_filters, 8, strides=(2,2), padding='same')
#   self.c1 = my_conv(2 * self.nb_filters, 6, strides=2, padding='valid')
    self.c1 = l0Conv(2, 2*nb_filters, 6, strides=2, padding='valid', temp=self.temperature, **kwargs)
#   self.c2 = l0Conv(2, 2*nb_filters, 6, strides=2, padding='valid', temp=self.temperature, **kwargs)
    self.c2 = my_conv(2 * self.nb_filters, 5, strides=1, padding='valid')
    self.f3 = tf.keras.layers.Flatten()
    self.d4 = tf.keras.layers.Dense(
        self.nb_classes)
#       kernel_initializer=HeReLuNormalInitializer)

  def call(self, x, training=True, **kwargs):
    del kwargs
    x = ops.convert_to_tensor(x, dtype=self.dtype)

    net = self.c0(x)

    net1_cum, p1= self.c1(net, training)
#   net1_cum = tf.zeros_like(net1, dtype=tf.float32)
    for i in range(L):
      net1, penalty = self.c1(net, training)
      net1_cum = tf.add(net1_cum, net1)

    net = net1_cum / float(L)   # average MC samples

#   net = self.c1(net, training)
#   net = self.c1(net)
#   if isinstance(net, tuple):
#     net, penalty = net
#   else:
#     penalty = tfe.Variable(0.0).value()

    net = self.c2(net)
#   net, penalty = self.c2(net, training)
    net = self.f3(net)
    logits = self.d4(net)

#   return {self.O_LOGITS: logits,
#           self.O_PROBS: tf.nn.softmax(logits=logits)}
    return logits, penalty


  @property
  def temperature(self):
    return self._temperature

  @temperature.setter
  def temperature(self, value):
    self._temperature = value




###################################################################################
L = 16

###################################################################################


class L0ModelBasicCNN(tf.keras.Model):
  def __init__(self, nb_classes, nb_filters, temp, **kwargs):
#   del kwargs
#       Model.__init__(self, scope, nb_classes, locals())
    super(L0ModelBasicCNN, self).__init__()
    self.nb_classes = nb_classes
    self.nb_filters = nb_filters

    self._temperature = temp
    my_conv = functools.partial(tf.keras.layers.Conv2D, activation=tf.nn.relu) #,
#                               kernel_initializer=HeReLuNormalInitializer)
#       with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):

    self.c0 = my_conv(self.nb_filters, 8, strides=(2,2), padding='same')
#   self.c1 = my_conv(2 * self.nb_filters, 6, strides=2, padding='valid')
    self.c1 = l0Conv(2, 2*nb_filters, 6, strides=2, padding='valid', temp=self.temperature, **kwargs)
#   self.c2 = l0Conv(2, 2*nb_filters, 6, strides=2, padding='valid', temp=self.temperature, **kwargs)
    self.c2 = my_conv(2 * self.nb_filters, 5, strides=1, padding='valid')
    self.f3 = tf.keras.layers.Flatten()
    self.d4 = tf.keras.layers.Dense(
        self.nb_classes)
#       kernel_initializer=HeReLuNormalInitializer)

  def call(self, x, training=True, **kwargs):
    del kwargs
    x = ops.convert_to_tensor(x, dtype=self.dtype)

    net = self.c0(x)

    net1_cum, p1= self.c1(net, training)
#   net1_cum = tf.zeros_like(net1, dtype=tf.float32)
    for i in range(L):
      net1, penalty = self.c1(net, training)
      net1_cum = tf.add(net1_cum, net1)

    net = net1_cum / float(L)   # average MC samples

#   net = self.c1(net, training)
#   net = self.c1(net)
#   if isinstance(net, tuple):
#     net, penalty = net
#   else:
#     penalty = tfe.Variable(0.0).value()

    net = self.c2(net)
#   net, penalty = self.c2(net, training)
    net = self.f3(net)
    logits = self.d4(net)

#   return {self.O_LOGITS: logits,
#           self.O_PROBS: tf.nn.softmax(logits=logits)}
    return logits, penalty


  @property
  def temperature(self):
    return self._temperature

  @temperature.setter
  def temperature(self, value):
    self._temperature = value


###################################################################################
tf.set_random_seed(0)

class ModelBasicDense(tf.keras.Model):
  def __init__(self, temp=1.0):
    super(ModelBasicDense, self).__init__()
    self.num_classes = 10 
    self.d0  = l.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.he_normal(seed=None))
    self.d1  = l.Dense(512, activation='relu')

    self.d2  = l.Dense(self.num_classes, activation=None)

    self.temperature = temp


  def call(self, inputs, training=True):
    inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
    net = self.d0(inputs)

    net1 = self.d1(net)

    net2= self.d2(net1)

    return (net2,)


###################################################################################


class L0ModelBasicDense(tf.keras.Model):
  def __init__(self, temp=1.0):
    super(L0ModelBasicDense, self).__init__()
    self.num_classes = 10 
    self.d0  = l.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.he_normal(seed=None))
#    self.d1  = l.Dense(512, activation='relu')
    self.d1 = l0Dense(512, activation='relu', temp=temp)

    self.d2  = l.Dense(self.num_classes, activation=None)

    self.temperature = temp


  def call(self, inputs, training=True):
    net1_cum = tf.zeros([128, 512], dtype=tf.float32)
#    penalty  = 0 #tf.zeros([1,], dtype=tf.float32)

    inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
    net = self.d0(inputs)

    # generate MC samples of masks for weights
    for i in range(L):
      net1, p1= self.d1(net)
      net1_cum = tf.add(net1_cum, net1)

    net1a = net1_cum / float(L)   # average MC samples
    penalty = p1                  # penalty term does not change with MC samples
#    ipdb.set_trace()
    net2= self.d2(net1a)

#    merged_model = tf.keras.layers.concatenate([first_part_output, other_data_input])

    penalty = penalty
    return (net2, penalty)


###################################################################################
