


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


import functools

import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.keras import backend as K
tf.enable_eager_execution()
l = tf.keras.layers
plt.ion()

###################################################################################
lamba = 0.1
temperature = 0.1
Max_iter   = 100
L = 16
###################################################################################
def loss(model, x, y, training=True):
  logits, penalty = model(x, training)
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
  penalty = penalty
#  ipdb.set_trace()

  return (cross_entropy + lamba*penalty, cross_entropy, penalty)

def grad(model, x):
  with tf.GradientTape() as tape:
    loss_value = loss(model, x, y)
  return tape.gradient(loss_value, model.variables)

class HeReLuNormalInitializer(tf.initializers.random_normal):
  def __init__(self, dtype=tf.float32):
    self.dtype = tf.as_dtype(dtype)

  def get_config(self):
    return dict(dtype=self.dtype.name)

  def __call__(self, shape, dtype=None, partition_info=None):
    del partition_info
    dtype = self.dtype if dtype is None else dtype
    std = tf.rsqrt(tf.cast(tf.reduce_prod(shape[:-1]), tf.float32) + 1e-7)
    return tf.random_normal(shape, stddev=std, dtype=dtype)
###################################################################################
num_classes = 10
tf.set_random_seed(0)
class ModelBasicCNN(tf.keras.Model):
  def __init__(self, nb_classes, nb_filters, **kwargs):
    del kwargs
#       Model.__init__(self, scope, nb_classes, locals())
    super(ModelBasicCNN, self).__init__()
    self.nb_classes = nb_classes
    self.nb_filters = nb_filters
    my_conv = functools.partial(tf.keras.layers.Conv2D, activation=tf.nn.relu) #,
#                               kernel_initializer=HeReLuNormalInitializer)
#       with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
    self.c0 = my_conv(self.nb_filters, 8, strides=(2,2), padding='same')
    self.c1 = my_conv(2 * self.nb_filters, 6, strides=2, padding='valid')
    self.c2 = my_conv(2 * self.nb_filters, 5, strides=1, padding='valid')
    self.f3 = tf.keras.layers.Flatten()
    self.d4 = tf.keras.layers.Dense(
        self.nb_classes)
#       kernel_initializer=HeReLuNormalInitializer)

  def call(self, x, training=True, **kwargs):
    del kwargs
    x = ops.convert_to_tensor(x, dtype=self.dtype)

    net = self.c0(x)
    net = self.c1(net)
    net = self.c2(net)
    net = self.f3(net)
    logits = self.d4(net)


#   return {self.O_LOGITS: logits,
#           self.O_PROBS: tf.nn.softmax(logits=logits)}
    penalty = tfe.Variable(0.0).value()
    return logits, penalty


###################################################################################
batch_size = 128
x = np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)



################################################################################### 
def runmymodel(model, optimizer, step_counter, learning_rate, temperature=0.1, max_iter=1000, inst=0, checkpoint=None):
# model2.temperature = temperature

  test_size = mnist.test.num_examples
  total_batch = int(test_size / batch_size)

  print('test batch: ', total_batch)



  checkpoint_dir = './ckpt'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)

  for i in range(0,max_iter):
    global_step.assign_add(1)
    batch = mnist.train.next_batch(batch_size)
    x = batch[0]
    x = batch[0].reshape(batch_size, 28, 28, 1)
    y = batch[1]

  #  ipdb.set_trace()
    with writer.as_default(), tf.contrib.summary.always_record_summaries():

  #    grads = grad(model, x)
      with tf.GradientTape() as tape:
        loss_value, rloss , penalty = loss(model, x, y)
      grads = tape.gradient(loss_value, model.variables)

  #    ipdb.set_trace()
      optimizer.apply_gradients(zip(grads, model.variables),
                                  global_step=tf.train.get_or_create_global_step())

      loss_value, rloss , penalty = loss(model, x, y)
      tf.contrib.summary.scalar('loss', loss_value)

      if i % 1000 == 0:
        print("Loss at step {:04d}: {:.3f} {:.3f} {:.4f}".format(i, loss_value, rloss, penalty))


        loss_buffer = []
        for i in range(total_batch):

          batch = mnist.test.next_batch(batch_size)
          x = batch[0]
          x = batch[0].reshape(batch_size, 28, 28, 1)
          y = batch[1]

          loss_value, rloss, penalty = loss(model, x, y, training=False)

          loss_buffer.append(loss_value.numpy())
        print('test loss', np.array(loss_buffer).mean(), np.array(loss_buffer).sum())

# learning_rate.assign(learning_rate / 2.0)
  checkpoint.save(file_prefix=checkpoint_prefix)
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

  loss_buffer = []
  for i in range(total_batch):

    batch = mnist.test.next_batch(batch_size)
    x = batch[0]
    x = batch[0].reshape(batch_size, 28, 28, 1)
    y = batch[1]

    loss_value, rloss, penalty = loss(model, x, y, training=False)

    loss_buffer.append(loss_value.numpy())
  print('test loss', np.array(loss_buffer).mean(), np.array(loss_buffer).sum())

  # debug info

# temp = str(model.layers[1].temperature)
  w1=model.layers[1].weights[0].numpy().flatten()
# w2 = model.layers[1].loc.numpy().flatten()
# w3 = model.layers[1].loc2.flatten()

# print('#weights quantile:', temp, ' : ', np.percentile(w1, [0,25,50,75,100]))
# print('#loc     quantile:', temp, ' : ', np.percentile(w2, [0,25,50,75,100]))

  plt.figure()
  ax1 = plt.subplot(3,1,1)
  ax1.hist(w1, 40)
  ax1.grid(True)

  ax2 = plt.subplot(3,1,2)
# ax2.hist(w2, 40)
  ax2.grid(True)

  ax3 = plt.subplot(3,1,3, sharex=ax2)
# ax3.hist(w3, 40)
  ax3.grid(True)
# f, axarr = plt.subplots(3,1, sharex=True)
# axarr[0].hist(w1, 40)
# axarr[0].grid(True)

# axarr[1].hist(w2, 40)
# axarr[1].grid(True)

# axarr[2].hist(w3, 40)
# axarr[2].grid(True)

# plt.title('temp = '+str(temp))
# plt.savefig(str(inst)+'_weights_temp_'+str(temp)+'.png')
  plt.show()


###################################################################################
startTime = datetime.now()

learn_rate = 3e-4
#model2 = ModelBasicCNN()

learning_rate = tfe.Variable(learn_rate, name='learning_rate')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

n_filters = 64
n_classes = 10


model_objects = {'model': ModelBasicCNN(n_classes, n_filters),
                  'optimizer': optimizer,
                  'learning_rate':learning_rate,
                  'step_counter':tf.train.get_or_create_global_step(),
                  }

logdir = './test2'
writer = tf.contrib.summary.create_file_writer(logdir) ## Tensorboard
global_step=tf.train.get_or_create_global_step()
writer.set_as_default()

checkpoint_dir = './ckpt'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
if latest_ckpt:
  print('Using latest checkpoint at ' + latest_ckpt)  
checkpoint = tf.train.Checkpoint(**model_objects)

checkpoint.restore(latest_ckpt)

runmymodel(**model_objects, temperature=0.1,  max_iter=Max_iter, inst=0, checkpoint=checkpoint)
#runmymodel(**model_objects, temperature=0.05, max_iter=Max_iter, inst=0, checkpoint=checkpoint)
#runmymodel(**model_objects, temperature=0.01, max_iter=Max_iter, inst=0, checkpoint=checkpoint)



ipdb.set_trace()


