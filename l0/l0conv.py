
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("always")

import tensorflow.contrib.eager as tfe
from tensorflow.keras import backend as K

import sys
from datetime import datetime
import ipdb
import traceback

from l0norm import l0Dense, l0Conv

l = tf.keras.layers

tf.enable_eager_execution()

from tensorflow.python.framework import ops
###################################################################################
lamba = 0.1
temperature = 0.1
Max_iter   = 1000
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

###################################################################################
num_classes = 10
tf.set_random_seed(0)

class MyModel(tf.keras.Model):
  def __init__(self, temp=1.0):
    super(MyModel, self).__init__()
    self.d0  = l.Dense(512, activation='relu')
#    self.d1  = l.Dense(512, activation='relu')
    self.d1 = l0Dense(512, activation='relu', temp=temp)

    self.d2  = l.Dense(num_classes, activation=None)

    self.temperature = temp

    self.c1 = l0Conv(2, 50, (3,3))

#    self.d2  = l.Dense(num_classes, activation='softmax')
#    self.dtype = K.floatx()

  def call(self, inputs, training=True):
    net1_cum = tf.zeros([128, 512], dtype=tf.float32)
#    penalty  = 0 #tf.zeros([1,], dtype=tf.float32)

    inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
    net = self.d0(inputs)

    # generate MC samples of masks for weights
    self.d1.temperature = self.temperature
    for i in range(L):
      net1, p1= self.d1(net)
      net1_cum = tf.add(net1_cum, net1)


    net1a = net1_cum / float(L)   # average MC samples
    penalty = p1                  # penalty term does not change with MC samples
#    ipdb.set_trace()
    net2= self.d2(net1a)

#    merged_model = tf.keras.layers.concatenate([first_part_output, other_data_input])

    penalty = penalty
    return net2, penalty

class MyModel2(tf.keras.Model):
  def __init__(self, temp=1.0):
    super(MyModel2, self).__init__()

    self.c1 = l0Conv(2, 50, (3,3))
    self.f1 = l.Flatten()
#   self.d1 = l0Dense(512, activation='relu', temp=temp)
    self.d1  = l.Dense(num_classes, activation='softmax')
    self.temperature = temp

    self.c0  = l.Conv2D(50, (3,3) , activation='relu')

#    self.dtype = K.floatx()

  def call(self, inputs, training=True):
    net1_cum = tf.zeros([128, 512], dtype=tf.float32)
#    penalty  = 0 #tf.zeros([1,], dtype=tf.float32)

    inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
    self.c1.temperature = self.temperature
    net, penalty = self.c1(inputs)
    net = self.f1(net)

    # generate MC samples of masks for weights
#   for i in range(N):

    net1 = self.d1(net)


#    merged_model = tf.keras.layers.concatenate([first_part_output, other_data_input])

    penalty = penalty
    return net1, penalty

###################################################################################
batch_size = 128
x = np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)



################################################################################### 
def runmymodel(model, learning_rate=0.01, temperature=0.1, max_iter=1000, inst=0):
  model.temperature = temperature

  test_size = mnist.test.num_examples
  total_batch = int(test_size / batch_size)

  print('test batch: ', total_batch)

  optimizer = tf.train.GradientDescentOptimizer(learning_rate=3e-4)
  optimizer = tf.train.AdamOptimizer(learning_rate=3e-4)


  logdir = './test2'
  writer = tf.contrib.summary.create_file_writer(logdir) ## Tensorboard
  global_step=tf.train.get_or_create_global_step()
  writer.set_as_default()


  for i in range(0,max_iter):
    global_step.assign_add(1)
    batch = mnist.train.next_batch(batch_size)
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
          x = batch[0].reshape(batch_size, 28, 28, 1)
          y = batch[1]

          loss_value, rloss, penalty = loss(model, x, y, training=False)

          loss_buffer.append(loss_value.numpy())
        print('test loss', np.array(loss_buffer).mean(), np.array(loss_buffer).sum())

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

  temp = str(model.temperature)
  w1=model.layers[0].weights[0].numpy().flatten()
  w2 = model.layers[0].loc.numpy().flatten()
  w3 = model.layers[0].loc2.flatten()

  print('#weights quantile:', temp, ' : ', np.percentile(w1, [0,25,50,75,100]))
  print('#loc     quantile:', temp, ' : ', np.percentile(w2, [0,25,50,75,100]))

  ax1 = plt.subplot(3,1,1)
  ax1.hist(w1, 40)
  ax1.grid(True)

  ax2 = plt.subplot(3,1,2)
  ax2.hist(w2, 40)
  ax2.grid(True)

  ax3 = plt.subplot(3,13, sharex=ax2)
  ax3.hist(w3, 40)
  ax3.grid(True)

  plt.title('temp = '+str(temp))
  plt.savefig(str(inst)+'_weights_temp_'+str(temp)+'.png')
  plt.show()



###################################################################################
startTime = datetime.now()

learning_rate = 3e-4
model = MyModel2()
runmymodel(model, learning_rate=learning_rate, temperature=0.1,  max_iter=Max_iter, inst=0)
runmymodel(model, learning_rate=learning_rate, temperature=0.05, max_iter=Max_iter, inst=1)
runmymodel(model, learning_rate=learning_rate, temperature=0.01, max_iter=Max_iter, inst=2)


print('time elapsed: ', datetime.now() - startTime)

ipdb.set_trace()
###################################################################################

