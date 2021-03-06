
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
import os
from datetime import datetime
import ipdb
import traceback

from tensorflow.python.framework import ops


from l0norm import l0Dense, l0Conv

tf.enable_eager_execution()
l = tf.keras.layers

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

###################################################################################
num_classes = 10
tf.set_random_seed(0)

class MyModel(tf.keras.Model):
  def __init__(self, temp=1.0):
    super(MyModel, self).__init__()
    self.d0  = l.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.he_normal(seed=None))
#    self.d1  = l.Dense(512, activation='relu')
    self.d1 = l0Dense(512, activation='relu', temp=temp)

    self.d2  = l.Dense(num_classes, activation=None)

    self.temperature = temp


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


###################################################################################
batch_size = 128
x = np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)



################################################################################### 
def runmymodel(model, optimizer, step_counter, learning_rate, temperature=0.1, max_iter=1000, inst=0, checkpoint=None):
  model2.temperature = temperature

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
    x = batch[0]
    y = batch[1]

    loss_value, rloss, penalty = loss(model, x, y, training=False)

    loss_buffer.append(loss_value.numpy())
  print('test loss', np.array(loss_buffer).mean(), np.array(loss_buffer).sum())

  # debug info

  temp = str(model.layers[1].temperature)
  w1=model.layers[1].weights[0].numpy().flatten()
  w2 = model.layers[1].loc.numpy().flatten()
  w3 = model.layers[1].loc2.flatten()

  print('#weights quantile:', temp, ' : ', np.percentile(w1, [0,25,50,75,100]))
  print('#loc     quantile:', temp, ' : ', np.percentile(w2, [0,25,50,75,100]))

  plt.figure()
  ax1 = plt.subplot(3,1,1)
  ax1.hist(w1, 40)
  ax1.grid(True)

  ax2 = plt.subplot(3,1,2)
  ax2.hist(w2, 40)
  ax2.grid(True)

  ax3 = plt.subplot(3,1,3, sharex=ax2)
  ax3.hist(w3, 40)
  ax3.grid(True)
# f, axarr = plt.subplots(3,1, sharex=True)
# axarr[0].hist(w1, 40)
# axarr[0].grid(True)

# axarr[1].hist(w2, 40)
# axarr[1].grid(True)

# axarr[2].hist(w3, 40)
# axarr[2].grid(True)

  plt.title('temp = '+str(temp))
  plt.savefig(str(inst)+'_weights_temp_'+str(temp)+'.png')
  plt.show()


###################################################################################
startTime = datetime.now()

learn_rate = 3e-4
model2 = MyModel()

learning_rate = tfe.Variable(learn_rate, name='learning_rate')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
model_objects = {'model': MyModel(),
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
runmymodel(**model_objects, temperature=0.05, max_iter=Max_iter, inst=0, checkpoint=checkpoint)
runmymodel(**model_objects, temperature=0.01, max_iter=Max_iter, inst=0, checkpoint=checkpoint)


#runmymodel(model, learning_rate=learning_rate, temperature=0.01, max_iter=Max_iter, inst=2)



#root.save(file_prefix=checkpoint_prefix)
print('time elapsed: ', datetime.now() - startTime)
ipdb.set_trace()
###################################################################################

#if __name__ == '__main__':
# tf.enable_eager_execution()
