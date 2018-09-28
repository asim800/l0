


import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

import pickle


tf.enable_eager_execution()
from mymodels import *

l = tf.keras.layers
plt.ion()
sns.set(style="whitegrid")
try:
  tmpdir = os.environ["TMPDIR"]
except KeyError:
  tmpdir = ""
os.environ["TMPDIR"] = os.environ['HOME'] + '/tmp'

print('................', os.environ.get('TMPDIR'))
###################################################################################
def loss(yhat, y, training=True):
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=yhat))
  return cross_entropy


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




###################################################################################



###################################################################################
def runmymodel(model, optimizer, step_counter, learning_rate, temp=0.1, max_iter=1000, inst=0, checkpoint=None):
# model2.temperature = temperatu

  test_size = mnist.test.num_examples
  total_batch = int(test_size / batch_size)

  print('test batch: ', total_batch)
  # get all L0 classes in the model and set temperature to temp
  L0layers = [m for m in model.layers if (type(m) is l0Conv) or (type(m) is l0Dense)]
  for m in L0layers:
    m.temperature = temp

  checkpoint_dir = './ckpt'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)

  logdir = './test2'
  writer = tf.contrib.summary.create_file_writer(logdir) ## Tensorboard
  global_step=tf.train.get_or_create_global_step()
  writer.set_as_default()

  global_step = step_counter

  for i in range(0,max_iter+1):
    global_step.assign_add(1)
    batch = mnist.train.next_batch(batch_size)
    x = batch[0]
    x = batch[0].reshape(input_shape)
    y = batch[1]

    with writer.as_default(), tf.contrib.summary.always_record_summaries():
  #    grads = grad(model, x)
      with tf.GradientTape() as tape:
        model_out = model(x, training=True)
        if len(model_out) == 1:
          yhat = model_out[0]
          penalty = None
        elif len(model_out) == 2:
          yhat, penalty = model_out
        else:
          print('Non-standard model output')
        loss_value = loss(yhat, y)
        if len(model_out) == 2:
          loss_value = loss_value + lamba*penalty

      if i==0:
        print(model.summary())


      grads = tape.gradient(loss_value, model.variables)

      optimizer.apply_gradients(zip(grads, model.variables),
                                  global_step=tf.train.get_or_create_global_step())

      tf.contrib.summary.scalar('loss', loss_value)

#     model_out = model_objects['model'](x, False)
#     if len(model_out) == 1:
#       yhat = model_out[0]
#     elif len(model_out) == 2:
#       yhat, penalty = model_out
#     else:
#       print('Non-standard model output')
      corrects = tf.equal(tf.argmax(y, axis=-1), tf.argmax(yhat, axis=-1))
      corrects = tf.cast(corrects, tf.float32)
      acc = tf.reduce_mean(tf.cast(corrects, tf.float32))
      tf.contrib.summary.scalar('accuracy', acc)


      if i % 1000 == 0:
        if penalty is None:
          print("Loss at step {:04d}:  {:.3f} {:.3f}".format(i, loss_value, acc))
        else:
          print("Loss at step {:04d}: {:.3f} {:.3f} {:.4f}".format(i, loss_value, acc, penalty))

        loss_buffer = []; acc_buffer=[]
        for i in range(total_batch):

          batch = mnist.test.next_batch(batch_size)
          x = batch[0]
          x = batch[0].reshape(input_shape)
          y = batch[1]

          model_out = model(x, False)
          if len(model_out) == 1:
            yhat = model_out[0]
          elif len(model_out) == 2:
            yhat, penalty = model_out
          else:
            print('Non-standard model output')
          corrects = tf.equal(tf.argmax(y, axis=-1), tf.argmax(yhat, axis=-1))
          corrects = tf.cast(corrects, tf.float32)
          acc = tf.reduce_mean(tf.cast(corrects, tf.float32))

          loss_buffer.append(loss_value.numpy())
          acc_buffer.append(acc.numpy())
        print('test loss', np.array(loss_buffer).mean(), np.array(loss_buffer).sum())
        print('test accuracy', np.array(acc_buffer).mean())




# learning_rate.assign(learning_rate / 2.0)
# checkpoint.save(file_prefix=checkpoint_prefix)
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

  loss_buffer = []
  for i in range(total_batch):

    batch = mnist.test.next_batch(batch_size)
    x = batch[0]
    x = batch[0].reshape(input_shape)
    y = batch[1]

    model_out = model(x, training=False)
    if len(model_out) == 1:
      yhat = model_out[0]
      penalty = None
    elif len(model_out) == 2:
      yhat, penalty = model_out
    else:
      print('Non-standard model output')
    loss_value = loss(yhat, y)
    if len(model_out) == 2:
      loss_value = loss_value + lamba*penalty

    loss_buffer.append(loss_value.numpy())
  print('test loss', np.array(loss_buffer).mean(), np.array(loss_buffer).sum())

# if (issubclass(type(model.c1), l0Conv) or issubclass(type(model.c1), l0Dense) ):
#   model.c1._plot_weights()

  eps = 1e-2
  for m in L0layers:
    mask  = m._get_mask(False)[0].numpy()
    mp = mask[mask > eps]
    print('compression ratio: ',  mp.shape[0] / np.prod(mask.shape))
    m._plot_weights(name=m.name+str(inst))

  # debug info


###################################################################################
startTime = datetime.now()

###################################################################################
lamba = 0.1
#temperature = 0.1
Max_iter   = 10
learn_rate = 3e-4
#model2 = ModelBasicCNN()/coding/python/tf/l0v1

batch_size = 128

Npop = 5     # 50 realizations
Navg = 1
Nsamples = [1, 2, 5, 10, 50, 100]
Nsamples = [1, 2, ]



mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
input_shape = (batch_size, 784)
###################################################################################

def main():

  parser = argparse.ArgumentParser(description='Example with non-optional arguments')

#  parser.add_argument('lamba', action="store", type=float)
#  parser.add_argument('Npop', action="store", default=10, type=int)

#  print(parser.parse_args())

  startTime = datetime.now()
  for i,arg in enumerate(sys.argv[1:]):
    if i==0:
      lamba = float(arg)
    elif i==1:
      Max_iter=int(arg)
    elif i==2:
      Npop = int(arg)


    print(arg)

  learning_rate = tfe.Variable(learn_rate, name='learning_rate')
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

  n_filters = 64
  n_classes = 10

  #input_shape = (batch_size, 28, 28, 1)
  #model_obj = ModelBasicCNN(n_classes, n_filters, temp=0.1)
  #model_obj = L0ModelBasicCNN(n_classes, n_filters, temp=0.1)

  model_obj = L0ModelBasicDense()
  #model_obj = ModelBasicDense()


  model_objects = {'model': model_obj,
                    'optimizer': optimizer,
                    'learning_rate':learning_rate,
                    'step_counter':tf.train.get_or_create_global_step(),
                    }


  checkpoint_dir = './ckpt'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
  if latest_ckpt:
    print('Using latest checkpoint at ' + latest_ckpt)
  checkpoint = tf.train.Checkpoint(**model_objects)

  checkpoint.restore(latest_ckpt)

  runmymodel(**model_objects, temp=1.0,  max_iter=Max_iter, inst=0, checkpoint=checkpoint)
  runmymodel(**model_objects, temp=0.5, max_iter=Max_iter, inst=1, checkpoint=checkpoint)
  runmymodel(**model_objects, temp=0.01, max_iter=Max_iter, inst=2, checkpoint=checkpoint )

  print('time elapsed: ', datetime.now() - startTime)
  startTime = datetime.now()

  #####################################

  test_size = mnist.test.num_examples
  test_batch_size = int(test_size / batch_size)

  model = model_obj

  #acc_buffer = [list() for i in Npop for j in range(len(Nsamples))]
  #acc_buffer = [[[list() for k in range(z)] for i in range(Npop)] for z in Nsamples]
  acc_buffer  = [[list() for i in range(Npop)] for k in range(len(Nsamples))]
  acc_buffer2 = [[list() for i in range(Npop)] for k in range(len(Nsamples))]
  m_batch = np.zeros((test_batch_size,))
  m_stats = np.zeros((len(Nsamples), Npop))
  s_stats = np.zeros((len(Nsamples), Npop))
  for k, Nsamps in enumerate(Nsamples):
    loop_cnt = 0
    loopTime = datetime.now()
    for i in range(Npop):

      m_batch_samps = []
      model_objects['model'].nsamps = Nsamps
      print('Nsamps .. ', model_objects['model'].d0.L)
      m_arr = np.zeros((Nsamps, test_batch_size))

      for l in range(test_batch_size):
        batch = mnist.test.next_batch(batch_size)
        x = batch[0]
        x = batch[0].reshape(input_shape)
        y = batch[1]

        #model_out = tf.stop_gradient(model_objects['model'](x, True))
        model_out = model_objects['model'](x, True)
        if len(model_out) == 1:
          yhat = model_out[0]
        elif len(model_out) == 2:
          yhat, penalty = model_out
        else:
          print('Non-standard model output')
        corrects = tf.equal(tf.argmax(y, axis=-1), tf.argmax(yhat, axis=-1))
        corrects = tf.cast(corrects, tf.float32)
        m_batch[l] = tf.reduce_mean(tf.cast(corrects, tf.float32)).numpy()

        loop_cnt += 1

      acc_buffer2[k][i].append(m_batch) # j
      m_batch_samps.append(m_batch)

  #   print('i: ', i,' ', loop_cnt)

      m_batch2 = np.dstack(acc_buffer2[k][i])
      #m_batch2 = np.dstack(m_batch_samps)
      acc_buffer[k][i] = np.dstack(acc_buffer2[k][i]).squeeze(0).mean(axis=1)
  #np.concatenate(acc_buffer[k][i]) # for i 0 aggregate batches

    mean_arr = np.array([np.array(buf).mean() for buf in acc_buffer[k]]) # mean accross samples
    std_arr  = np.array([np.array(buf).std()  for buf in acc_buffer[k]])

    print('time elapsed: ', datetime.now() - loopTime)
    m_stats[k,:] = mean_arr
    s_stats[k,:] = std_arr

  # ipdb.set_trace()
    print('k: ', k, ' ', loop_cnt)

  acc = np.array(m_stats).T
  plt.figure()
  plt.plot(acc)
  plt.savefig('acc.png', bbox_inches='tight')
  acc = pd.DataFrame(acc, columns=Nsamples)
  #plt.figure();ax = sns.boxplot(acc);plt.show()
  plt.figure()
  ax=sns.boxplot(x="variable", y="value",  data=pd.melt(acc));plt.show()
  ax.axes.get_xaxis().set_ticklabels(Nsamples)
  ax.set_xlabel('Number of Samples')
  ax.set_ylabel('Accuracy')
  ax.set_title('Sparse Neural Network with L0 Regularization')
  plt.savefig('cv001.png', bbox_inches='tight')

  with open('l02_data.pkl', 'wb') as f:
    pickle.dump([acc_buffer2, acc_buffer, acc], f)

  if tmpdir == "":
    del os.environ['TMPDIR']
  else:
    os.environ["TMPDIR"] = tmpdir
  print('time elapsed: ', datetime.now() - startTime)

if __name__== "__main__":
  main()

# aa, p1=model_objects['model'].c1._get_mask(False)
# plt.hist(aa.numpy().flatten(),40)
# zt = model_objects['model'].c1._get_mask(False)
# zt = model_objects['model'].c1._get_mask(True)[0].numpy()
# zf = model_objects['model'].c1._get_mask(False)[0].numpy()
# ztp = zt[zt>0.001]
# ztp.shape[0]/np.prod(zt.shape)



# traceback.print_stack()
# print(sys.exc_info()[0])

# import sys
# print("%x" % sys.maxsize, sys.maxsize > 2**32)
