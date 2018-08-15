#!/usr/bin/python3

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
import ipdb
import traceback


from tensorflow.python.keras.engine import Layer
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.keras.engine import InputSpec
from tensorflow.python.ops import nn


#from official.mnist import dataset as mnist_dataset

lamba = 0.1
temperature = 0.1
Max_iter   = 3000
N = 16

l = tf.keras.layers

tf.enable_eager_execution()
plt.ion()

def hard_sigmoid(x):
  return tf.minimum(tf.maximum(x, tf.zeros_like(x)), tf.ones_like(x))


class l0Norm(Layer):
  def __init__(self, output_dim, **kwargs):
    self.output_dim = output_dim
    super(l0Norm, self).__init__(**kwargs)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_dim = int(input_shape[1])
    self.kernel = self.add_weight(name='kernel',
                                    shape=(input_dim, self.output_dim),
                                    initializer='uniform',
                                    trainable=True)
#    traceback.print_stack()                                      
    super(l0Norm, self).build(input_shape)

  def call(self, x):
    x = ops.convert_to_tensor(x, dtype=self.dtype)
    print(self.kernel.shape, x.shape)

    return K.dot(x, self.kernel)

  def _get_mask(self):
    pass

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    return input_shape[:-1].concatenate(self.output_dim)
#    return tensor_shape.TensorShape([input_shape[0]] + [self.output_dim])
#    return (input_shape[0], self.output_dim)

class L0norm():
  def __init__(self, temp=1.0, **kwargs):
    if 'gamma' not in kwargs:
      self.gamma = -0.1
    else:
      self.gamma = gamma
    if 'zeta' not in kwargs:
      self.zeta = 1.1
    else:
      self.zeta = zeta
    if 'loc' not in kwargs:
      self.loc_mean = 0.
    else:
      self.loc_mean = loc_mean
    self.loc_stddev = 0.1

    self.temperature = temp

    self.beta=2 / 3
    self.gamma_zeta_ratio = np.log(-self.gamma / self.zeta)


##############################################################
class l0Dense(tf.keras.layers.Dense, L0norm):
  def __init__(self, units, activation=None, temp=1.0, **kwargs):

    L0norm.__init__(self, temp, **kwargs)
    tf.keras.layers.Dense.__init__(self,  units, activation, **kwargs)
#   super(l0Dense, self).__init__(units, **kwargs)


  @property
  def temperature(self):
    return self._temperature

  @temperature.setter
  def temperature(self, value):
    self._temperature = value


  def call(self, inputs, training=True):
    inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
    shape = inputs.get_shape().as_list()
    self.training = training
    mask, penalty = self._get_mask()
    if len(shape) > 2:
      # Broadcasting is required for the inputs.
      kernel_new = tf.multiply(self.kernel, mask)
      outputs = standard_ops.tensordot(inputs, kernel_new, [[len(shape) - 1],
                                                             [0]])
      # Reshape the output back to the original ndim of the input.
      if not context.executing_eagerly():
        output_shape = shape[:-1] + [self.units]
        outputs.set_shape(output_shape)
    else:

      kernel_new = tf.multiply(self.kernel, mask)
      outputs = gen_math_ops.mat_mul(inputs, kernel_new)

    if self.use_bias:
      outputs = nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      return self.activation(outputs), penalty  # pylint: disable=not-callable

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if input_shape[-1].value is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self.input_spec = InputSpec(min_ndim=2,
                                axes={-1: input_shape[-1].value})
    self.kernel = self.add_variable('kernel',
                                    shape=[input_shape[-1].value, self.units],
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint,
                                    dtype=self.dtype,
                                    trainable=True)

    self.loc = self.add_variable('loc',
                                shape=[input_shape[-1].value, self.units],
#                                  initializer=tf.keras.initializers.TruncatedNormal(mean=self.loc_mean, stddev=self.loc_stddev, seed=None),
                                  initializer=tf.keras.initializers.RandomNormal(mean=self.loc_mean, stddev=self.loc_stddev, seed=None),
                                  regularizer=None,
                                  constraint=None,
                                  dtype=self.dtype,
                                  trainable=True)

    self.loc2 = self.loc.numpy()
#    self.trainable_weights.extend([self.loc])

    if self.use_bias:
      self.bias = self.add_variable('bias',
                                    shape=[self.units,],
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint,
                                    dtype=self.dtype,
                                    trainable=True)
    else:
      self.bias = None
    self.built = True


  def _get_mask(self):
    ''' phi = (log alpha, beta)
    '''

    if self.training:
      mask = tf.ones_like(self.kernel)
      uni = tf.random_uniform(self.kernel.get_shape(), dtype=self.dtype)
      s = tf.log(uni) - tf.log(1.-uni)
      s   = tf.sigmoid((tf.log(uni) - tf.log(1.-uni) + self.loc ) / self.temperature )   # s RV
      sp   = s * (self.zeta - self.gamma) + self.gamma                            # stretched RV
      penalty = tf.reduce_mean(tf.sigmoid(self.loc - self.temperature * self.gamma_zeta_ratio))
    else:
      sp = tf.sigmoid(self.loc) * (self.zeta - self.gamma) + self.gamma
      penalty=0.
#    plt.hist(ss.numpy().flatten(),20)
#    ipdb.set_trace()
#    self.mask = hard_sigmoid(ss)    
    return hard_sigmoid(sp), penalty

##############################################################

class l0Conv(tf.keras.layers.Dense):
  def __init__(self, units, temp=1.0, **kwargs):
    if 'gamma' not in kwargs:
      self.gamma = -0.1
    else:
      self.gamma = gamma
    if 'zeta' not in kwargs:
      self.zeta = 1.1
    else:
      self.zeta = zeta
    if 'loc' not in kwargs:
      self.loc_mean = 0.
    else:
      self.loc_mean = loc_mean
    self.loc_stddev = 0.1

    self.temperature = temp

    self.beta=2 / 3
    self.gamma_zeta_ratio = np.log(-self.gamma / self.zeta)

    super(l0Dense, self).__init__(units, **kwargs)


  @property
  def temperature(self):
    return self._temperature

  @temperature.setter
  def temperature(self, value):
    self._temperature = value


  def call(self, inputs, training=True):
    inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
    shape = inputs.get_shape().as_list()
    self.training = training
    mask, penalty = self._get_mask()
    if len(shape) > 2:
      # Broadcasting is required for the inputs.
      kernel_new = tf.multiply(self.kernel, mask)
      outputs = standard_ops.tensordot(inputs, kernel_new, [[len(shape) - 1],
                                                             [0]])
      # Reshape the output back to the original ndim of the input.
      if not context.executing_eagerly():
        output_shape = shape[:-1] + [self.units]
        outputs.set_shape(output_shape)
    else:

      kernel_new = tf.multiply(self.kernel, mask)
      outputs = gen_math_ops.mat_mul(inputs, kernel_new)

    if self.use_bias:
      outputs = nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      return self.activation(outputs), penalty  # pylint: disable=not-callable

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if input_shape[-1].value is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self.input_spec = InputSpec(min_ndim=2,
                                axes={-1: input_shape[-1].value})
    self.kernel = self.add_variable('kernel',
                                    shape=[input_shape[-1].value, self.units],
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint,
                                    dtype=self.dtype,
                                    trainable=True)

    self.loc = self.add_variable('loc',
                                shape=[input_shape[-1].value, self.units],
#                                  initializer=tf.keras.initializers.TruncatedNormal(mean=self.loc_mean, stddev=self.loc_stddev, seed=None),
                                  initializer=tf.keras.initializers.RandomNormal(mean=self.loc_mean, stddev=self.loc_stddev, seed=None),
                                  regularizer=None,
                                  constraint=None,
                                  dtype=self.dtype,
                                  trainable=True)

    self.loc2 = self.loc.numpy()
#    self.trainable_weights.extend([self.loc])

    if self.use_bias:
      self.bias = self.add_variable('bias',
                                    shape=[self.units,],
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint,
                                    dtype=self.dtype,
                                    trainable=True)
    else:
      self.bias = None
    self.built = True


  def _get_mask(self):
    ''' phi = (log alpha, beta)
    '''

    if self.training:
      mask = tf.ones_like(self.kernel)
      uni = tf.random_uniform(self.kernel.get_shape(), dtype=self.dtype)
      s = tf.log(uni) - tf.log(1.-uni)
      s   = tf.sigmoid((tf.log(uni) - tf.log(1.-uni) + self.loc ) / self.temperature )   # s RV
      sp   = s * (self.zeta - self.gamma) + self.gamma                            # stretched RV
      penalty = tf.reduce_mean(tf.sigmoid(self.loc - self.temperature * self.gamma_zeta_ratio))
    else:
      sp = tf.sigmoid(self.loc) * (self.zeta - self.gamma) + self.gamma
      penalty=0.
#    plt.hist(ss.numpy().flatten(),20)
#    ipdb.set_trace()
#    self.mask = hard_sigmoid(ss)    
    return hard_sigmoid(sp), penalty

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

class MyModel(tf.keras.Model):
  def __init__(self, temp=1.0):
    super(MyModel, self).__init__()
    self.d0  = l.Dense(512, activation='relu')
#    self.d1  = l.Dense(512, activation='relu')
    self.d1 = l0Dense(512, activation='relu', temp=temp)

    self.d2  = l.Dense(num_classes, activation=None)

    self.temperature = temp


#    self.d2  = l.Dense(num_classes, activation='softmax')
#    self.dtype = K.floatx()

  def call(self, inputs, training=True):
    net1_cum = tf.zeros([128, 512], dtype=tf.float32)
#    penalty  = 0 #tf.zeros([1,], dtype=tf.float32)

    inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
    net = self.d0(inputs)

    # generate MC samples of masks for weights
    self.d1.temperature = self.temperature
    for i in range(N):
      net1, p1= self.d1(net)
      net1_cum = tf.add(net1_cum, net1)


    net1a = net1_cum / float(N)   # average MC samples
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

  loss_buffer = []
  for i in range(total_batch):

    batch = mnist.test.next_batch(batch_size)
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

  f, axarr = plt.subplots(3,1, sharex=True)
  axarr[0].hist(w1, 40)
  axarr[0].grid(True)

  axarr[1].hist(w2, 40)
  axarr[1].grid(True)

  axarr[2].hist(w3, 40)
  axarr[2].grid(True)

  plt.title('temp = '+str(temp))
  plt.savefig(str(inst)+'_weights_temp_'+str(temp)+'.png')
  plt.show()


# ipdb.set_trace()

learning_rate = 3e-4
model = MyModel()
runmymodel(model, learning_rate=learning_rate, temperature=0.1,  max_iter=Max_iter, inst=0)
runmymodel(model, learning_rate=learning_rate, temperature=0.05, max_iter=Max_iter, inst=1)
runmymodel(model, learning_rate=learning_rate, temperature=0.01, max_iter=Max_iter, inst=2)


ipdb.set_trace()
###################################################################################


#checkpoint_dir = ‘/path/to/model_dir’
#checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
#root = tfe.Checkpoint(optimizer=optimizer,
#                      model=model,
#                      optimizer_step=tf.train.get_or_create_global_step())

#root.save(file_prefix=checkpoint_prefix)

###################################################################################


#quantile: 0.1      [-0.28802124 -0.03171337  0.00899071  0.04992321  0.28346333]
#quantile: 0.01  :  [-0.26325771 -0.03194267  0.00892028  0.04979009  0.29554248]
#quantile: 1.0  :  [-0.26444501 -0.03347584  0.00796746  0.04977663  0.2775794 ]








import sys
# if __name__ == '__main__':
#   mymain(sys.argv[1])

