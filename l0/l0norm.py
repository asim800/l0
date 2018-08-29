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
import pickle


from tensorflow.python.keras.engine import Layer
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.keras.engine import InputSpec
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops


from tensorflow.python.keras.utils import conv_utils


#from official.mnist import dataset as mnist_dataset


l = tf.keras.layers

#tf.enable_eager_execution()
plt.ion()

def hard_sigmoid(x):
  return tf.minimum(tf.maximum(x, tf.zeros_like(x)), tf.ones_like(x))



class L0norm():
  def __init__(self, temp=1.0, **kwargs):
    if 'gamma' not in kwargs:
      self.gamma = -0.1
    else:
      self.gamma = kwargs.get(gamma)
    if 'zeta' not in kwargs:
      self.zeta = 1.1
    else:
      self.zeta = kwargs.get(zeta)
    if 'loc_mean' not in kwargs:
      self.loc_mean = 0.
    else:
      self.loc_mean = kwargs.get('loc_mean')
    if 'loc_std' not in kwargs:
      self.loc_stddev = 0.1
    else:
      self.loc_stddev = kwargs.get('loc_std')

    self.temperature = temp

    self.beta=2 / 3
    self.gamma_zeta_ratio = np.log(-self.gamma / self.zeta)

  @property
  def temperature(self):
    return self._temperature

  @temperature.setter
  def temperature(self, value):
    self._temperature = value

  def _get_mask(self, training):
    ''' phi = (log alpha, beta)
    '''
    if training:
      mask = tf.ones_like(self.kernel)
      uni = tf.random_uniform(self.kernel.get_shape(), dtype=self.dtype)
      s = tf.log(uni) - tf.log(1.-uni)
      s   = tf.sigmoid((tf.log(uni) - tf.log(1.-uni) + self.loc ) / self.temperature )   # s RV
      sp   = s * (self.zeta - self.gamma) + self.gamma                            # stretched RV
      penalty = tf.reduce_mean(tf.sigmoid(self.loc - self.temperature * self.gamma_zeta_ratio))
    else:
      sp = tf.sigmoid(self.loc) * (self.zeta - self.gamma) + self.gamma
      penalty=0.
    return hard_sigmoid(sp), penalty

  def _plot_weights(self, name=0):

    w1 = self.weights[0].numpy().flatten()
    w2 = self.loc.numpy().flatten()
    w3 = self.loc2.flatten()
    temp = str(self.temperature)

# print('#weights quantile:', temp, ' : ', np.percentile(w1, [0,25,50,75,100]))
# print('#loc     quantile:', temp, ' : ', np.percentile(w2, [0,25,50,75,100]))

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

    plt.title('temp = '+str(temp))
    plt.savefig('_weights_temp_'+str(temp)+'.png')
    plt.show()

    plt.figure()
    loc=self.loc.numpy().flatten()
    mask_t, p1=self._get_mask(True)
    mask_tf = mask_t.numpy().flatten()
    mask_f, p1=self._get_mask(False)
    mask_ff = mask_f.numpy().flatten()
    plt.hist(loc, 40);
    plt.hist(mask_tf, 40);
    plt.hist(mask_ff, 40);
    plt.grid(True)
    plt.legend(['loc', 'mask_true', 'mask_false'])
    plt.show()
    plt.savefig(str(name)+'_loc.png')
    for i, w in enumerate(self.weights):
      with open(str(name)+'_w'+str(i)+'.pkl', 'wb') as f:
        pickle.dump(w.numpy(), f)

#     np.savetxt(str(name)+'_w'+str(i)+'.txt', w.numpy())


##############################################################
class l0Dense(tf.keras.layers.Dense, L0norm):
  def __init__(self, units, activation=None, temp=1.0, L=16, **kwargs):

#   L0norm.__init__(self, temp, **kwargs)
    L0norm.__init__(self, temp, loc_mean=0.0, loc_std=0.1, **kwargs)
    tf.keras.layers.Dense.__init__(self,  units, activation, **kwargs)
    self.L = L
#   super(l0Dense, self).__init__(units, **kwargs)


  def call(self, inputs, training=True):
    inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
    shape = inputs.get_shape().as_list()
    self.training = training
    for i in range(self.L):
      mask, penalty = self._get_mask(training)
      kernel_new = tf.multiply(self.kernel, mask)
      if len(shape) > 2:
        # Broadcasting is required for the inputs.
        if i==0:
          outputs = standard_ops.tensordot(inputs, kernel_new, [[len(shape) - 1],
                                                                 [0]])
        else:
          outputs += standard_ops.tensordot(inputs, kernel_new, [[len(shape) - 1],
                                                                 [0]])
        # Reshape the output back to the original ndim of the input.
        if not context.executing_eagerly():
          output_shape = shape[:-1] + [self.units]
          outputs.set_shape(output_shape)
      else:
        if i==0:
          outputs = gen_math_ops.mat_mul(inputs, kernel_new)
        else:
          outputs += gen_math_ops.mat_mul(inputs, kernel_new)

      # add bias inside the sample loop
      if self.use_bias:
        outputs = nn.bias_add(outputs, self.bias)

    # take mean
    outputs = outputs / float(self.L)

    if self.activation is not None:
      return self.activation(outputs), penalty  # pylint: disable=not-callable
    return outputs

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


  def _get_mask(self, training):
    ''' phi = (log alpha, beta)
    '''
    if training:
      mask = tf.ones_like(self.kernel)
#     ipdb.set_trace()
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

class l0Conv(Conv, L0norm):
  def __init__(self, rank, filter_size, kernel_size, temp=1.0, L=16,  **kwargs):

    Conv.__init__(self, rank, filter_size, kernel_size, **kwargs)
    L0norm.__init__(self, temp, loc_mean=0.0, loc_std=0.1, **kwargs)
    self.filter_size1 = filter_size
    self.kernel_size1 = kernel_size
    self.L = L

  def call(self, inputs, training=True):
    inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
    shape = inputs.get_shape().as_list()
    self.l0_input_shape = shape
    self.training = training
    for i in range(self.L):
      mask, penalty = self._get_mask(training)
      kernel_new = tf.multiply(self.kernel, mask)
      if i==0:
        outputs = self._convolution_op(inputs, kernel_new)
      else:
        outputs += self._convolution_op(inputs, kernel_new)

      # add bias inside the sample loop
      if self.use_bias:
        if self.data_format == 'channels_first':
          if self.rank == 1:
            # nn.bias_add does not accept a 1D input tensor.
            bias = array_ops.reshape(self.bias, (1, self.filters, 1))
            outputs += bias
          if self.rank == 2:
            outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
          if self.rank == 3:
            # As of Mar 2017, direct addition is significantly slower than
            # bias_add when computing gradients. To use bias_add, we collapse Z
            # and Y into a single dimension to obtain a 4D input tensor.
            outputs_shape = outputs.shape.as_list()
            if outputs_shape[0] is None:
              outputs_shape[0] = -1
            outputs_4d = array_ops.reshape(outputs,
                                           [outputs_shape[0], outputs_shape[1],
                                            outputs_shape[2] * outputs_shape[3],
                                            outputs_shape[4]])
            outputs_4d = nn.bias_add(outputs_4d, self.bias, data_format='NCHW')
            outputs = array_ops.reshape(outputs_4d, outputs_shape)
        else:
          outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

    # take mean
    outputs = outputs / float(self.L)
    if self.activation is not None:
      return self.activation(outputs), penalty
    return outputs, penalty

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    kernel_shape = self.kernel_size + (input_dim, self.filters)

    self.kernel = self.add_weight(
        name='kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)
    self.loc = self.add_variable('loc',
                                  shape= kernel_shape,
                                  initializer=tf.keras.initializers.RandomNormal(mean=self.loc_mean, stddev=self.loc_stddev, seed=None),
                                  regularizer=None,
                                  constraint=None,
                                  dtype=self.dtype,
                                  trainable=True)
    self.loc2 = self.loc.numpy()
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    self.input_spec = InputSpec(ndim=self.rank + 2,
                                axes={channel_axis: input_dim})
    self._convolution_op = nn_ops.Convolution(
        input_shape,
        filter_shape=self.kernel.get_shape(),
        dilation_rate=self.dilation_rate,
        strides=self.strides,
        padding=self.padding.upper(),
        data_format=conv_utils.convert_data_format(self.data_format,
                                                   self.rank + 2))
    self.built = True



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

