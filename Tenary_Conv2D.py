#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: conv2d.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from .common import layer_register, VariableHolder
from ..utils.argtools import shape2d, shape4d

__all__ = ['Tenary_Conv2D', 'compute_threshold', 'compute_alpha', 'tenary_opration' ]


def compute_threshold(x):
    # x_max=tf.reduce_max(x,reduce_indices= None, keep_dims= False, name= None)
    x_sum = tf.reduce_sum(tf.abs(x),reduction_indices= None, keep_dims =False ,name= None)
    threshold = tf.div(x_sum,tf.cast(tf.size(x), tf.float32),name= None)
    threshold = tf.multiply(0.7,threshold,name= None)
    return threshold

def compute_alpha(x):
    threshold = compute_threshold(x)
    alpha1_temp1 = tf.where(tf.greater(x,threshold), x, tf.zeros_like(x, tf.float32))
    alpha1_temp2 = tf.where(tf.less(x,-threshold), x, tf.zeros_like(x, tf.float32))
    alpha_array = tf.add(alpha1_temp1,alpha1_temp2,name = None)
    alpha_array_abs = tf.abs(alpha_array)
    alpha_array_abs1 = tf.where(tf.greater(alpha_array_abs,0),tf.ones_like(alpha_array_abs,tf.float32), tf.zeros_like(alpha_array_abs, tf.float32))
    alpha_sum = tf.reduce_sum(alpha_array_abs)
    n = tf.reduce_sum(alpha_array_abs1)
    alpha = tf.div(alpha_sum,n)
    return alpha
	
def tenary_opration(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()

    with tf.name_scope("tenarized") as name:
        with g.gradient_override_map({"Sign": "Identity"}):
            threshold =compute_threshold(x)
            x=tf.sign(tf.add(tf.sign(tf.add(x,threshold)),tf.sign(tf.add(x,-threshold))))
            return x


@layer_register(log_shape=True)
def Tenary_Conv2D(x, out_channel, kernel_shape,
           padding='SAME', stride=1,
           W_init=None, b_init=None,
           nl=tf.identity, split=1, use_bias=True,
           data_format='NHWC'):
    """
    2D convolution on 4D inputs.

    Args:
        x (tf.Tensor): a 4D tensor.
            Must have known number of channels, but can have other unknown dimensions.
        out_channel (int): number of output channel.
        kernel_shape: (h, w) tuple or a int.
        stride: (h, w) tuple or a int.
        padding (str): 'valid' or 'same'. Case insensitive.
        split (int): Split channels as used in Alexnet. Defaults to 1 (no split).
        W_init: initializer for W. Defaults to `variance_scaling_initializer`.
        b_init: initializer for b. Defaults to zero.
        nl: a nonlinearity function.
        use_bias (bool): whether to use bias.

    Returns:
        tf.Tensor named ``output`` with attribute `variables`.

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """
    in_shape = x.get_shape().as_list()
    channel_axis = 3 if data_format == 'NHWC' else 1
    in_channel = in_shape[channel_axis]
    assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
    assert in_channel % split == 0
    assert out_channel % split == 0

    kernel_shape = shape2d(kernel_shape)
    padding = padding.upper()
    filter_shape = kernel_shape + [in_channel / split, out_channel]
    stride = shape4d(stride, data_format=data_format)

    if W_init is None:
        W_init = tf.contrib.layers.variance_scaling_initializer()
    if b_init is None:
        b_init = tf.constant_initializer()

    W = tf.get_variable('W', filter_shape, initializer=W_init)
    Tenary_W = tf.get_variable('Tenary_W', filter_shape, initializer=W_init)
    Tenary_W = tenary_opration(W)
    tf.summary.histogram('Tenary_W',Tenary_W)#I can see Tenary_W is -1,0,+1,but can't saver it correctly in checkpoint.
	                                     #Thanks for helping solving it.
    if use_bias:
        b = tf.get_variable('b', [out_channel], initializer=b_init)
   
    
    if split == 1:
        conv = tf.nn.conv2d(x, Tenary_W, stride, padding, data_format=data_format)
    else:
        inputs = tf.split(x, split, channel_axis)
        kernels = tf.split(Tenary_W, split, 3)
        outputs = [tf.nn.conv2d(i, k, stride, padding, data_format=data_format)
                   for i, k in zip(inputs, kernels)]
        conv = tf.concat(outputs, channel_axis)

    ret = nl(tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv, name='output')
    ret.variables = VariableHolder(W=Tenary_W)
    if use_bias:
        ret.variables.b = b
    return ret
