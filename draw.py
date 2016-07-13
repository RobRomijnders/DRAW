#!/usr/bin/env python

"""
Created by Rob Romijnders on July 9 2016

Many of the implementation was inspired by other Github repo's
- In Torch:
https://github.com/vivanov879/draw
- In Theano
https://github.com/jbornschein/draw
- In Tensorflow with pretty tensor
https://github.com/ikostrikov/TensorFlow-VAE-GAN-DRAW
- In Tensorflow
https://github.com/ericjang/draw


"""

import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os
import sys
import socket
if 'rob-laptop' in socket.gethostname():
  sys.path.append('/home/rob/Dropbox/ml_projects/DRAW/')
  direc = '/home/rob/Dropbox/ConvNets/tf/MNIST'
  direc_plot = '/home/rob/Dropbox/ml_projects/DRAW/canvas/'
elif 'rob-com' in socket.gethostname():
  sys.path.append('/home/rob/Documents/DRAW/')
  direc = '/home/rob/Documents/DRAW/MNIST'
  direc_plot = '/home/rob/Documents/DRAW/canvas/'
from plot_DRAW import *


"""Load data"""
#data_directory = os.path.join(direc, "mnist")
#if not os.path.exists(data_directory):
#	os.makedirs(data_directory)
train_data = mnist.input_data.read_data_sets(direc, one_hot=True).train # binarized (0-1) mnist data
tup = train_data.next_batch(2)

"""Hyperparameters"""
hidden_size_enc = 256       # hidden size of the encoder
hidden_size_dec = 256       # hidden size of the decoder
patch_read = 5              # Size of patch to read
patch_write = 5             # Size of patch to write
read_size = 2*patch_read*patch_read
write_size = patch_write*patch_write
num_l=10                    # Dimensionality of the latent space
sl=40                       # Sequence length
batch_size=100
max_iterations=40000
learning_rate=1e-3
dropout = 0.8
eps=1e-8                    # Small number to prevent numerical instability
A,B = 28,28                 # Size of image
cv_size = B*A               # canvas size
assert cv_size == tup[0].shape[1], 'You data doesn;t batch A*B'


REUSE_T=None               # indicator to reuse variables. See comments below
read_params = []           # A list where we save al read paramaters trhoughout code. Allows for nice visualizations at the end
write_params = []          # A list to save write parameters


def filterbank(gx, gy, sigma2,delta, N):
  # gx,gy in [batch_size,1]
  # delta in [batch_size,1]
  # grid in [1,patch_read]
  # x_mu in [batch_size,patch_read]
  # x_mu in [batch_size,patch_read,1]
  # Fx in [batch_size, patch_read, 28]
  # a in [1,1,28]
  a = tf.reshape(tf.cast(tf.range(A),tf.float32),[1,1,-1])
  b = tf.reshape(tf.cast(tf.range(B),tf.float32),[1,1,-1])
  grid = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
  x_mu = gx+(grid-N/2-0.5)*delta # eq 19
  x_mu = tf.reshape(x_mu,[-1,N,1])
  y_mu = gy+(grid-N/2-0.5)*delta # eq 20
  y_mu = tf.reshape(y_mu,[-1,N,1])
  sigma2 = tf.reshape(sigma2,[-1,1,1])
  Fx = tf.exp(-tf.square((x_mu-a)/(-2*sigma2)))
  Fy = tf.exp(-tf.square((y_mu-b)/(-2*sigma2)))
  Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps)
  Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),eps)
  return Fx,Fy

def linear(x,output_dim):
  """ Function to compute linear transforms
  when x is in [batch_size, some_dim] then output will be in [batch_size, output_dim]
  Be sure to use right variable scope n calling this function
  """
  w=tf.get_variable("w", [x.get_shape()[1], output_dim])
  b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
  return tf.nn.xw_plus_b(x,w,b)


def attn_window(scope,h_dec,N):
  with tf.variable_scope(scope,reuse=REUSE_T):
    params=linear(h_dec,5)
  gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(1,5,params)  #eq (21)
  gx=(A+1)/2*(gx_+1)  #eq (22)
  gy=(B+1)/2*(gy_+1)  #eq (23)
  sigma2=tf.exp(log_sigma2)
  delta=(max(A,B)-1)/(N-1)*tf.exp(log_delta) # batch x N    #eq (24)
  Fx,Fy = filterbank(gx,gy,sigma2,delta,N)    #eq (25,26)
  gamma = tf.exp(log_gamma)
  if scope == 'read': read_params.append(tf.concat(1,[gx,gy,delta]))
  if scope == 'write': write_params.append(tf.concat(1,[gx,gy,delta]))
  return (Fx,Fy,gamma)


def read(x,x_hat,h_dec_prev):
  """Function to implement eq 27"""
  Fx,Fy,gamma=attn_window("read",h_dec_prev,patch_read)
  # gamma in [batch_size,1]
  # Fx in [batch_size, patch_read, 28]
  def filter_img(img,Fx,Fy,gamma,N):
    Fxt=tf.transpose(Fx,perm=[0,2,1])
    img=tf.reshape(img,[-1,B,A])  # in [batch_size, 28,28]
    glimpse=tf.batch_matmul(Fy,tf.batch_matmul(img,Fxt)) #in [batch_size, patch_read, patch_read]
    glimpse=tf.reshape(glimpse,[-1,N*N])     # in batch_size, patch_read*patch_read
    return glimpse*tf.reshape(gamma,[-1,1])
  x=filter_img(x,Fx,Fy,gamma,patch_read) # batch x (patch_read*patch_read)
  x_hat=filter_img(x_hat,Fx,Fy,gamma,patch_read)
  # x in [batch_size, patch_read^2]
  # x_hat in [batch_size, patch_read^2]
  return tf.concat(1,[x,x_hat]) # concat along feature axis


def encode(state,input):
  #Run one step of the encoder
  with tf.variable_scope("encoder",reuse=REUSE_T):
    return lstm_enc(input,state)

def decode(state,input):
  #Run one step of the decoder
  with tf.variable_scope("decoder",reuse=REUSE_T):
      return lstm_dec(input, state)

def sample_lat(h_enc):
  """Sample in the latent space, using the reparametrization trick
  """
  #Mind the variable scopes, so that linear() uses the right Tensors
  with tf.variable_scope("mu",reuse=REUSE_T):
    mu=linear(h_enc,num_l)
  with tf.variable_scope("sigma",reuse=REUSE_T):
    logsigma=linear(h_enc,num_l)
    sigma=tf.exp(logsigma)
  return (mu + sigma*e, mu, logsigma, sigma)



def write(h_dec):
  """Function to implement 29"""
  with tf.variable_scope("writeW",reuse=REUSE_T):
      w=linear(h_dec,write_size) # batch x (patch_write*patch_write)
  N=patch_write
  w=tf.reshape(w,[batch_size,N,N])
  Fx,Fy,gamma=attn_window("write",h_dec,patch_write)
  Fyt=tf.transpose(Fy,perm=[0,2,1])
  wr=tf.batch_matmul(Fyt,tf.batch_matmul(w,Fx))
  wr=tf.reshape(wr,[batch_size,B*A])
  #gamma=tf.tile(gamma,[1,B*A])
  return wr*tf.reshape(1.0/gamma,[-1,1])


with tf.variable_scope("placeholders") as scope:
  x = tf.placeholder(tf.float32,shape=(batch_size,cv_size)) # input (batch_size * cv_size)
  e = tf.random_normal((batch_size,num_l), mean=0, stddev=1) # Qsampler noise
  keep_prob = tf.placeholder("float")
  lstm_enc = tf.nn.rnn_cell.LSTMCell(hidden_size_enc, read_size+hidden_size_dec) # encoder Op
  lstm_enc = tf.nn.rnn_cell.DropoutWrapper(lstm_enc,output_keep_prob = keep_prob)
  lstm_dec = tf.nn.rnn_cell.LSTMCell(hidden_size_dec, num_l) # decoder Op
  lstm_dec = tf.nn.rnn_cell.DropoutWrapper(lstm_dec,output_keep_prob = keep_prob)

with tf.variable_scope("States") as scope:
  canvas=[0]*sl # The canves gets sequentiall painted on
  mus,logsigmas,sigmas=[0]*sl,[0]*sl,[0]*sl # parameters for the Gaussian
  # Initialize the states
  h_dec_prev=tf.zeros((batch_size,hidden_size_dec))
  enc_state=lstm_enc.zero_state(batch_size, tf.float32)
  dec_state=lstm_dec.zero_state(batch_size, tf.float32)

with tf.variable_scope("DRAW") as scope:
  #Unroll computation graph
  for t in range(sl):
    c_prev = tf.zeros((batch_size,cv_size)) if t==0 else canvas[t-1]
    x_hat=x-tf.sigmoid(c_prev) # error image
    r=read(x,x_hat,h_dec_prev)
    h_enc,enc_state=encode(enc_state,tf.concat(1,[r,h_dec_prev]))
    z,mus[t],logsigmas[t],sigmas[t]=sample_lat(h_enc)
    h_dec,dec_state=decode(dec_state,z)
    canvas[t]=c_prev+write(h_dec) # store results
    h_dec_prev=h_dec
    REUSE_T=True # Comment on REUSE_T below
#REUSE_T is initialized as None/False. That way, all the get_variable() functions initialize
# a new parameter. After we run the above for-loop once, we'd like the Computation graph to
# to share the matrices in the LSTM's. Therefore we set REUSE_T=True after the first loop


with tf.variable_scope("Loss_comp") as scope:
  def bernoulli_cost(t,o):
    return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))

  # Squash canvas, to obtain image in range [0,1]
  x_recons=tf.nn.sigmoid(canvas[-1])

  cost_recon=tf.reduce_sum(bernoulli_cost(x,x_recons),1) # Bernoulli cost for reconstructed imae with true image as mean
  cost_recon=tf.reduce_mean(cost_recon)

  kls=[0]*sl  #list with saving the KL divergences
  for t in range(sl):
    mu2=tf.square(mus[t])
    sigma2=tf.square(sigmas[t])
    logsigma=logsigmas[t]
    kls[t]=0.5*tf.reduce_sum(mu2+sigma2-2*logsigma-1.0,1)  #-sl*.5
  kl=tf.add_n(kls) # adds tensors over the list
  cost_lat=tf.reduce_mean(kl) # average over minibatches

  cost=cost_recon+cost_lat

with tf.variable_scope("Optimization") as scope:
  optimizer=tf.train.AdamOptimizer(learning_rate)
  grads=optimizer.compute_gradients(cost)
  for i,(g,v) in enumerate(grads):  #g = gradient, v = variable
    if g is not None:
        grads[i]=(tf.clip_by_norm(g,5),v) # Clip the gradients of LSTM
        print(v.name)
  train_op=optimizer.apply_gradients(grads)

print('Finished comp graph')


fetches=[cost_recon,cost_lat,train_op]
costs_recon=[0]*max_iterations
costs_lat=[0]*max_iterations

sess=tf.Session()

sess.run(tf.initialize_all_variables())

for i in range(max_iterations):
  xtrain,_=train_data.next_batch(batch_size)
  feed_dict={x:xtrain,keep_prob: dropout}
  results=sess.run(fetches,feed_dict)
  costs_recon[i],costs_lat[i],_=results
  if i%100==0:
    print("iter=%d : cost_recon: %f cost_lat: %f" % (i,costs_recon[i],costs_lat[i]))

read_params_fetch = sess.run(read_params,feed_dict) #List of length (sl) with np arrays in [batch_size,3]
write_params_fetch = sess.run(write_params,feed_dict)
canvases = sess.run(canvas,feed_dict)
plot_DRAW_read(read_params_fetch, feed_dict[x],direc_plot,5)
plot_DRAW_read(write_params_fetch, feed_dict[x],direc_plot,5,cv=canvases)

# debug = sess.run(mu2,feed_dict)

#Now go to the directory and run  (after install ImageMagick)
#  convert -delay 20 -loop 0 *.png mnist.gif


