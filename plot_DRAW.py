# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 10:51:18 2016

@author: rob
"""
import numpy as np
import matplotlib.pyplot as plt

def clip(a,b=None):
  if b is None:
    return min(max(0,a),27)
  else:
    a = min(max(0,a),27)
    b = min(max(0,b),27)
    return range(a,b)

def sigmoid(value):
  return np.divide(1.0, 1+np.exp(-value))



def plot_DRAW_read(par, X ,direc = '/tmp/',Nplot = 5,cv=None):
  """Function to plot the read boxes for the DRAW model
  input:
  - par: the parameters of the read operation.
      expected list of length seq_len of np arrays in [batch_size,3]
      or expected np array in [seq_len, batch_size, 3]
  - X: the input data in [batch_size, 784]
  - direc: a directory where we can save consequetives canvases
  - Nplot: how many plots you want
  - cv: the canvases. If cv is None, we plot the digits as background"""
  batch_size,im_size = X.shape
  assert im_size == 784, 'For now, this function works with 28x28 MNIST images'
  assert par[0].shape[0] == batch_size, 'Your paramaters and input data have different batch_sizes'

  if isinstance(par,list):
    par = np.stack(par)
  seq_len = par.shape[0]
  if isinstance(cv,list):
    cv = np.stack(cv)
    assert seq_len == cv.shape[0], 'Your canvas size doesnt match your parameters'

  #Convert parameters to integers
  par = par.astype(int)

  im_width = 28   #MNIST image width
  im_ind = np.random.choice(batch_size,Nplot**2)



  for t in range(seq_len):
    canvas = np.zeros((Nplot*im_width,Nplot*im_width))
    for l in range(Nplot**2):
      if cv is None:
        im_boxed = np.reshape(X[im_ind[l]],(im_width,im_width)).copy()
      else:
        im_boxed = sigmoid(np.reshape(cv[t,im_ind[l],:],(im_width,im_width)))
      delta = par[t,im_ind[l],2]
      gx = par[t,im_ind[l],0]
      gy = par[t,im_ind[l],1]
      #Draw left line
      im_boxed[clip(gy-delta,gy+delta),clip(gx-delta)] = 1
      #Draw right line
      im_boxed[clip(gy-delta,gy+delta),clip(gx+delta)] = 1
      # Draw lower line
      im_boxed[clip(gy+delta),clip(gx-delta,gx+delta)] = 1
      #Draw upper line
      im_boxed[clip(gy-delta),clip(gx-delta,gx+delta)] = 1

      #Calculate indices
      ix = (l%Nplot)*im_width
      iy = int(np.floor(l/Nplot))*im_width

      canvas[iy:iy+im_width,ix:ix+im_width] = im_boxed
    imname = direc+'canvas'+str(10+t)+'.png'
    plt.figure()
    plt.imshow(canvas,cmap='gray')
    plt.savefig(imname)
    plt.close('all')




