# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 20:46:41 2019

@author: makigondesk
"""

from __future__ import division
import numpy as np
import pylab as py
import random
import matplotlib
matplotlib.use('Agg') # グラフをウィンドウ表示しない。GUI表示できないサーバモニタ等で重宝する[1]
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv
import math

##### 生データ

num=100
h=0.125/5
delta_t=0.1
K=4



time=100
    
    
"""   
omega=[1*math.pi,1.2*math.pi,1.4*math.pi,1.6*math.pi,1.8*math.pi,2.0*math.pi,
       2.2*math.pi,2.4*math.pi,2.6*math.pi,2.8*math.pi]
"""

omega = [1+0.03*x for x in range(num)]

theta = [random.uniform(0,1.0)*2*math.pi for x in range(num)]
theta_next=[0 for x in range(num)]
theta_np=np.array([theta])



for t in range(time):
    
  for i in range(num):
      sum_sin=0
      for j in range(num):
          sum_sin += math.sin(theta[j]-theta[i])
      theta_next[i] =theta[i]+(omega[i]+K*sum_sin/num)*delta_t
      theta_next[i] = theta_next[i]%(2*math.pi)
    
    
  for i in range(num):
    theta[i] = theta_next[i]
    theta[i] = theta[i]%(2*math.pi)
    
  theta_gyou = np.array([theta[i] for i in range(num)])
  bins = np.linspace(0,2*math.pi,628)
  bins_radix = np.digitize(theta_gyou,bins,right=False)
  theta_gyou = bins[bins_radix]
  theta_np=np.append(theta_np,[theta_gyou],axis=0)
  
plt.xlim(0,6.28) 
a =plt.hist(theta_np[90], bins=100,range=(0,2*math.pi))


xdata = np.linspace(0,2*math.pi,100)
ydata = np.array(a[0])
print(ydata)

T = 2
L = T / 2
tau = 0.045


# "f(x)" function definition.


# "a" coefficient calculation.
def a(n, L, accuracy = 50):
    a, b = -L, L
    dx = (b - a) / accuracy
    integration = 0
    for x in np.linspace(a, b, accuracy):
        integration += ydata * np.cos((n * np.pi * x) / L)
    integration *= dx
    return (1 / L) * integration

# "b" coefficient calculation.
def b(n, L, accuracy = 50):
    a, b = -L, L
    dx = (b - a) / accuracy
    integration = 0
    for x in np.linspace(a, b, accuracy):
        integration += ydata * np.sin((n * np.pi * x) / L)
    integration *= dx
    return (1 / L) * integration

# Fourier series.   
def Sf(x, L, n = 1):
    a0 = a(0, L)
    sum = np.zeros(np.size(x))
    for i in np.arange(1, n + 1):
        print(a(i,L))
        sum += ((a(i, L) * np.cos((i * np.pi * x) / L)) + (b(i, L) * np.sin((i * np.pi * x) / L)))
    return (a0 / 2) + sum 



# x axis.
py.plot(xdata, np.zeros(np.size(xdata)), color = 'black')

# y axis.
py.plot(np.zeros(np.size(xdata)), xdata, color = 'black')

# Original signal.
py.plot(xdata, ydata, linewidth = 1.5, label = 'Signal')

# Approximation signal (Fourier series coefficients).
py.plot(xdata, Sf(xdata, L), '.', color = 'red', linewidth = 1.5, label = 'Fourier series')

# Specify x and y axes limits.
py.xlim([0, 6.28])


py.legend(loc = 'upper right', fontsize = '10')

py.show()
