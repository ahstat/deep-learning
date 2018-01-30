##########################################
# Lagged time series prediction with RNN #
# 2018/01/24 - 2018/01/30                #
##########################################
# Prediction for time series. 
# Focus on predicting time series with lag. 
# 
# Based on question https://stackoverflow.com/questions/41947039 called
# "Keras RNN with LSTM cells for predicting multiple output time series based 
# on multiple input time series"
#
# - In part A, we predict short time series with stateless LSTM.
# - In part B, we try to predict long time series with stateless LSTM, and we
# conclude that training is too long.
# - In part C, we consider stateful LSTM to perform prediction with long
# time series (with user defined number of cuts and batches).
# - In part D, we apply those predictions with multiple inputs and outputs
#
# To deal with part C, we consider a 0/1 time series described by Philippe
# Remy in http://philipperemy.github.io/keras-stateful-lstm/ and we follow
# stateful implementation in Keras according to 
# https://stackoverflow.com/questions/43882796/
import os
import tensorflow
import numpy as np
tensorflow.VERSION
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, TimeDistributed
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
os.chdir('G:/4_rnn')

## Checking and creating directory
def create(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)
create('models')

#######################################################################
#######################################################################
## Helper functions which are not related with stateful computations ##
#######################################################################
#######################################################################
# Note: technical helper functions related to stateful computations are
# postponed in Part C.

################################################
# Sample data with multiple inputs and outputs #
################################################
# Model defined in https://stackoverflow.com/questions/41947039
# x1, x2, x3 and x4 are random time series following uniform distribution,
# y1, y2, y3 such that:
#   * y1[t] = x1[t-2] for t >= 2,
#   * y2[t] = x2[t-1]*x3[t-2] for t >= 2,
#   * y3[t] = x4[t-3] for t >= 3,
def sample_time_series_roll(N, T, seed = 1337):
    np.random.seed(seed)
    
    ##
    # Inputs
    ##
    x1 = np.array(np.random.uniform(size=2*T*N)).reshape(2*N, T)
    x2 = np.array(np.random.uniform(size=2*T*N)).reshape(2*N, T)
    x3 = np.array(np.random.uniform(size=2*T*N)).reshape(2*N, T)
    x4 = np.array(np.random.uniform(size=2*T*N)).reshape(2*N, T)
    
    ##
    # Outputs
    ##
    y1 = np.roll(x1, 2) # y1[t] = x1[t-2]
    y2 = np.roll(x2, 1) * np.roll(x3, 2) # y2[t] = x2[t-1]*x3[t-2]
    y3 = np.roll(x4, 3) # y3[t] = x4[t-3]
    # Note that first 2 terms cannot be predicted for y1 and y2,
    #           first 3 terms cannot be predicted for y3.
    
    ##
    # Training/test sets
    ##
    x1_train, x2_train, x3_train, x4_train = [x[0:N] for x in [x1, x2, x3, x4]]
    x1_test, x2_test, x3_test, x4_test = [x[N:2*N] for x in [x1, x2, x3, x4]]
    y1_train, y2_train, y3_train = [y[0:N] for y in [y1, y2, y3]]
    y1_test, y2_test, y3_test = [y[N:2*N] for y in [y1, y2, y3]]
    
    return(x1_train, x2_train, x3_train, x4_train,
           x1_test, x2_test, x3_test, x4_test,
           y1_train, y2_train, y3_train,
           y1_test, y2_test, y3_test)

##########################################
# Sample data with very long term memory #
##########################################
# Inspired from http://philipperemy.github.io/keras-stateful-lstm/ 
# Here defined as follows:
# x a time series of length T such that: 
#   * P(x[0] = 0) = P(x[0] = 1) = 1/2, and
#   * For t > 0: x[t] = 0 .
# y a time series of length T such that:
#   * for all t: y[t] = x[0]. 
def sample_time_series_1(N, T, seed = 1337):
    np.random.seed(seed)
    
    ##
    # Inputs
    ##
    x = np.zeros(2*N*T).reshape(2*N, T)
    one_indexes = np.random.choice(a = 2*N, size = N, replace = False)
    x[one_indexes, 0] = 1 # very long term memory
    
    ##
    # Outputs
    ##
    # We have chosen to output a time series, it could also be a number
    y = np.repeat(x[:,0], T).reshape(2*N, T)

    ##
    # Training/test sets
    ##
    x_train = x[0:N]
    x_test = x[N:2*N]
    y_train = y[0:N]
    y_test = y[N:2*N]
    
    return(x_train, x_test, y_train, y_test)

####################################################
# Plotting loss and val_loss as function of epochs #
####################################################
def plotting(history):
    plt.plot(history.history['loss'], color = "red")
    plt.plot(history.history['val_loss'], color = "blue")
    red_patch = mpatches.Patch(color='red', label='Training')
    blue_patch = mpatches.Patch(color='blue', label='Test')
    plt.legend(handles=[red_patch, blue_patch])
    plt.xlabel('Epochs')
    plt.ylabel('MSE loss')
    plt.show()

##############################################
##############################################
## A/ Short time series with stateless LSTM ##
##############################################
##############################################

###################################
# Sample data (short time series) #
###################################
N = 13*17*3 # size of samples = 1547
T = 37 # length of each sample is short = 37

x1_train, x2_train, x3_train, x4_train, \
x1_test, x2_test, x3_test, x4_test, \
y1_train, y2_train, y3_train, \
y1_test, y2_test, y3_test = sample_time_series_roll(N, T)

#######################
# Learning y1 from x1 #
#######################

##
# Data
##
m = 1
inputs = x1_train.reshape(N, T, m)
outputs = y1_train.reshape(N, T, m)
inputs_test = x1_test.reshape(N, T, m)
outputs_test = y1_test.reshape(N, T, m)

##
# Model
##
model=Sequential()
dim_in = m
dim_out = m
nb_units = 10 # will also work with 2 units, but too long to train

model.add(LSTM(input_shape=(None, dim_in),
                    return_sequences=True, 
                    units=nb_units))
model.add(TimeDistributed(Dense(activation='linear', units=dim_out)))
model.compile(loss = 'mse', optimizer = 'rmsprop')

##
# Training
##
# 2 seconds for each epoch
np.random.seed(1337)
history = model.fit(inputs, outputs, epochs = 100, batch_size = 32,
                    validation_data=(inputs_test, outputs_test))
plotting(history)
#model.save('models/4_A_y1_from_x1.h5')
load_model('models/4_A_y1_from_x1.h5')
# After 100 epochs: loss: 0.0048 / val_loss: 0.0047. 

##
# Visual checking for a time series
##
i = 0 # time series selected (between 0 and N-1)
idx = range(i, i+1)
x = inputs_test[idx].flatten()
y_hat = model.predict(inputs_test[idx]).flatten()
y = outputs_test[idx].flatten()
plt.plot(range(T), y)
plt.plot(range(T), y_hat)
#plt.plot(range(T), x)

## Conclusion: works very well to learn y[t] = x[t-2].

###############################################
# Learning [y1, y2, y3] from [x1, x2, x3, x4] #
###############################################

##
# Data
##
# Concatenate on a new axis 'dimension'
inputs = np.concatenate((x1_train.reshape(N, T, 1), 
                         x2_train.reshape(N, T, 1),
                         x3_train.reshape(N, T, 1),
                         x4_train.reshape(N, T, 1)), axis=2)
outputs = np.concatenate((y1_train.reshape(N, T, 1), 
                          y2_train.reshape(N, T, 1),
                          y3_train.reshape(N, T, 1)), axis=2)
inputs_test = np.concatenate((x1_test.reshape(N, T, 1), 
                              x2_test.reshape(N, T, 1),
                              x3_test.reshape(N, T, 1),
                              x4_test.reshape(N, T, 1)), axis=2)
outputs_test = np.concatenate((y1_test.reshape(N, T, 1), 
                               y2_test.reshape(N, T, 1),
                               y3_test.reshape(N, T, 1)), axis=2)

##
# Model
##
model=Sequential()
dim_in = 4
dim_out = 3
nb_units = 10
model.add(LSTM(input_shape=(None, dim_in),
                    return_sequences=True, 
                    units=nb_units))
model.add(TimeDistributed(Dense(activation='linear', units=dim_out)))
model.compile(loss = 'mse', optimizer = 'rmsprop')

##
# Training
##
# 2 seconds for each epoch
np.random.seed(1337)
history = model.fit(inputs, outputs, epochs = 500, batch_size = 32,
                    validation_data=(inputs_test, outputs_test))
plotting(history)
#model.save('models/4_A_y123_from_x1234.h5')
load_model('models/4_A_y123_from_x1234.h5')
# After 500 epochs: loss: 0.0061 / val_loss: 0.0061

##
# Visual checking for a time series
##
i = 0 # time series selected (between 0 and N-1)
idx = range(i, i+1)
x = inputs_test[idx].flatten()
y_hat = model.predict(inputs_test[idx]).flatten()
y = outputs_test[idx].flatten()
# y1
plt.plot(range(T), y[0:T])
plt.plot(range(T), y_hat[0:T])
# y2
plt.plot(range(T), y[(T):(2*T)])
plt.plot(range(T), y_hat[(T):(2*T)])
# y3
plt.plot(range(T), y[(2*T):(3*T)])
plt.plot(range(T), y_hat[(2*T):(3*T)])

## Conclusion: works well to learn lagged time series with short sequences.

#############################################
#############################################
## B/ Long time series with stateless LSTM ##
#############################################
#############################################

##################################
# Sample data (long time series) #
##################################
N = 17 # size of samples = 17
T = 37*13*3 # length of each sample is long = 1443
# Note that the product N*T is the same in part A and B

x1_train, x2_train, x3_train, x4_train, \
x1_test, x2_test, x3_test, x4_test, \
y1_train, y2_train, y3_train, \
y1_test, y2_test, y3_test = sample_time_series_roll(N, T)

#######################
# Learning y1 from x1 #
#######################
# stateless LSTM on the long sequence

##
# Data
##
m = 1
inputs = x1_train.reshape(N, T, m)
outputs = y1_train.reshape(N, T, m)
inputs_test = x1_test.reshape(N, T, m)
outputs_test = y1_test.reshape(N, T, m)

##
# Model
##
model=Sequential()
dim_in = m
dim_out = m
nb_units = 10
model.add(LSTM(input_shape=(None, dim_in),
                    return_sequences=True, 
                    units=nb_units))
model.add(TimeDistributed(Dense(activation='linear', units=dim_out)))
model.compile(loss = 'mse', optimizer = 'rmsprop')

##
# Training
##
# 2 seconds for each epoch
np.random.seed(1337)
history = model.fit(inputs, outputs, epochs = 500, batch_size = 32,
                    validation_data=(inputs_test, outputs_test))
plotting(history)
#model.save('models/4_B_y1_from_x1.h5')
load_model('models/4_B_y1_from_x1.h5')
# After 500 epochs: loss: 0.0361 / val_loss: 0.0356

##
# Visual checking for a time series
##
i = 0 # time series selected (between 0 and N-1)
idx = range(i, i+1)
x = inputs_test[idx].flatten()
y_hat = model.predict(inputs_test[idx]).flatten()
y = outputs_test[idx].flatten()
plt.plot(range(T)[0:50], y[0:50])
plt.plot(range(T)[0:50], y_hat[0:50])

## Conclusion: not so good to learn y[t] = x[t-2]. 
# We need more epochs to continue learning.

#############################################################
#############################################################
## Helper functions for stateful network (i.e. for Part C) ##
#############################################################
#############################################################
# Technical part.
# Those helper functions can be skipped in a first read.
# There are designed to manage multiple inputs, outputs, also with user defined 
# batch size and number of cuts.
#
# Idea of stateful LSTM: used for long time series, we want to cut this
# time series into small parts and then doing computations.
# Example:
# t=0         t=T
# [============] <-- time series
# cut into:
# t=0                   t=T
# [===] [===] [===] [===]
#
# Problem:
# If we apply LSTM on each cut, there is a problem: the beginning of a cut 
# doesn't remember the state during the last cut (states = hidden states here).
#
# Solution:
# With stateful=True, we keep in memory this state between batches.
# BUT, if there are multiple time series to train and cut, then we don't want
# to propagate hidden state from one time series to the other one... So we
# need to reinitialize weights with a callback... Although this will slow the
# computations a lot.
# Example:
# 3 time series:
# t=0         t=T  t=0         t=T  t=0         t=T  
# [============]   [============]   [============]   
#
# T is very large, so we cut into pieces:
# t=0                  t=T  t=0                  t=T  t=0                  t=T
# [===] [===] [===] [===]   [===] [===] [===] [===]   [===] [===] [===] [===]  
#
# Then we run LSTM (here with batch_size = 1):
# [===] [===] [===] [===]   [===] [===] [===] [===]   [===] [===] [===] [===]  
#   -->   -->   -->      ||    -->   -->  -->      ||    -->   -->   -->
#   keep states      reinit states   keep states  reinit    keep states
#
# If we run with batch_size > 1, we need to be careful and build something like
# this (here with batch_size = 3:)
# [===] [===] [===] [===]   
# [===] [===] [===] [===]
# [===] [===] [===] [===]
#    ---> keep states
#
# With batch_size = N the number of sampled time series as shown in previous
# figure, it is quite simple to perform cut.
# With batch_size < N, it is more technical: Done with the following function
# stateful_cut.

####################################################################
# Cut (N, T, m) array into (N*T_after_cut, T/T_after_cut, m) array #
####################################################################
# The time series is long (length T) and we want to cut it into smaller ones
# (length T_after_cut). We need to be careful with reinitialization of states
# for our stateful model.
def stateful_cut(arr, batch_size, T_after_cut):
    if len(arr.shape) != 3:
        # N: Independent sample size,
        # T: Time length,
        # m: Dimension
        print("ERROR: please format arr as a (N, T, m) array.")
        # if len(arr.shape) == 1, reshape is as follows:
        # N = 1
        # T = arr.shape[0]
        # m = 1
        # arr.reshape(N, T, m)
        #
        # if len(arr.shape) == 2, there are two ways to reshape:
        # N = arr.shape[0]
        # T = arr.shape[1]
        # m = 1
        # arr.reshape(N, T, m)
        #     or
        # N = 1
        # T = arr.shape[0]
        # m = arr.shape[1]
        # arr.reshape(N, T, m)
    N = arr.shape[0]
    T = arr.shape[1]
    
    # We need T_after_cut * nb_cuts = T
    nb_cuts = int(T / T_after_cut)
    if nb_cuts * T_after_cut != T:
        print("ERROR: T_after_cut must divide T")
    
    # We need batch_size * nb_reset = N
    # If nb_reset = 1, we only reset after the whole epoch, so no need to reset
    nb_reset = int(N / batch_size)
    if nb_reset * batch_size != N:
        print("ERROR: batch_size must divide N")

    # We can observe: 
    # nb_batches = (N*T)/(T_after_cut*batch_size)
    # nb_batches = nb_reset * nb_cuts

    # Cutting (technical)
    cut1 = np.split(arr, nb_reset, axis=0)
    cut2 = [np.split(x, nb_cuts, axis=1) for x in cut1]
    cut3 = [np.concatenate(x) for x in cut2]
    cut4 = np.concatenate(cut3)
    return(cut4)

##
# Reproductible example for stateful_cut with m = 1
##
N = 5*3
T = 2*7
m = 1
arr = sample_time_series_roll(N, T)[0].round(2).reshape(N, T, m)
batch_size = 3
T_after_cut = 7
stateful_cut(arr, batch_size, T_after_cut).reshape(-1, T_after_cut)
# Before cut:
#       t=0  t=1                                                         t=T-1
# n=0   0.26 0.16 0.28 0.46 0.32 0.52 0.26 0.98 0.73 0.12 0.39 0.63 0.13 0.98
# n=1   0.44 0.79 0.79 0.36 0.42 0.58 0.76 0.19 0.89 0.67 0.50 0.18 0.41 0.20
# n=2   0.53 0.83 0.19 0.96 0.43 0.50 0.51 0.02 0.73 0.99 0.16 0.13 0.37 0.69
# n=3   0.00 0.37 0.06 0.79 0.35 0.70 0.49 0.97 0.84 0.61 0.56 1.00 0.25 0.01
# n=4   0.09 0.94 0.97 0.49 0.34 0.72 0.01 0.76 0.67 0.19 0.67 0.91 0.16 0.91
# n=5    *    *    *    *    *    *    *    *    *    *    *    *    *    *
# n=6    *    *    *    *    *    *    *    *    *    *    *    *    *    *
# n=7    *    *    *    *    *    *    *    *    *    *    *    *    *    *
# n=8    *    *    *    *    *    *    *    *    *    *    *    *    *    *
# n=9    *    *    *    *    *    *    *    *    *    *    *    *    *    *
# n=10   *    *    *    *    *    *    *    *    *    *    *    *    *    *
# n=11   *    *    *    *    *    *    *    *    *    *    *    *    *    *
# n=12   *    *    *    *    *    *    *    *    *    *    *    *    *    *
# n=13   *    *    *    *    *    *    *    *    *    *    *    *    *    *
# n=N-1 0.28 0.78 0.85 0.55 0.72 0.43 0.82 0.82 0.02 0.73 0.72 0.72 0.79 0.94
#
# After cut:
#      t=0 t=1                       t=T_after_cut-1
# n=0  0.26 0.16 0.28 0.46 0.32 0.52 0.26             ^
# n=1  0.44 0.79 0.79 0.36 0.42 0.58 0.76             | batch_size = 3
# n=2  0.53 0.83 0.19 0.96 0.43 0.50 0.51             v
# --------------------------------
#     t=T_after_cut                  t=T
# n=0  0.98 0.73 0.12 0.39 0.63 0.13 0.98
# n=1  0.19 0.89 0.67 0.50 0.18 0.41 0.20
# n=2  0.02 0.73 0.99 0.16 0.13 0.37 0.69
# --------------------------------         reset
# n=3  0.00 0.37 0.06 0.79 0.35 0.70 0.49            ^
# n=4  0.09 0.94 0.97 0.49 0.34 0.72 0.01            |
# n=5   *   *   *   *   *   *   *                    | 
# --------------------------------                   | batch_size*nb_cuts = 6
# n=3  0.97 0.84 0.61 0.56 1.00 0.25 0.01            |
# n=4  0.76 0.67 0.19 0.67 0.91 0.16 0.91            |
# n=5   *   *   *   *   *   *   *                    v
# --------------------------------         reset
# n=6   *   *   *   *   *   *   *
# n=7   *   *   *   *   *   *   *
# n=8   *   *   *   *   *   *   *
# --------------------------------
# n=6   *   *   *   *   *   *   *
# n=7   *   *   *   *   *   *   *
# n=8   *   *   *   *   *   *   *
# --------------------------------         reset
# n=9   *   *   *   *   *   *   *
# n=10  *   *   *   *   *   *   *
# n=11  *   *   *   *   *   *   *
# --------------------------------
# n=9   *   *   *   *   *   *   *
# n=10  *   *   *   *   *   *   *
# n=11  *   *   *   *   *   *   *
# --------------------------------         reset
# n=12  *   *   *   *   *   *   *
# n=13  *   *   *   *   *   *   *
# n=N-1 0.28 0.78 0.85 0.55 0.72 0.43 0.82
# --------------------------------
# n=12  *   *   *   *   *   *   *
# n=13  *   *   *   *   *   *   *
# n=N-1 0.82 0.02 0.73 0.72 0.72 0.79 0.94
#                                         reset, so on the whole nb_reset = 5.

##############################################################
# Function to define 'Callback resetting model states' class #
##############################################################
# Need to reset states only when nb_reset > 1
# This callback will slow down computations.
def define_reset_states_class(nb_cuts):
    class ResetStatesCallback(Callback):
        def __init__(self):
            self.counter = 0
    
        def on_batch_begin(self, batch, logs={}):
            # We reset states when nb_cuts batches are completed, as
            # shown in the after cut figure
            if self.counter % nb_cuts == 0:
                self.model.reset_states()
            self.counter += 1
    return(ResetStatesCallback)

#################################################################
# Function to define 'Callback computing validation loss' class #
#################################################################
# Callback to reset states are not properly called with validation data, as
# noted by Philippe Remy in http://philipperemy.github.io/keras-stateful-lstm :
# "be careful as it seems that the callbacks are not properly called when using
# the parameter validation_data in model.fit(). You may have to do your 
# validation/testing manually by calling predict_on_batch() or 
# test_on_batch()."
#
# We introduce a callback to compute validation loss to circumvent this.
# Result will looks like this:
# Epoch 56/100 750/750 [======] - 1s - loss: 1.5133e-04     val_loss: 2.865e-04
def batched(i, arr, batch_size):
    return(arr[i*batch_size:(i+1)*batch_size])

def test_on_batch_stateful(model, inputs, outputs, batch_size, nb_cuts):
    nb_batches = int(len(inputs)/batch_size)
    sum_pred = 0
    for i in range(nb_batches):
        if i % nb_cuts == 0:
            model.reset_states()
        x = batched(i, inputs, batch_size)
        y = batched(i, outputs, batch_size)
        sum_pred += model.test_on_batch(x, y)
    mean_pred = sum_pred / nb_batches
    return(mean_pred)

def define_stateful_val_loss_class(inputs, outputs, batch_size, nb_cuts):
    class ValidationCallback(Callback):
        def __init__(self):
            self.val_loss = []
    
        def on_epoch_end(self, batch, logs={}):
            mean_pred = test_on_batch_stateful(self.model, inputs, outputs, 
                                               batch_size, nb_cuts)
            print('val_loss: {:0.3e}'.format(mean_pred), end = '')
            self.val_loss += [mean_pred]
            
        def get_val_loss(self):
            return(self.val_loss)
            
    return(ValidationCallback)

#######################################
#######################################
## C/ Time series with stateful LSTM ##
#######################################
#######################################

#################################
# Sample data (0/1 time series) #
#################################
N = 5*3*5 # size of samples
T = 10*2*7 # length of each time series
x_train, x_test, y_train, y_test = sample_time_series_1(N, T)

#####################
# Learning y from x #
#####################

##
# Data
##
m = 1 # dimension of each time series
batch_size = 5*3 # number of time series considered together: batch_size | N
T_after_cut = 2*7 # length of each cut part of the time series: T_after_cut | T

# Reshape each time series as (N, T, m)
x_train, x_test, y_train, y_test = \
  [arr.reshape(N, T, m) for arr in [x_train, x_test, y_train, y_test]]

inputs, outputs, inputs_test, outputs_test = \
  [stateful_cut(arr, batch_size, T_after_cut) for arr in \
  [x_train, y_train, x_test, y_test]]

##
# Model
##
dim_in = m
dim_out = m
nb_units = 10
model = Sequential()
model.add(LSTM(batch_input_shape=(batch_size, None, dim_in),
               return_sequences=True, units=nb_units, stateful=True))
model.add(TimeDistributed(Dense(activation='linear', units=dim_out)))
model.compile(loss = 'mse', optimizer = 'rmsprop')

##
# Training
##
epochs = 100

nb_reset = int(N / batch_size)
nb_cuts = int(T / T_after_cut)
if nb_reset > 1:
    ResetStatesCallback = define_reset_states_class(nb_cuts)
    ValidationCallback = define_stateful_val_loss_class(inputs_test, 
                                                        outputs_test, 
                                                        batch_size, nb_cuts)
    validation = ValidationCallback()
    history = model.fit(inputs, outputs, epochs = epochs, 
                        batch_size = batch_size, shuffle=False,
                        callbacks = [ResetStatesCallback(), validation])
    history.history['val_loss'] = ValidationCallback.get_val_loss(validation)
else:
    # If nb_reset = 1, we do not need to reinitialize states
    history = model.fit(inputs, outputs, epochs = epochs, 
                        batch_size = batch_size, shuffle=False,
                        validation_data=(inputs_test, outputs_test))
    
plotting(history) # Evolution of training/test loss
#model.save('models/4_C_0-1_time_series.h5')
load_model('models/4_C_0-1_time_series.h5')
# After 100 epochs: loss: 6.1434e-05 / val_loss: 1.013e-04

##
# Visual checking for a time series
##
# Prediction with stateful model through model.predict need a complete batch.
# It is possible but boring to write a function taking the correct batch and
# then the time series of interest...
#
# Instead, we write a mime model: Same weights, but not stateful.

## Mime model which is stateless but containing stateful weights
model_stateless = Sequential()
model_stateless.add(LSTM(input_shape=(None, dim_in),
                         return_sequences=True, units=nb_units))
model_stateless.add(TimeDistributed(Dense(activation='linear', units=dim_out)))
model_stateless.compile(loss = 'mse', optimizer = 'rmsprop')
model_stateless.set_weights(model.get_weights())

## Prediction of a new set
i = 0 # time series selected (between 0 and N-1)
x = x_train[i]
y = y_train[i]
y_hat = model_stateless.predict(np.array([x]))[0]
plt.plot(range(T), y)
plt.plot(range(T), y_hat)

## Conclusion: works well to learn this time series with ultra long 
# dependencies.

###################################################################
###################################################################
## D/ Time series with stateful LSTM and multiple inputs/outputs ##
###################################################################
###################################################################

##################################
# Sample data (long time series) #
##################################
N = 2*8 # size of samples
T = 37*13*3 # length of each time series
# Note that (N,T)=(16,1443) here and it was (17,1443) in 
# part B (almost the same)

x1_train, x2_train, x3_train, x4_train, \
x1_test, x2_test, x3_test, x4_test, \
y1_train, y2_train, y3_train, \
y1_test, y2_test, y3_test = sample_time_series_roll(N, T)

###########################################
# Learning y1, y2, y3 from x1, x2, x3, x4 #
###########################################

##
# Data
##
batch_size = 8 # number of time series considered together: batch_size | N
T_after_cut = 37 # length of each cut part of the time series: T_after_cut | T
# Here we can select batch_size = N, but we select 8 to test again the code.

# Concatenate on a new axis 'dimension' i.e.
# Reshape each time series as (N, T, dim_in) or (N, T, dim_out)
x_train = np.concatenate((x1_train.reshape(N, T, 1), 
                          x2_train.reshape(N, T, 1),
                          x3_train.reshape(N, T, 1),
                          x4_train.reshape(N, T, 1)), axis=2)
y_train = np.concatenate((y1_train.reshape(N, T, 1), 
                          y2_train.reshape(N, T, 1),
                          y3_train.reshape(N, T, 1)), axis=2)
x_test = np.concatenate((x1_test.reshape(N, T, 1), 
                         x2_test.reshape(N, T, 1),
                         x3_test.reshape(N, T, 1),
                         x4_test.reshape(N, T, 1)), axis=2)
y_test = np.concatenate((y1_test.reshape(N, T, 1), 
                         y2_test.reshape(N, T, 1),
                         y3_test.reshape(N, T, 1)), axis=2)

inputs, outputs, inputs_test, outputs_test = \
  [stateful_cut(arr, batch_size, T_after_cut) for arr in \
  [x_train, y_train, x_test, y_test]]

dim_in = 4 # dimension of input time series
dim_out = 3 # dimension of output time series

##
# Model
##
nb_units = 10

model = Sequential()
model.add(LSTM(batch_input_shape=(batch_size, None, dim_in),
               return_sequences=True, units=nb_units, stateful=True))
model.add(TimeDistributed(Dense(activation='linear', units=dim_out)))
model.compile(loss = 'mse', optimizer = 'rmsprop')

##
# Training
##
epochs = 100

nb_reset = int(N / batch_size)
nb_cuts = int(T / T_after_cut)
if nb_reset > 1:
    ResetStatesCallback = define_reset_states_class(nb_cuts)
    ValidationCallback = define_stateful_val_loss_class(inputs_test, 
                                                        outputs_test, 
                                                        batch_size, nb_cuts)
    validation = ValidationCallback()
    history = model.fit(inputs, outputs, epochs = epochs, 
                        batch_size = batch_size, shuffle=False,
                        callbacks = [ResetStatesCallback(), validation])
    history.history['val_loss'] = ValidationCallback.get_val_loss(validation)
else:
    # When nb_reset = 1, we do not need to reinitialize state
    history = model.fit(inputs, outputs, epochs = epochs, 
                        batch_size = batch_size, shuffle=False,
                        validation_data=(inputs_test, outputs_test))

plotting(history) # Evolution of training/test loss
#model.save('models/4_D_y123_from_x1234.h5')
load_model('models/4_D_y123_from_x1234.h5')

# With batch_size = 8 (i.e. need to reset states): 6s per epoch
# After 100 epochs: loss: 0.0023 / val_loss: 0.0024

# With batch_size = N = 16 (i.e. no need to reset states): 3s per epoch
# After 100 epochs: loss: 0.0057 / val_loss: 0.0058

##
# Visual checking for a time series
##
## Mime model which is stateless but containing stateful weights
model_stateless = Sequential()
model_stateless.add(LSTM(input_shape=(None, dim_in),
               return_sequences=True, units=nb_units))
model_stateless.add(TimeDistributed(Dense(activation='linear', units=dim_out)))
model_stateless.compile(loss = 'mse', optimizer = 'rmsprop')
model_stateless.set_weights(model.get_weights())

## Prediction of a new set
i = 1 # time series selected (between 0 and N-1)
x = x_train[i]
y = y_train[i]
y_hat = model_stateless.predict(np.array([x]))[0]

dim = 2 # dim=0 for y1 ; dim=1 for y2 ; dim=2 for y3.
plt.plot(range(T)[0:100], y[:,dim][0:100])
plt.plot(range(T)[0:100], y_hat[:,dim][0:100])
## Conclusion: works well again to learn this time series