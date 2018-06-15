##########################################
# Simple processing in RNN, LSTM and GRU #
# 2017/12/18 - 2017/12/28                #
##########################################
# This code do not train anything useful, instead highlight structure of 
# simple RNN, LSTM and GRU algorithms.
#
# (using tensorflow 1.1.0 and keras 2.0.3)
#
# Organization of this code:
# - A/ TimeDistributed component
# - B/ Simple RNN
# - C/ Simple RNN with two hidden layers
# - D/ LSTM
# - E/ GRU

####################
# Helper functions #
####################
## Sigmoid function
import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

## Reshape x_train and y_train with correct dimensions
def reshape_training(vect_seed, dim, sample_size):
    '''
    Take vect_seed of length T, for example [1, 2, 3]
    Repeat it 'dim' times to obtain, for example with dim=2:
    [[1, 1],
     [2, 2],
     [3, 3]]
    Repeat it 'sample_size' times to obtain the elements:
    [[1, 1],    [[1, 1],    [[1, 1],    [[1, 1],
     [2, 2],     [2, 2],     [2, 2],     [2, 2],  ...
     [3, 3]]     [3, 3]]     [3, 3]]     [3, 3]]
    '''
    T = len(vect_seed)
    vect_train = np.array([vect_seed]*dim)
    vect_train = np.repeat(vect_train, sample_size)
    vect_train = np.reshape(vect_train, (sample_size, T, dim), order = 'F')
    # vect_train[0]
    return(vect_train)

#################################################
#################################################
## A/ Explanation of TimeDistributed component ##
#################################################
#################################################
import tensorflow
import numpy as np
tensorflow.VERSION
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
# Reference: https://stackoverflow.com/questions/38294046

#######################################
# Inputs and outputs for this section #
#######################################
# Each input is a sequence of size T = 6 (input of dimension 6 x 1)
# Each output is a sequence of the same size T.
# The training set comprises 256 inputs/outputs.

## Training input
# The input format should be three-dimensional: dimensions values are:
# - sample size (256),
# - number of time steps and dimension (6 x 1)
# So x_train input must be a 3-dim vector of size 256 x 6 x 1
sample_size = 256
x_seed = [1, 0.8, 0.6, 0.2, 1, 0]
x_train = np.array([[x_seed] * sample_size]).reshape(sample_size,len(x_seed),1)
# Each sample is a time series of length 6: (x_t)_t = [x_0, x_1, ..., x_5],
# where each x_t is a real number. In this example, we have the same (x_t)
# [1, 0.8, 0.6, 0.2, 1, 0] for each sampled element.

## Training output
# The output has the same format as the input format in this example
y_seed = [0.7, 0.6, 0.5, 0.3, 0.8, 0.2] 
y_train = np.array([[y_seed]*sample_size]).reshape(sample_size,len(y_seed),1)
# The output size is 256 (because 1 input is linked with 1 output),
# and each output is a time series of length 6: (y_t)_t = [y_0, y_1, ..., y_5],
# where each y_t is a real number. In this example, we have the same (y_t)
# [0.7, 0.6, 0.5, 0.3, 0.8, 0.2]  for each sampled element.

## Note: y_seed chosen to be like sigmoid(2*x_seed-1)
[sigmoid(2*x-1) for x in x_seed]

#################################
# Model definition and training #
#################################
## Model definition
model=Sequential()
model.add(TimeDistributed(Dense(activation='sigmoid', units=1),
                          input_shape=(None, 1)))

## Model compilation
model.compile(loss = 'mse', optimizer = 'rmsprop')

## Model training (less than one minute)
np.random.seed(1337)
model.fit(x_train, y_train, epochs = 500, batch_size = 32)

############################
# Prediction of new inputs #
############################
# The new input must be of dimension 3: k x l x 1.
# Where k is the size sample of new inputs (for example k=1),
#       l is the length of the sequence (for example 6 as before). Note that
# we apply sigmoid in parallel here, so the length sequence is not fixed.
#       1 is the input dimension at each time.
# So we have k inputs which are inputs of size l x 1.

## With k=1, l=6, we predict only 1 new input, a sequence of size T=6.
new_input = np.array([[[1],[0.8],[0.6],[0.2],[1],[0]]])
new_input.shape # (1, 6, 1)
print(model.predict(new_input))
#[[[ 0.72903323]
#  [ 0.62707138]
#  [ 0.51240331]
#  [ 0.29101145]
#  [ 0.72903323]
#  [ 0.20415471]]]
# The output has dimension (1, 6, 1)

## With k=2, l=6:
new_input = np.array([[[1],[0.8],[0.6],[0.2],[1],[0]],
                      [[0],[0],  [0],  [0],  [0],[0]]])
new_input.shape # (2, 6, 1)
print(model.predict(new_input))
# The output has dimension (2, 6, 1)
#[[[ 0.72903323]
#  [ 0.62707138]
#  [ 0.51240331]
#  [ 0.29101145]
#  [ 0.72903323]
#  [ 0.20415471]]
#
# [[ 0.20415471]
#  [ 0.20415471]
#  [ 0.20415471]
#  [ 0.20415471]
#  [ 0.20415471]
#  [ 0.20415471]]]

## With k=1, l=8:
new_input = np.array([[[1],[0.8],[0.6],[0.2],[1],[0],[1],[1]]])
new_input.shape # (1, 8, 1)
print(model.predict(new_input))
# The output has dimension (1, 8, 1)
#[[[ 0.72903323]
#  [ 0.62707138]
#  [ 0.51240331]
#  [ 0.29101145]
#  [ 0.72903323]
#  [ 0.20415471]
#  [ 0.72903323]
#  [ 0.72903323]]]

#################################
# Understanding the computation #
#################################
# At each time, we apply the same Dense model with logistic activation.
# The Dense model takes input of size 1 (because global input of size T x 1,
# so input at each time is of size 1).
# So we apply a linear model with 2 parameters a and b: y = a*x+b
model.get_weights() # a=2.35025001; b=-1.36052668

model.get_weights()[0].shape # 'a' has shape (1,1)
model.get_weights()[1].shape # 'b' has shape (1,)

# So: from the input [x_0 x_1 x_2 x_3 x_4 x_5] = [1, 0.8, 0.6, 0.2, 1, 0],
# we apply a*x+b and then sigmoid, i.e. predicted_y = sigmoid(a*x+b)
a = model.get_weights()[0]
b = model.get_weights()[1]
[sigmoid(a*x+b) for x in x_seed]
# [0.7290, 0.6271, 0.5124, 0.2910, 0.7290, 0.2042]
# We find the same result as before, as expected.

#######################################################
# Explanation of TimeDistributed with more dimensions #
#######################################################
sample_size = 256
x_seed = [1, 0.8, 0.6, 0.2, 1, 0]
y_seed = [0.7, 0.6, 0.5, 0.3, 0.8, 0.2]
dim_in = 2
dim_out = 3

x_train = reshape_training(x_seed, dim_in, sample_size)
y_train = reshape_training(y_seed, dim_out, sample_size)
# x_train and y_train of shape (256, 6, dim)
# Each element of x_train is:
# [[ 1. ,  1. ],
#  [ 0.8,  0.8],
#  [ 0.6,  0.6],
#  [ 0.2,  0.2],
#  [ 1. ,  1. ],
#  [ 0. ,  0. ]]

model=Sequential()
model.add(TimeDistributed(Dense(activation='sigmoid', 
                                units=dim_out), # target is dim_out-dimensional
                          input_shape=(None, dim_in))) # input is dim_in-dimensional
model.compile(loss = 'mse', optimizer = 'rmsprop')
np.random.seed(1337)
model.fit(x_train, y_train, epochs = 100, batch_size = 32)

new_input = np.array([[[1,1],[0.8,0.8],[0.6,0.6],[0.2,0.2],[1,1],[0,0]]])
new_input.shape # (1, 6, 2) as we need
print(model.predict(new_input))
# [[[ 0.67353392  0.59669352  0.57625091]
#   [ 0.61093992  0.56769931  0.55657816]
#   [ 0.54446143  0.53823376  0.53672636]
#   [ 0.40912622  0.47870329  0.4967348 ]
#   [ 0.67353392  0.59669352  0.57625091]
#   [ 0.34512752  0.44905871  0.47672269]]]
# output is (1, 6, 3) as expected.
# Note that first column and second column have been trained differently

a = model.get_weights()[0] # this is a (3,2) matrix
b = model.get_weights()[1] # this is a (3,1) vector
# At each time, we have this neural network without hidden layer
# input          output
#   O              O
#   O              O
#   o (bias term)  O
# So on the whole, 9 parameters (the same parameters at each time).

[[sigmoid(y)
  for y in np.dot(x,a)+b] # like doing X * beta
  for x in [[1,1],[0.8,0.8],[0.6,0.6],[0.2,0.2],[1,1],[0,0]]]
# We obtain the same results

##################################
##################################
## B/ Explanation of simple RNN ##
##################################
##################################
# Simple RNN is the Elman network
import tensorflow
import numpy as np
tensorflow.VERSION
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import SimpleRNN

#######################################
# Inputs and outputs for this section #
#######################################
sample_size = 256
dim_in = 3
dim_out = 2
x_seed = [1, 0, 0, 0, 0, 0]
y_seed = [0.7, 0.6, 0.5, 0.3, 0.8, 0.2]
x_train = reshape_training(x_seed, dim_in, sample_size)
y_train = reshape_training(y_seed, dim_out, sample_size)

#################################
# Model definition and training #
#################################
nb_units = 5

model=Sequential()
model.add(SimpleRNN(input_shape=(None, dim_in), 
                    return_sequences=True, 
                    units=nb_units))
model.add(TimeDistributed(Dense(activation='sigmoid', units=dim_out)))
model.compile(loss = 'mse', optimizer = 'rmsprop')
np.random.seed(1337)
model.fit(x_train, y_train, epochs = 100, batch_size = 32)

#############################
# Understanding the weights #
#############################

## Observations about model weights
model.get_weights()
model.get_weights()[0].shape
model.get_weights()[1].shape
model.get_weights()[2].shape
model.get_weights()[3].shape
model.get_weights()[4].shape
# There are 5 matrix of parameters. 
# With dim_in=3, dim_out=2, nb_units=5, we have:
# (3,5) # (dim_in, nb_units)
# (5,5) # (nb_units, nb_units)
# (5,) # (nb_units,)
# (5,2) # (nb_units, dim_out)
# (2,) # (dim_out,)

## Description of the Elman network
# From wikipedia:
# h_{-1} = [0, 0, 0, 0, 0]
# h_t = tanh(W_h * x_t + U_h * h_{t-1} + b_h)  (A)
# y_t = sigmoid(W_y * h_t + b_y)               (B)
#
## For the (input + hidden) --> hidden layer
W_h = model.get_weights()[0] # W_h a (3,5) matrix
U_h = model.get_weights()[1] # U_h a (5,5) matrix
b_h = model.get_weights()[2] # b_h a (5,1) vector
## For the hidden --> output layer
W_y = model.get_weights()[3] # W_y a (5,2) matrix
b_y = model.get_weights()[4] # b_y a (2,1) vector

##################################
# Understanding the computations #
##################################
# We compute network output from some new inputs, and we retrieve step by 
# step the result.

##
# Behavior with 3 inputs (from the trained network)
##
new_input = [[4,2,1], [1,1,1], [1,1,1]]
print(model.predict(np.array([new_input])))
# [4,2,1] at time 0,
# [1,1,1] at time 1,
# [1,1,1] at time 2.
#
# Output is:
# [[[ 0.55979645  0.57693499]
#   [ 0.74226248  0.74133486]
#   [ 0.63655949  0.6355176 ]]]

##
# Retrive the result (from the trained weights)
##
# We initialize the hidden vector with 0 (cf Elman's network definition)
h_minus1 = np.array([0, 0, 0, 0, 0]) # h_{-1} a (5,1) vector

## t = 0
# We apply formula (A) at t=0
x_0 = np.array(new_input[0]) # x_0 a (3,1) vector
h_0_before_tanh = np.dot(x_0, W_h) + np.dot(h_minus1, U_h) + b_h
h_0 = [math.tanh(x) for x in h_0_before_tanh]

# We apply formula (B) at t=0
y_0_before_sigmoid = np.dot(h_0, W_y) + b_y
y_0 = [sigmoid(x) for x in y_0_before_sigmoid]
print(y_0) # ok, we get same output: [0.5597964507974701, 0.5769349825548512]

## t = 1
# We apply formula (A) at t=1
x_1 = np.array(new_input[1]) # x_1 a (3,1) vector
h_1_before_tanh = np.dot(x_1, W_h) + np.dot(h_0, U_h) + b_h
h_1 = [math.tanh(x) for x in h_1_before_tanh]

# We apply formula (B) at t=1
y_1_before_sigmoid = np.dot(h_1, W_y) + b_y
y_1 = [sigmoid(x) for x in y_1_before_sigmoid]
print(y_1) # ok, we get same output: [0.7422624674301803, 0.7413348457274395]

## t = 2
# We apply formula (A) at t=2
x_2 = np.array(new_input[2]) # x_2 a (3,1) vector
h_2_before_tanh = np.dot(x_2, W_h) + np.dot(h_1, U_h) + b_h
h_2 = [math.tanh(x) for x in h_2_before_tanh]

# We apply formula (B) at t=2
y_2_before_sigmoid = np.dot(h_2, W_y) + b_y
y_2 = [sigmoid(x) for x in y_2_before_sigmoid]
print(y_2) # ok, we get same output: [0.6365595486435094, 0.6355176253524718]

#########################################################
#########################################################
## C/ Explanation of simple RNN with two hidden layers ##
#########################################################
#########################################################
# This is still the Elman's network, but with 2 stacks of hidden layers
import tensorflow
import numpy as np
tensorflow.VERSION
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import SimpleRNN

#######################################
# Inputs and outputs for this section #
#######################################
sample_size = 256
dim_in = 3
dim_out = 2
x_seed = [1, 0, 0, 0, 0, 0]
y_seed = [0.7, 0.6, 0.5, 0.3, 0.8, 0.2]
x_train = reshape_training(x_seed, dim_in, sample_size)
y_train = reshape_training(y_seed, dim_out, sample_size)

#################################
# Model definition and training #
#################################
model=Sequential()
model.add(SimpleRNN(input_shape=(None, dim_in), 
                    return_sequences=True, 
                    units=5))
model.add(SimpleRNN(input_shape=(None,4), 
                    return_sequences=True, 
                    units=7))
model.add(TimeDistributed(Dense(activation='sigmoid', units=dim_out)))
model.compile(loss = 'mse', optimizer = 'rmsprop')
np.random.seed(1337)
model.fit(x_train, y_train, epochs = 100, batch_size = 32)

#############################
# Understanding the weights #
#############################
model.get_weights()
model.get_weights()[0].shape # (3,5)
model.get_weights()[1].shape # (5,5)
model.get_weights()[2].shape # (5,1)
model.get_weights()[3].shape # (5,7)
model.get_weights()[4].shape # (7,7)
model.get_weights()[5].shape # (7,1)
model.get_weights()[6].shape # (7,2)
model.get_weights()[7].shape # (2,1)

## Elman network with 2 hidden layers
# h_0 = [0, 0, 0, 0, 0]
# h_t = tanh(W_h * x_t + U_h * h_{t-1} + b_h)  (A)
# h'_0 = [0, 0, 0, 0, 0, 0, 0] # the second hidden layer is called h' here
# h'_t = tanh(W_h' * h_t + U_h' * h'_{t-1} + b_h')  (A')
# y_t = sigmoid(W_y * h'_t + b_y)               (B)
#
# Note: in the code, "return_sequences=True" means that we output the whole
# vector (h_t) and (h'_t) and not only the last element, see
# https://stats.stackexchange.com/questions/129411

## For the (input + hidden) --> hidden layer
W_h = model.get_weights()[0] # W_h a (3,5) matrix
U_h = model.get_weights()[1] # U_h a (5,5) matrix
b_h = model.get_weights()[2] # b_h a (5,1) vector

## For the hidden --> hidden' layer
W_hp = model.get_weights()[3] # W_hp a (5,7) matrix
U_hp = model.get_weights()[4] # U_hp a (7,7) matrix
b_hp = model.get_weights()[5] # b_hp a (7,1) matrix

## For the hidden --> output layer
W_y = model.get_weights()[6] # W_y a (7,2) matrix
b_y = model.get_weights()[7] # b_y a (2,1) vector

# We initialize the hidden vectors with 0
h_0 = np.array([0, 0, 0, 0, 0]) # h_0 a (5,1) vector
hp_0 = np.array([0, 0, 0, 0, 0, 0, 0]) # hp_0 a (7,1) vector

##################################
# Understanding the computations #
##################################

##
# Behavior with 3 inputs (from the trained network)
##
new_input = [[4,2,1], [1,1,1], [1,1,1]]
print(model.predict(np.array([new_input])))
# [[[ 0.73741674  0.70267308]
#   [ 0.73841119  0.70142651]
#   [ 0.46848312  0.42730641]]]

##
# Retrive the result (from the trained weights)
##

## t = 1

# We apply formula (A) at t=1
x_1 = np.array(new_input[0]) # x_1 a (3,1) vector
h_1_before_tanh = np.dot(x_1, W_h) + np.dot(h_0, U_h) + b_h
h_1 = [math.tanh(x) for x in h_1_before_tanh]

# We apply formula (A') at t=1
hp_1_before_tanh = np.dot(h_1, W_hp) + np.dot(hp_0, U_hp) + b_hp
hp_1 = [math.tanh(x) for x in hp_1_before_tanh]

# We apply formula (B) at t=1
y_1_before_sigmoid = np.dot(hp_1, W_y) + b_y
y_1 = [sigmoid(x) for x in y_1_before_sigmoid]
print(y_1) # ok, we get same output: [0.7374167703539019, 0.7026730621971279]

## t = 2

# We apply formula (A) at t=2
x_2 = np.array(new_input[1]) # x_2 a (3,1) vector
h_2_before_tanh = np.dot(x_2, W_h) + np.dot(h_1, U_h) + b_h
h_2 = [math.tanh(x) for x in h_2_before_tanh]

# We apply formula (A') at t=2
hp_2_before_tanh = np.dot(h_2, W_hp) + np.dot(hp_1, U_hp) + b_hp
hp_2 = [math.tanh(x) for x in hp_2_before_tanh]

# We apply formula (B) at t=2
y_2_before_sigmoid = np.dot(hp_2, W_y) + b_y
y_2 = [sigmoid(x) for x in y_2_before_sigmoid]
print(y_2) # ok, we get same output: [0.738411158352165, 0.7014265342794316]

## t = 3

# We apply formula (A) at t=3
x_3 = np.array(new_input[2]) # x_3 a (3,1) vector
h_3_before_tanh = np.dot(x_3, W_h) + np.dot(h_2, U_h) + b_h
h_3 = [math.tanh(x) for x in h_3_before_tanh]

# We apply formula (A') at t=3
hp_3_before_tanh = np.dot(h_3, W_hp) + np.dot(hp_2, U_hp) + b_hp
hp_3 = [math.tanh(x) for x in hp_3_before_tanh]

# We apply formula (B) at t=3
y_3_before_sigmoid = np.dot(hp_3, W_y) + b_y
y_3 = [sigmoid(x) for x in y_3_before_sigmoid]
print(y_3) # ok, we get same output: [0.4684830640827, 0.4273064183103436]

############################
############################
## D/ Explanation of LSTM ##
############################
############################
import tensorflow
import numpy as np
tensorflow.VERSION
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, TimeDistributed

#######################################
# Inputs and outputs for this section #
#######################################
sample_size = 256
dim_in = 7
dim_out = 5
nb_units = 13

# T = 6:
x_seed = [1, 0, 0, 0, 0, 0]
y_seed = [0.7, 0.6, 0.5, 0.3, 0.8, 0.2]

x_train = reshape_training(x_seed, dim_in, sample_size)
y_train = reshape_training(y_seed, dim_out, sample_size)
# x_train has shape 256 x 6 x 7
# y_train has shape 256 x 6 x 5.
#
# Each element of x_train is called X. 
# X is a time series X_0, X_1, ..., X_{T-1} with T = 6 here.
# Each X_t has dimension dim_in=7
# So in this example: x_train[0]=X is:
# [[1, 1, 1, 1, 1, 1, 1],  <-- t = 0
#  [0, 0, 0, 0, 0, 0, 0],  <-- t = 1
#  [0, 0, 0, 0, 0, 0, 0],  <-- t = 2
#  [0, 0, 0, 0, 0, 0, 0],  <-- t = 3
#  [0, 0, 0, 0, 0, 0, 0],  <-- t = 4
#  [0, 0, 0, 0, 0, 0, 0]]  <-- t = 5
#   ^                 ^
#   |                 |
# first dim        last dim    
#
# After training model, we can feed the network by giving X_0 (a 7x1 vector),
# then X_1, then X_2, ... until X_{T-1}.
# We give each X_t in this order, and the LSTM model will keep in memory
# some information (through h_t and C_t elements).

#################################
# Model definition and training #
#################################
model=Sequential()
model.add(LSTM(input_shape=(None, dim_in), 
                    return_sequences=True, 
                    units=nb_units,
                    recurrent_activation='sigmoid',
                    activation='tanh'))
model.add(TimeDistributed(Dense(activation='sigmoid', units=dim_out)))

model.compile(loss = 'mse', optimizer = 'rmsprop')
np.random.seed(1337)
model.fit(x_train, y_train, epochs = 5, batch_size = 32)

#############################
# Understanding the weights #
#############################
model.get_weights()
model.get_weights()[0].shape
model.get_weights()[1].shape
model.get_weights()[2].shape
model.get_weights()[3].shape
model.get_weights()[4].shape
# LSTM: There are 5 parameters. With dim_in=7, dim_out=5, nb_units=13, we have:
# (7,13*4) # (dim_in, nb_units*4)
# (13,13*4) # (nb_units, nb_units*4)
# (13*4,) # (nb_units*4,)
# (13,5) # (nb_units, dim_out)
# (5,) # (dim_out,)

## We can separate the LSTM layer from the output TimeDistributed layer:

##
# LSTM hidden layer
##
for n in model.layers[0].trainable_weights:
    print(n)
W_x = model.get_weights()[0]
W_h = model.get_weights()[1]
b_h = model.get_weights()[2]

##
# Hidden to output layer
##
for n in model.layers[1].trainable_weights:
    print(n)
W_y = model.get_weights()[3]
b_y = model.get_weights()[4]

## We can play with weights:
#W_x = np.zeros([7, 52])
#W_h = np.zeros([13, 52])
#b_h = np.zeros([52,])
#
#W_y = np.zeros([13, 5])
#b_y = np.zeros([5, ])
model.set_weights([W_x, W_h, b_h, W_y, b_y])

##################################
# Understanding the computations #
##################################

##
# LSTM layer
##

# In this example, we consider:
# - a size of input of 7,
# - number of units in hidden layer of 13.
# The number 4 arises because there are 4 equations in LSTM
# using [h_{t-1}, x_t].
def through_LSTM_layer(x_t = [0,0,0,0,0,0,0], # 7
                       h_t_minus_1 = [0,0,0,0,0,0,0,0,0,0,0,0,0], # 13
                       C_t_minus_1 = [0,0,0,0,0,0,0,0,0,0,0,0,0], # 13
                       W_x = np.zeros([7, 4*13]),
                       W_h = np.zeros([13, 4*13]),
                       b_h = np.zeros([4*13,])):
    # From http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    # we have the LSTM formula:
    # 
    # * First, there are 4 vector of size (13,) computed as:
    #    i_t = sigma(W_i . [h_{t-1}, x_t] + b_i)
    #    f_t = sigma(W_f . [h_{t-1}, x_t] + b_f)
    #    \hat{C}_t = tanh(W_C . [h_{t-1}, x_t] + b_C)
    #    o_t = sigma(W_o . [h_{t-1}, x_t] + b_o)
    # Here, [a,b] is the concatenation. h_{t-1} has size (13,) and x_t
    # has size (7,), so [h_{t-1}, x_t] has size (13+7,)=(20,).
    # So: W_i, W_f, W_C, W_o are each (20, 13) 
    # and b_i, b_f, b_C, b_o are each (13,)
    #
    # In the Keras implementation, we take h_{t-1} and x_t separately (not
    # the concatenation matrix, so we get:
    #    W_ix, W_fx, W_Cx, W_ox are each (7, 13),
    #    W_ih, W_fh, W_Ch, W_oh are each (13, 13),
    #    b_i, b_f, b_C, b_o are each (13,)
    #
    # Also, in the Keras implementation, those 4 matrices are together,
    # so W_x := [W_ix, W_fx, W_Cx, W_ox] of shape (7, 52) = (7, 13*4)
    #    W_h := [W_ih, W_fh, W_Ch, W_oh] of shape (13, 52) = (13, 13*4)
    #    b_h := [b_i, b_f, b_C, b_o] of shape (52,)
    # 
    # * Then we update the C_t layer and the h_t layer with:
    #    C_t = f_t * C_{t-1} + i_t * \hat{C}_t
    #    h_t = o_t * tanh(C_t)
    #
    # On the whole, x_t is (7,); h_{t-1} is (13,); C_{t-1} is (13,).
    # Through first computations, we obtain: i_t, f_t, \hat{C}_t, o_t, each
    # being (13,)
    # Through following computations, we obtain: C_t and h_t, each being (13,).
    
    # raw_t has size 52 x 1
    raw_t = np.dot(x_t, W_x) + np.dot(h_t_minus_1, W_h) + b_h
    nb_units = len(raw_t)//4
    raw_t_0 = raw_t[0:nb_units]
    raw_t_1 = raw_t[nb_units:(2*nb_units)]
    raw_t_2 = raw_t[(2*nb_units):(3*nb_units)]
    raw_t_3 = raw_t[(3*nb_units):(4*nb_units)]

    # https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py
    # Line 1750 of source code to know the right order!
    i_t = [sigmoid(x) for x in raw_t_0]
    f_t = [sigmoid(x) for x in raw_t_1]
    hatC_t = [math.tanh(x) for x in raw_t_2]
    o_t = [sigmoid(x) for x in raw_t_3]
    
    C_t = np.multiply(f_t, C_t_minus_1) + np.multiply(i_t, hatC_t) # (13,)
    h_t = np.multiply(o_t, [math.tanh(x) for x in C_t]) # h of size (13,)
    
    return(h_t, C_t)
   
##
# Output layer
##
# This is the output layer has before, to convert (13,) to (5,) via
# matrix (13,5) and bias (5,).
def through_output_layer(h_t = [0,0,0,0,0,0,0,0,0,0,0,0,0], # 13
                         W_y = np.zeros([13, 5]),
                         b_y = np.zeros([5, ])):
    ## For the hidden --> output layer
    y_t_before_sigmoid = np.dot(h_t, W_y) + b_y
    y_t = [sigmoid(x) for x in y_t_before_sigmoid]
    return(y_t)

##
# Checking computations
##

## Results with weights:
x_1 = [1,1,1,1,2,3,4] # size 7
x_2 = [1,2,3,4,3,2,1]
new_input = [x_1, x_2]

h_0 = [0,0,0,0,0,0,0,0,0,0,0,0,0] # size 13
C_0 = [0,0,0,0,0,0,0,0,0,0,0,0,0] # size 13

C_t_minus_1 = C_0
h_t_minus_1 = h_0
x_t = x_1

(h_1, C_1) = through_LSTM_layer(x_1, h_0, C_0, W_x, W_h, b_h)
y_1 = through_output_layer(h_1, W_y, b_y)
print(y_1) # output of size 5

(h_2, C_2) = through_LSTM_layer(x_2, h_1, C_1, W_x, W_h, b_h)
y_2 = through_output_layer(h_2, W_y, b_y)
print(y_2)

## Comparison with model results:
print(model.predict(np.array([new_input])))

###########################
###########################
## E/ Explanation of GRU ##
###########################
###########################
import tensorflow
import numpy as np
tensorflow.VERSION
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU, TimeDistributed

#######################################
# Inputs and outputs for this section #
#######################################
# Copy paste from "D/ Explanation of LSTM"
sample_size = 256
dim_in = 7
dim_out = 5
nb_units = 13

# T = 6:
x_seed = [1, 0, 0, 0, 0, 0]
y_seed = [0.7, 0.6, 0.5, 0.3, 0.8, 0.2]

x_train = reshape_training(x_seed, dim_in, sample_size)
y_train = reshape_training(y_seed, dim_out, sample_size)
# x_train has shape 256 x 6 x 7
# y_train has shape 256 x 6 x 5.
#
# Each element of x_train is called X. 
# X is a time series X_0, X_1, ..., X_{T-1} with T = 6 here.
# Each X_t has dimension dim_in=7
# So in this example: x_train[0]=X is:
# [[1, 1, 1, 1, 1, 1, 1],  <-- t = 0
#  [0, 0, 0, 0, 0, 0, 0],  <-- t = 1
#  [0, 0, 0, 0, 0, 0, 0],  <-- t = 2
#  [0, 0, 0, 0, 0, 0, 0],  <-- t = 3
#  [0, 0, 0, 0, 0, 0, 0],  <-- t = 4
#  [0, 0, 0, 0, 0, 0, 0]]  <-- t = 5
#   ^                 ^
#   |                 |
# first dim        last dim    
#
# After training model, we can feed the network by giving X_0 (a 7x1 vector),
# then X_1, then X_2, ... until X_{T-1}.
# We give each X_t in this order, and the LSTM model will keep in memory
# some information (through h_t and C_t elements).

from keras.initializers import Constant
#################################
# Model definition and training #
#################################
# Copy paste from "D/ Explanation of LSTM", the only difference is using GRU
model=Sequential()
model.add(GRU(input_shape=(None, dim_in), 
                    return_sequences=True, 
                    units=nb_units,
                    recurrent_activation='sigmoid',
                    activation='tanh',
                    recurrent_initializer=Constant(value=0)))
model.add(TimeDistributed(Dense(activation='sigmoid', units=dim_out)))

model.compile(loss = 'mse', optimizer = 'rmsprop')
np.random.seed(1337)
model.fit(x_train, y_train, epochs = 5, batch_size = 32)

#############################
# Understanding the weights #
#############################
# Copy paste from "D/ Explanation of LSTM", the only difference is we only
# need nb_units*3 instead of nb_units*4 to perform computations.

model.get_weights()
model.get_weights()[0].shape
model.get_weights()[1].shape
model.get_weights()[2].shape
model.get_weights()[3].shape
model.get_weights()[4].shape
# GRU: There are 5 parameters. With dim_in=7, dim_out=5, nb_units=13, we have:
# (7,13*3) # (dim_in, nb_units*3)
# (13,13*3) # (nb_units, nb_units*3)
# (13*3,) # (nb_units*3,)
# (13,5) # (nb_units, dim_out)
# (5,) # (dim_out,)

## We can separate the GRU layer from the output TimeDistributed layer:

##
# GRU hidden layer
##
for n in model.layers[0].trainable_weights:
    print(n)
W_x = model.get_weights()[0]
W_h = model.get_weights()[1]
b_h = model.get_weights()[2]

##
# Hidden to output layer
##
for n in model.layers[1].trainable_weights:
    print(n)
W_y = model.get_weights()[3]
b_y = model.get_weights()[4]

## We can play with weights:
#W_x = np.zeros([7, 39])
#W_h = np.zeros([13, 39])
#b_h = np.zeros([39,])

#
#W_y = np.zeros([13, 5])
#b_y = np.zeros([5, ])

model.set_weights([W_x, W_h, b_h, W_y, b_y])

##################################
# Understanding the computations #
##################################

##
# GRU layer
##
# Many changes compared to "D/ Explanation of LSTM"
#
# In this example, we consider:
# - a size of input of 7,
# - number of units in hidden layer of 13.
# The number 3 arises because there are 3 equations in GRU using [h_{t-1}, x_t]
def through_GRU_layer(x_t = [0,0,0,0,0,0,0], # 7
                      h_t_minus_1 = [0,0,0,0,0,0,0,0,0,0,0,0,0], # 13
                      W_x = np.zeros([7, 3*13]),
                      W_h = np.zeros([13, 3*13]),
                      b_h = np.zeros([3*13,])):
    # From http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    # we have the GRU formula:
    # 
    # * First, there are 3 vector of size (13,) computed as:
    #    z_t = sigma(W_z . [h_{t-1}, x_t]) + b_z
    #    r_t = sigma(W_r . [h_{t-1}, x_t]) + b_r
    #    \hat{h}_t = tanh(W_o . [r_t * h_{t-1}, x_t]) + b_o
    # Here, [a,b] is the concatenation. h_{t-1} has size (13,) and x_t
    # has size (7,), so [h_{t-1}, x_t] has size (13+7,)=(20,).
    # So: W_z, W_r, W_o are each (20, 13) 
    # and b_z, b_r, b_o are each (13,)
    #
    # In the Keras implementation, we take h_{t-1} and x_t separately (not
    # the concatenation matrix, so we get:
    #    W_zx, W_rx, W_ox are each (7, 13),
    #    W_zh, W_rh, W_oh are each (13, 13),
    #    b_z, b_r, b_o are each (13,)
    #
    # Also, in the Keras implementation, those 3 matrices are together,
    # so W_x := [W_zx, W_rx, W_ox] of shape (7, 39) = (7, 13*3)
    #    W_h := [W_zh, W_rh, W_oh] of shape (13, 39) = (13, 13*3)
    #    b_h := [b_z, b_r, b_o] of shape (39,)
    # 
    # * Then we update the h_t layer with:
    #    h_t = (1 - z_t) * h_{t-1} + z_t * \hat{h}_t
    #
    # On the whole, x_t is (7,) and h_{t-1} is (13,).
    # Through first computations, we obtain: z_t, r_t, \hat{h}_t, each
    # being (13,)
    # Through following computations, we obtain: h_t, of size (13,).
    
    # raw_t has size 39 x 1
    raw_t = np.dot(x_t, W_x) + np.dot(h_t_minus_1, W_h) + b_h
    nb_units = len(raw_t)//3
    raw_t_0 = raw_t[0:nb_units]
    raw_t_1 = raw_t[nb_units:(2*nb_units)]
    #raw_t_2 = raw_t[(2*nb_units):(3*nb_units)] # this one is not useful here.

    # https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py
    # Line 1257 of source code to know the right order!
    z_t = np.array([sigmoid(x) for x in raw_t_0])
    r_t = [sigmoid(x) for x in raw_t_1]
    
    # hath_t use r_t in its formula, so we cannot compute it from raw_t_2.
    # This is specific to GRU.
    # We proceed as follows to get 
    # \hat{h}_t = tanh(W_o . [r_t * h_{t-1}, x_t]) + b_o
    r_t_h_t_minus_1 = np.multiply(r_t, h_t_minus_1)
    raw_t_new = np.dot(x_t, W_x) + np.dot(r_t_h_t_minus_1, W_h) + b_h
    raw_t_2 = raw_t_new[(2*nb_units):(3*nb_units)]
    hath_t = [math.tanh(x) for x in raw_t_2]
    
    # https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py
    # Line 1343 of source code to know the correct weights!
    h_t = np.multiply(z_t, h_t_minus_1) + np.multiply(1 - z_t, hath_t) # (13,)

    return(h_t)
   
##
# Output layer
##
# Copy paste from "D/ Explanation of LSTM"
# This is the output layer has before, to convert (13,) to (5,) via
# matrix (13,5) and bias (5,).
def through_output_layer(h_t = [0,0,0,0,0,0,0,0,0,0,0,0,0], # 13
                         W_y = np.zeros([13, 5]),
                         b_y = np.zeros([5, ])):
    ## For the hidden --> output layer
    y_t_before_sigmoid = np.dot(h_t, W_y) + b_y
    y_t = [sigmoid(x) for x in y_t_before_sigmoid]
    return(y_t)

##
# Checking computations
##

## Results with weights:
x_1 = [1,1,1,1,2,3,4] # size 7
x_2 = [1,2,3,4,3,2,1]
new_input = [x_1, x_2]

h_0 = [0,0,0,0,0,0,0,0,0,0,0,0,0] # size 13

h_t_minus_1 = h_0
x_t = x_1

h_1 = through_GRU_layer(x_1, h_0, W_x, W_h, b_h)
y_1 = through_output_layer(h_1, W_y, b_y)
print(y_1) # output of size 5

h_2 = through_GRU_layer(x_2, h_1, W_x, W_h, b_h)
y_2 = through_output_layer(h_2, W_y, b_y)
print(y_2)

## Comparison with model results:
print(model.predict(np.array([new_input])))

## All is working as expected!