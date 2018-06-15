####################################
# Toy example in RNN, LSTM and GRU #
# 2017/12/25 - 2017/12/28          #
####################################
# This code shows a toy problem containing long dependence.
# This problem is modeled with 3 recurrent neural networks: RNN, LSTM and GRU.
# 
# RNN can learn long dependence, but the problem (from literature) is gradient
# vanishing and exploding. We don't have this problem in this example.
#
# For all models here, there is a gradient clipping problem. This can be
# managed with regularization (not considered in this code).
#
# (using tensorflow 1.1.0 and keras 2.0.3)
#
# Organization of this code:
# - A/ Simple RNN
# - B/ LSTM
# - C/ GRU

import os
import tensorflow
import numpy as np
tensorflow.VERSION
from keras.models import Sequential
from keras.layers import Dense
os.chdir(os.getenv('HOME') + '/Documents/GitHub/deep-learning/rnn/')

####################
# Helper functions #
####################
## Checking and creating directory
def create(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

## Sigmoid function
import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

#####################
# Rules of the game #
#####################
# There are 4 elements which forms E
# 0 : neutral element
# 1 : blue element
# 2 : red element
# 3 : ask last red/blue element

# Inputs and outputs are sequence of E:
# output has the following rule:
# if input is 0, then output 0,
# if input is 1, then output 1,
# if input is 2, then output 2,
# if input is 3, then output the last blue or red element, if none output 0.
#
# Examples:
# input  0 0 0 0 0 1 0 0 0 2 0 0 0 3 3 2 0 0 0 3 3 0 1 3
# output 0 0 0 0 0 1 0 0 0 2 0 0 0 2 2 2 0 0 0 2 2 0 1 1

# input  3 0 0 3 1 3 0 1 3 2 3 3 3
# output 0 0 0 0 1 1 0 1 1 2 2 2 2

###############
# Sample data #
###############
# We take long dependence
#
# Final sample data consist of 'inputs' and 'outputs'
# 'inputs' has shape (1000, 200, 4)
# 'outputs' has shape (1000, 200, 4).
# Shape is (N, T, m)~~
#
# For example: inputs[0][7] is [0,0,0,1] which is hot-encoding for 3 = 'ask'
# So for sample 0, at date t=7-1=6, the input is 'ask'.
#             outputs[0][7] is [0,1,0,0] which is hot-encoding for 1 = 'blue'
# So for sample 0, at date t=7-1=6, the output is 'blue'.

N = 1000 # size of samples
T = 200 # length of each sample
p = [0.8, 0.07, 0.03, 0.1]  # probability for each input

#names = ['neutral', 'blue', 'red', 'ask'] # name related to each probability
names = [0, 1, 2, 3] # name related to each probability
m = len(names)

##
# Inputs
##
np.random.seed(1337)
inputs = np.array(np.random.choice(names, size=T*N, p=p)).reshape(N, T)
inputs.shape # (1000, 20)

##
# Outputs
##
def replaced_color(idx, out_current):
    x = out_current[0:idx]
    
    for to_remove in [0, 3]: # remve 'neutral' and 'ask' 
        # https://stackoverflow.com/questions/1157106
        x = list(filter(lambda a: a != to_remove, x))
    
    out_current_replace = 0 # 0 for 'neutral'
    if len(x) > 0:
        out_current_replace = x[len(x)-1]
  
    return(out_current_replace)

##
outputs = inputs.copy()
for index, out_current in enumerate(outputs):
    #print(index)
    bool_current = out_current == 3 # 3 for 'ask'
    where_current = np.where(bool_current)[0]
    for idx in where_current:
        out_current[idx] = replaced_color(idx, out_current)
    outputs[index] = out_current

## Transform to one-hot encoding
from keras.utils import to_categorical
inputs = to_categorical(inputs, num_classes = m).reshape(N, T, m)
outputs = to_categorical(outputs, num_classes = m).reshape(N, T, m)

#########################
#########################
## A/ Modeling via RNN ##
#########################
#########################

##########################
# Create and train model #
##########################
from keras.layers import SimpleRNN, TimeDistributed

model=Sequential()
dim_in = m
dim_out = m
nb_units = 3

model.add(SimpleRNN(input_shape=(None, dim_in),
                    return_sequences=True, 
                    units=nb_units))
model.add(TimeDistributed(Dense(activation='sigmoid', units=dim_out)))
model.compile(loss = 'mse', optimizer = 'rmsprop')
# Each epoch cost about 2 seconds
np.random.seed(1337)
model.fit(inputs, outputs, epochs = 500, batch_size = 32)

create('models')
to_save = False
if(to_save):
    model.save('models/2_A_coloring_simple_rnn.h5')
else:
    # model has been saved with nb_units = 3
    from keras.models import load_model
    model = load_model('models/2_A_coloring_simple_rnn.h5')

##
# Comparison of mse error as a function of epoch and nb_units:
##
#
# epoch  100   200   300   400   500
# unit
#    1  2e-2  2e-2  2e-2  2e-2  2e-2  # stuck
#    2  4e-3  2e-3  1e-3  1e-3  1e-3  # stuck
#    3  3e-3  1e-3  2e-4  2e-7  4e-8  
#    5  1e-3  3e-6  1e-8  4e-9  2e-9

##########################
# Checking model results #
##########################

##
# Keeping in memory with short dependencies
##
new_input = [[1,0,0,0], [1,0,0,0], [1,0,0,0]]
print(np.around(model.predict(np.array([new_input]))))
# Neutral outputs neutral
#[[[ 1.  0.  0.  0.]
#  [ 1.  0.  0.  0.]
#  [ 1.  0.  0.  0.]]]

new_input = [[0,1,0,0], [0,1,0,0], [0,1,0,0]]
print(np.around(model.predict(np.array([new_input]))))
# Blue outputs blue
#[[[ 0.  1.  0.  0.]
#  [ 0.  1.  0.  0.]
#  [ 0.  1.  0.  0.]]]

new_input = [[0,0,1,0], [0,0,1,0], [0,0,1,0]]
print(np.around(model.predict(np.array([new_input]))))
# Red outputs red
#[[[ 0.  0.  1.  0.]
#  [ 0.  0.  1.  0.]
#  [ 0.  0.  1.  0.]]]

new_input = [[0,0,0,1], [0,0,0,1], [0,0,0,1]]
print(np.around(model.predict(np.array([new_input]))))
# Ask outputs neutral because there is no previous blue/red
#[[[ 1.  0.  0.  0.]
#  [ 1.  0.  0.  0.]
#  [ 1.  0.  0.  0.]]]

new_input = [[0,1,0,0], [1,0,0,0], [0,0,0,1]]
print(np.around(model.predict(np.array([new_input]))))
# Ask outputs blue because the last non neutral is blue
#[[[ 0.  1.  0.  0.]
#  [ 1.  0.  0.  0.]
#  [ 0.  1.  0.  0.]]]

##
# Keeping in memory with long dependencies
##
new_input = [[0,1,0,0], 
             [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0],
             [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0],
             [0,0,0,1]]
print(np.around(model.predict(np.array([new_input]))))
# Ask outputs blue because the last non neutral is blue
#[[[ 0.  1.  0.  0.]
#  [ 1.  0.  0.  0.]
#  [ 1.  0.  0.  0.]
#  ..., 
#  [ 1.  0.  0.  0.]
#  [ 1.  0.  0.  0.]
#  [ 0.  1.  0.  0.]]]

new_input = [[0,0,1,0], 
             [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0],
             [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0],
             [0,0,0,1]]
print(np.around(model.predict(np.array([new_input]))))
# Ask outputs red because the last non neutral is red
#[[[ 0.  0.  1.  0.]
#  [ 1.  0.  0.  0.]
#  [ 1.  0.  0.  0.]
#  ..., 
#  [ 1.  0.  0.  0.]
#  [ 1.  0.  0.  0.]
#  [ 0.  0.  1.  0.]]]

new_input = [[1,0,0,0], 
             [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0],
             [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0],
             [0,0,0,1]]
print(np.around(model.predict(np.array([new_input]))))
# Ask outputs neutral because their is no previous non neutral
#[[[ 1.  0.  0.  0.]
#  [ 1.  0.  0.  0.]
#  [ 1.  0.  0.  0.]
#  ..., 
#  [ 1.  0.  0.  0.]
#  [ 1.  0.  0.  0.]
#  [ 1.  0.  0.  0.]]]

#######################
# Analysis of weights #
#######################
model.get_weights()
model.get_weights()[0].shape
model.get_weights()[1].shape
model.get_weights()[2].shape
model.get_weights()[3].shape
model.get_weights()[4].shape
# There are 5 parameters (with dim_in=4, dim_out=4, units=3)
# (4,3) # (dim_in, units)
# (3,3) # (units, units)
# (3,) # (units,)
# (3,4) # (units, dim_out)
# (4,) # (dim_out,)

## Description of the Elman network
# From wikipedia:
# h_0 = [0, 0, 0]
# h_t = tanh(W_h * x_t + U_h * h_{t-1} + b_h)  (A)
# y_t = sigmoid(W_y * h_t + b_y)               (B)
#
## For the (input + hidden) --> hidden layer
W_h = model.get_weights()[0] # W_h a (4,3) matrix
U_h = model.get_weights()[1] # U_h a (3,3) matrix
b_h = model.get_weights()[2] # b_h a (3,1) vector

## For the hidden --> output layer
W_y = model.get_weights()[3] # W_y a (3,4) matrix
b_y = model.get_weights()[4] # b_y a (4,1) vector

#######################
# Changing parameters #
#######################
# Manually, the parameters has been rounded to try to interpret them.
b_y = 10 * np.array([0, -1, -1, -1])
W_y = 10 * np.array([[-1, 1, 0, 1],[1, -1, -1, 0],[1, 0, -1, -1]])

W_h[0,0] = -1.6 # critical parameter
W_h[0,1] = 3
W_h[0,2] = 0.9 # critical parameter, with 1 not ok
W_h[1,0] = 7
W_h[1,1] = -5
W_h[1,2] = 5
W_h[2,0] = -5
W_h[2,1] = -5
W_h[2,2] = -5
W_h[3,0] = -1
W_h[3,1] = -5
W_h[3,2] = 0.2

U_h[0,0] = 2
U_h[0,1] = 2
U_h[0,2] = 1
U_h[1,0] = 0
U_h[1,1] = 0
U_h[1,2] = 0
U_h[2,0] = 2
U_h[2,1] = -2
U_h[2,2] = 2

b_h[0] = -0.9 # critical, with -1 not ok suddenly
b_h[1] = 0 # any small value is ok
b_h[2] = 0.6 # critical, with 0.7 not ok suddenly

model.set_weights([W_h, U_h, b_h, W_y, b_y])
model.fit(inputs, outputs, epochs = 1, batch_size = 32)
# Before: 7e-9 error ; After: 1e-9.
# The essential point is that we should have the same behavior with this
# "rounded" model.

###############
# From h to y #
###############
## We first observe behavior of the output given h.
#
# y_t = sigmoid(W_y' * h_t + b_y)               (B)
# 4x1           3x4' * 3x1 + 4x1
def y_from_h(h = [0,0,0]):
  return np.around([sigmoid(x) for x in np.dot(W_y.transpose(), h) + b_y], 2)

## The different choices for y from some strong shapes of h
#
# We choose the corner of h, which is the cube [-1,1]^3 because h is computed
# after applying the tanh function. We add the 0 element to make our grid:
#
# Each line can be read like this: 
# If the h is [a,b,c], then y will be [d,e,f,g].
# y is a correct output only if it looks like a hot vector here (because the
# h --> y weights are so strong, in general we can take most probable element).
#
y_from_h([-1,-1,-1]) # red [0,0,1,0]
y_from_h([-1,-1,0])  # not a correct output
y_from_h([-1,-1,1])  # neutral [1,0,0,0]
y_from_h([-1,0,-1])  # not a correct output
y_from_h([-1,0,0])   # neutral [1,0,0,0]
y_from_h([-1,0,1])   # neutral [1,0,0,0]
y_from_h([-1,1,-1])  # neutral [1,0,0,0]
y_from_h([-1,1,0])   # neutral [1,0,0,0]
y_from_h([-1,1,1])   # neutral [1,0,0,0]
y_from_h([0,-1,-1])  # not a correct output
y_from_h([0,-1,0])   # not a correct output
y_from_h([0,-1,1])   # not a correct output
y_from_h([0,0,-1])   # not a correct output
y_from_h([0,0,0])    # not a correct output
y_from_h([0,0,1])    # neutral [1,0,0,0]
y_from_h([0,1,-1])   # not a correct output
y_from_h([0,1,0])    # neutral [1,0,0,0]
y_from_h([0,1,1])    # neutral [1,0,0,0]
y_from_h([1,-1,-1])  # not a correct output
y_from_h([1,-1,0])   # not a correct output
y_from_h([1,-1,1])   # blue [0,1,0,0]
y_from_h([1,0,-1])   # not a correct output
y_from_h([1,0,0])    # not a correct output
y_from_h([1,0,1])    # not a correct output
y_from_h([1,1,-1])   # ask [0,0,0,1]
y_from_h([1,1,0])    # not a correct output
y_from_h([1,1,1])    # neutral [1,0,0,0]

## So in summary:
y_from_h([ 1,-1, 1]) # blue [0,1,0,0]    (Blue)
y_from_h([-1,-1,-1]) # red [0,0,1,0]     (Red)
y_from_h([ 1, 1,-1]) # ask [0,0,0,1] (this should not occur)
y_from_h([-1,-1, 1]) # neutral [1,0,0,0]
y_from_h([-1, 0, 0]) # neutral [1,0,0,0]
y_from_h([-1, 0, 1]) # neutral [1,0,0,0]
y_from_h([-1, 1,-1]) # neutral [1,0,0,0]
y_from_h([-1, 1, 0]) # neutral [1,0,0,0]
y_from_h([-1, 1, 1]) # neutral [1,0,0,0] (*)
y_from_h([ 0, 0, 1]) # neutral [1,0,0,0]
y_from_h([ 0, 1, 0]) # neutral [1,0,0,0]
y_from_h([ 0, 1, 1]) # neutral [1,0,0,0]
y_from_h([ 1, 1, 1]) # neutral [1,0,0,0] (**)

##############################
# From h to h (given inputs) #
##############################
# h_t = tanh(W_h' * x_t + U_h' * h_{t-1} + b_h)  (A)
# 3x1        4x3' * 4x1 + 3x3' * 3x1 + 3x1
def h_from_h(h = [0,0,0], x = [1, 0, 0, 0]):
  return np.around([math.tanh(x) for x in np.dot(W_h.transpose(), x) + 
                                          np.dot(U_h.transpose(), h) + 
                                          b_h], 2)

## From h = [0, 0, 0], which is the initial value of h
h_from_h([0, 0, 0], [1, 0, 0, 0]) # [-1,  1,  1] # to neutral, looks like (*)
h_from_h([0, 0, 0], [0, 1, 0, 0]) # [ 1, -1,  1] (looks like (Blue))
h_from_h([0, 0, 0], [0, 0, 1, 0]) # [-1, -1, -1] (looks like (Red))
h_from_h([0, 0, 0], [0, 0, 0, 1]) # [-1, -1, 0.66] # to neutral

## From h = [ 1,-1, 1], corresponding to the strongest memory of 'blue'
h_from_h([1, -1, 1], [1, 0, 0, 0]) # [ 1,  1,  1] # to neutral, looks like (**)
h_from_h([1, -1, 1], [0, 1, 0, 0]) # [ 1, -1,  1] (looks like (Blue))
h_from_h([1, -1, 1], [0, 0, 1, 0]) # [ 1, -1,  1] (looks like (Red))
h_from_h([1, -1, 1], [0, 0, 0, 1]) # [ 1, -1,  1] (looks like (Blue))
# Conclusion: even with this strong memory of 'blue', the network is able
# to switch memory to 'red'.

# Note: h cannot be taken outside the box [-1,1]^3. If we take outside it,
# the switch of memory can fail:
h_from_h([ 5, -5, 5], [0, 0, 1, 0]) # [ 1, -1,  1] 
# (looks like (Blue), but we would like (Red))

## From h = [-1,-1,-1], corresponding to the strong memory of 'red'
h_from_h([-1, -1, -1], [1, 0, 0, 0]) # [-1,  1, -1] # to neutral
h_from_h([-1, -1, -1], [0, 1, 0, 0]) # [ 1, -1,  1] (looks like (Blue))
h_from_h([-1, -1, -1], [0, 0, 1, 0]) # [-1, -1, -1] (looks like (Red))
h_from_h([-1, -1, -1], [0, 0, 0, 1]) # [-1, -1, -1] (looks like (Red))

## From h = [-1,  1, -1] (a 'h to neutral' obtained from red)
h_from_h([-1,  1, -1], [1, 0, 0, 0]) # [-1,  1, -1] # to neutral
h_from_h([-1,  1, -1], [0, 1, 0, 0]) # [ 1, -1,  1] (looks like (Blue))
h_from_h([-1,  1, -1], [0, 0, 1, 0]) # [-1  -1  -1] (looks like (Red))
h_from_h([-1,  1, -1], [0, 0, 0, 1]) # [-1, -1, -1] (looks like (Red))

## Try to let it forget... (but not succeed, probably not possible).
h_from_h([0, 0, 0], [0, 1, 0, 0]) # to [1, -1, 1] (= blue)
h_from_h([1, -1, 1], [0, 0, 1, 0]) # to [-0.96, -1, -0.89] (= red)
h_from_h([-0.96, -1, -0.89], [1, 0, 0, 0]) # to [-1, 0.99, -0.85] (= neutral)
h_from_h([-1, 0.99, -0.85], [1, 0, 0, 0]) # to [-1, 0.99, -0.83] (= neutral)
h_from_h([-1, 0.99, -0.83], [1, 0, 0, 0]) # to [-1, 0.99, -0.82] (= neutral)
h_from_h([-1, 0.99, -0.82], [1, 0, 0, 0]) # to [-1, 0.99, -0.81] (= neutral)
h_from_h([-1, 0.99, -0.81], [1, 0, 0, 0]) # to [-1, 0.99, -0.81] (= neutral)
h_from_h([-1, 0.99, -0.81], [1, 0, 0, 0]) # to [-1, 0.99, -0.81] (= neutral)
h_from_h([-1, 0.99, -0.81], [0, 0, 0, 1]) # to [-1.  , -1.  , -0.95] (= red)

##############
# Conclusion #
##############
# This simple RNN network is able to solve this toy problem, even with long
# dependencies.

##########################
##########################
## B/ Modeling via LSTM ##
##########################
##########################

##########################
# Create and train model #
##########################
from keras.layers import LSTM, TimeDistributed

model=Sequential()
dim_in = m
dim_out = m
nb_units = 5

model.add(LSTM(input_shape=(None, dim_in),
                    return_sequences=True, 
                    units=nb_units))
model.add(TimeDistributed(Dense(activation='sigmoid', units=dim_out)))
model.compile(loss = 'mse', optimizer = 'rmsprop')
# Each epoch cost 8 seconds with 1 unit
np.random.seed(1337)
model.fit(inputs, outputs, epochs = 500, batch_size = 32)

create('models')
to_save = False
if(to_save):
    model.save('models/2_B_coloring_lstm.h5')
else:
    # model has been saved with nb_units = 3
    from keras.models import load_model
    model = load_model('models/2_B_coloring_lstm.h5')

##
# Comparison of mse error as a function of epoch and nb_units:
##
# Here for LSTM:
# epoch  100   200   300   400   500
# unit
#    1  2e-2  2e-2  2e-2  2e-2  2e-2 # stuck, with slow decrease all along
#    2  2e-2  8e-3  2e-3  1e-3  1e-3 # stuck 
#    3  2e-3  1e-5  3e-7  1e-8  5e-9
#    5  1e-3  1e-5  6e-9  2e-9  1e-9

##########################
# Checking model results #
##########################
# Keeping in memory short and long dependencies as before.
# Code in Part A can be used.

###########
# Weights #
###########
# See 1_math_structure_of_rnn.py for more information
W_x = model.get_weights()[0] # (4, 12)
W_h = model.get_weights()[1] # (3, 12)
b_h = model.get_weights()[2] # (12,)
W_y = model.get_weights()[3] # (3, 4)
b_y = model.get_weights()[4] # (4,)

#######################
# Changing parameters #
#######################
# Manually, some parameters have been rounded to try to interpret them.
# However the model is quite complex here and it is difficult to make the 
# weights fully interpretable.
b_y = 10 * np.array([-1, -1, 0, -2])

W_y = 10 * np.array([[2, -2, -2, -2],
                     [1,  1, -2,  0],
                     [2, -2,  0,  0]])

# W_x.round(2).tolist()
W_x = np.array([
[  0, 0,  2,    1.69,  0.46,  1.11,    0.35,  0.57,  2.22,    2.23, -0.36, 20],
[  0, 2,  1,    0.72, -2.54,   -20,    1.25,    20,   -20,      -3,  3,    20],
[  0, 2, -2,   -0.37,   -20,   -20,    0.56,   -20,  0.50,      -3,  3,   -20],
[  1, 2,  1,   -0.52,    20,  1.36,    0.87,  0.58,  0.79,      -3,  3,     0]
])

W_h = np.array([
[0.69,  0.31, -0.37,  0,  0, -0.86,     0, -0.61, -1.57,   0.40,  0.18, -1.78],
[0.41,  1.20, -0.87,  0,  0.50, -0.50,  0,  2.00, -1.63,  -0.20,  1.11,  6.30],
[0.46, -0.54,  0.72,  0, -1,  0.68,     0, -0.76,  0,     -0.40, -0.31, -1.57]
])
 
b_h = np.array([
2.83, 1.16, 2.09,    1.35, 0.31, 1.30,    0.18, 0.66, 0.64,    0.21, 0.74, 3
])

model.set_weights([W_x, W_h, b_h, W_y, b_y])
model.fit(inputs, outputs, epochs = 1, batch_size = 32)
# Loss about 2e-12 with those manual parameters.

#########################
#########################
## C/ Modeling via GRU ##
#########################
#########################

##########################
# Create and train model #
##########################

from keras.layers import GRU, TimeDistributed

model=Sequential()
dim_in = m
dim_out = m
nb_units = 5

model.add(GRU(input_shape=(None, dim_in),
                    return_sequences=True, 
                    units=nb_units))
model.add(TimeDistributed(Dense(activation='sigmoid', units=dim_out)))
model.compile(loss = 'mse', optimizer = 'rmsprop')
# Each epoch cost 6 seconds with 1 unit
np.random.seed(1337)
model.fit(inputs, outputs, epochs = 500, batch_size = 32)

# Here for GRU:
# epoch  100   200   300   400   500
# unit
#    1  2e-2  2e-2  2e-2  2e-2  2e-2
#    2  1e-2  2e-3  5e-4  3e-4  2e-4
#    3  2e-3  3e-6  2e-8  5e-9  2e-9
#    5  3e-4  1e-8  1e-9  8e-10 5e-10

create('models')
to_save = False
if(to_save):
    model.save('models/2_C_coloring_gru.h5')
else:
    # model has been saved with nb_units = 3
    from keras.models import load_model
    model = load_model('models/2_C_coloring_gru.h5')

##########################
# Checking model results #
##########################
# Keeping in memory short and long dependencies as before.
# Code in Part A can be used.

###########
# Weights #
###########
# See 1_math_structure_of_rnn.py for more information
W_x = model.get_weights()[0] # (4, 9)
W_h = model.get_weights()[1] # (3, 9)
b_h = model.get_weights()[2] # (9,)
W_y = model.get_weights()[3] # (3, 4)
b_y = model.get_weights()[4] # (4,)