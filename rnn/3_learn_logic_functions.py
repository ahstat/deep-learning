############################################################
# 'For all' and other logic functions in RNN, LSTM and GRU #
# 2018/01/17 - 2018/01/23                                  #
############################################################
# Learning three logic functions using recurrent networks.
# Those problems contain long dependence and are easier compared to 
# 2_coloring_rnn.py.
#
# Functions learnt are 'for all', 'there exists' and 'there exists a unique'.
#
# (using tensorflow 1.1.0 and keras 2.0.3)
#
# - In part A, we make learning easy by selecting fixed size vectors. We train
# models with simple RNN, LSTM and GRU. We usually need few units. 
#   Interpretation of models is too difficult because they cannot generalize to 
# all size of inputs, i.e. outputs are only correct for selected size.
#   Thus, the model doesn't follow classic logic to guess functions.
#
# - In part B, we manage learning with different size vectors. We focus on
# learning with GRU. We interpret in details models built with function
# 'for all' and 'there exists'.
#   GRU usually takes 'tanh' activation and 'sigmoid' recurrent activation.
# This choice is discussed, and for logic purpose, we may wish to take sigmoid
# function for activation.
#
# - In part C, we replace 'tanh' activation with 'sigmoid'in GRU and model
# the 'there exists a unique' function. Model is trained and fully interpreted.
#   Finally, trained model is 'marbled' by replacing all sigmoid with Dirac
# delta function.
#
# Organization of this code:
# - A/ With fixed size vectors, learning functions with RNN, LSTM and GRU.
# - B/ With different size vectors, learning functions with GRU and interpret
#      'for all' and 'there exists'.
# - C/ With different size vectors replace activation function with 'sigmoid'
#      in GRU, then learn and interpret 'there exists a unique' function.

import os
import tensorflow
import numpy as np
tensorflow.VERSION
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN, LSTM, GRU
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

#############################################################################
#############################################################################
## A/ Fixed size modeling of logic functions with simple RNN, LSTM and GRU ##
#############################################################################
#############################################################################

######################################################
# Functions we wish to learn with recurrent networks #
######################################################
# The aim is to learn 'for_all' and 'there_exists' logic functions.
# Training and test set will be created using those true functions.

## 'For all' problem
# Given a vector of boolean of any size T, we output 1 is all elements
# of the vector are 1.

## 'There exists' problem
# Given a vector of boolean of any size T, we output 1 is there exists
# an element equal to 1.

## 'There exists a unique' problem
# Given a vector of boolean of any size T, we output 1 is there exists a unique
# element equal to 1.

def for_all(my_input):
    return(int(all(my_input == 1)))

def there_exists(my_input):
    return(int(any(my_input == 1)))

def there_exists_a_unique(my_input):
    return(int(sum(my_input) == 1))

for_all(np.array([0,0,0])) # 0
for_all(np.array([1,1,0])) # 0
for_all(np.array([0,0,1])) # 0
for_all(np.array([1,1,1])) # 1

there_exists(np.array([0,0,0])) # 0
there_exists(np.array([1,1,0])) # 1
there_exists(np.array([0,0,1])) # 1
there_exists(np.array([1,1,1])) # 1

there_exists_a_unique(np.array([0,0,0])) # 0
there_exists_a_unique(np.array([1,1,0])) # 0
there_exists_a_unique(np.array([0,0,1])) # 1
there_exists_a_unique(np.array([1,1,1])) # 0

######################################
# Sample data with fixed size vector #
######################################
## All binary vectors of size n are computed
n = 5

def combinations(n = 5, my_list = [0,1]):
    # From https://stackoverflow.com/questions/4709510
    my_rollaxis = np.rollaxis(np.indices((len(my_list),) * n), 0, n + 1)
    out = np.array(my_list)[my_rollaxis.reshape(-1, n)]
    return(out)

inputs = combinations(n)

## Three outputs sets are computed, one for each function we will learn
outputs_A = np.array([for_all(x) for x in inputs])
outputs_E = np.array([there_exists(x) for x in inputs])
outputs_EU = np.array([there_exists_a_unique(x) for x in inputs])

## Reshape inputs and outputs (to be able to use Keras on them)
inputs = inputs.reshape(inputs.shape + (1,)) # (32, 5, 1)
# shape is (sample, T, dim_in)

# Not ok: outputs = outputs.reshape(-1, 1, 1) # (32, 1, 1)
outputs_A = outputs_A.reshape(-1, 1) # (32, 1)
outputs_E = outputs_E.reshape(-1, 1)
outputs_EU = outputs_EU.reshape(-1, 1)
# shape is (sample, dim_out), since we want to do many-to-one.

######################
# Train the 9 models #
######################
def create_model(model_type = 'GRU', nb_units = 1, dim_in = 1, dim_out = 1):
    model=Sequential()    
    model = Sequential()
    
    if model_type == 'GRU':
        model.add(GRU(input_shape=(None, dim_in), units=nb_units))
    elif model_type == 'LSTM':
        model.add(LSTM(input_shape=(None, dim_in), units=nb_units))
    else:
        model.add(SimpleRNN(input_shape=(None, dim_in), units=nb_units))
    
    model.add(Dense(activation='sigmoid', units=dim_out))
    #model.compile(loss = 'mse', optimizer = 'rmsprop', 
    #              metrics=['accuracy'])
    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', 
                  metrics=['accuracy'])
    return(model)

def model_func(inputs, outputs, 
               model_type = 'GRU', epochs = 20000, nb_units = 1):
    dim_in = inputs.shape[2]
    dim_out = outputs.shape[1]
    
    model = create_model(model_type, nb_units, dim_in, dim_out)
    #np.random.seed(1337)
    model.fit(inputs, outputs, epochs = epochs, batch_size = 32)
    return(model)

# Note: Since we are using single size vectors, training models can be
# incorrect for generalization on other sequences.
#
# For 'for_all' and 'there_exists', loss is sometimes stuck around 
# 1/32 = 0.0313 for a long time (or forever for some runs), and then decrases 
# under 1e-6. This is because 1 of 32 in the sample is different.
#
# For 'there_exists_unique', with 1 unit, learning is only successful 
# with LSTM.
# - With simpleRNN: stuck with a loss of 0.1298, 0.1154 or 0.0313, depending 
#   of tries (10 tries with 1 unit). Model saved with loss = 0.0313.
# - With GRU, stuck with a loss of 0.0274, 0.0319 (10 tries with 1 unit). 
#   Model saved with loss = 0.0274.
# Possible cause: LSTM contain h_t and C_t, which allows more learning with 
# one unit compared to simpleRNN and GRU.
model_AR = model_func(inputs, outputs_A, model_type = 'SimpleRNN') # 2e-10
model_AL = model_func(inputs, outputs_A, model_type = 'LSTM') # 3e-7
model_AG = model_func(inputs, outputs_A, model_type = 'GRU') # 3e-10
model_ER = model_func(inputs, outputs_E, model_type = 'SimpleRNN') # 3e-10
model_EL = model_func(inputs, outputs_E, model_type = 'LSTM') # 4e-10
model_EG = model_func(inputs, outputs_E, model_type = 'GRU') # 3e-10
model_EUR = model_func(inputs, outputs_EU, model_type = 'SimpleRNN') # stuck
model_EUL = model_func(inputs, outputs_EU, model_type = 'LSTM') # 3e-8
model_EUG = model_func(inputs, outputs_EU, model_type = 'GRU') # stuck

# For stuck models, we compute models with 2 units
# EUR2 was ok only during second try and after 40000 iterations
model_EUR2 = model_func(inputs, outputs_EU, model_type = 'SimpleRNN', 
                        nb_units = 2) # :
model_EUG2 = model_func(inputs, outputs_EU, model_type = 'GRU', 
                        nb_units = 2) # 4e-10

create('models')
to_save = False
if(to_save):
    model_AR.save('models/3_A_forall_simplernn.h5')
    model_AL.save('models/3_A_forall_lstm.h5')
    model_AG.save('models/3_A_forall_gru.h5')
    model_ER.save('models/3_A_exists_simplernn.h5')
    model_EL.save('models/3_A_exists_lstm.h5')
    model_EG.save('models/3_A_exists_gru.h5')
    model_EUR.save('models/3_A_existsunique_simplernn_stuck.h5')
    model_EUL.save('models/3_A_existsunique_lstm.h5')
    model_EUG.save('models/3_A_existsunique_gru_stuck.h5')
    
    model_EUR2.save('models/3_A_existsunique_simplernn_2units.h5')
    model_EUG2.save('models/3_A_existsunique_gru_2units.h5')
else:
    # models have been saved with epochs = 20000 and nb_units = 1 
    from keras.models import load_model
    model_AR = load_model('models/3_A_forall_simplernn.h5')
    model_AL = load_model('models/3_A_forall_lstm.h5')
    model_AG = load_model('models/3_A_forall_gru.h5')
    model_ER = load_model('models/3_A_exists_simplernn.h5')
    model_EL = load_model('models/3_A_exists_lstm.h5')
    model_EG = load_model('models/3_A_exists_gru.h5')
    model_EUR = load_model('models/3_A_existsunique_simplernn_stuck.h5')
    model_EUL = load_model('models/3_A_existsunique_lstm.h5')
    model_EUG = load_model('models/3_A_existsunique_gru_stuck.h5')
    
    # model with epochs = 40000 and nb_units = 2
    model_EUR2 = load_model('models/3_A_existsunique_simplernn_2units.h5')
    
    # model with epochs = 20000 and nb_units = 2
    model_EUG2 = load_model('models/3_A_existsunique_gru_2units.h5')

##########################
# Checking model results #
##########################
models = [model_AR, model_AL, model_AG, # all three 100% accuracy
          model_ER, model_EL, model_EG, # all three 100% accuracy
          model_EUR, model_EUL, model_EUG, 
          # model_EUL has 100% accuracy ;
          # model_EUR and model_EUG cannot learn [0,0,0,0,0] --> 0. All others 
          # inputs give correct outputs.
          model_EUR2, model_EUG2] # both have 100% accuracy
[print(np.around(model.predict(np.array(inputs))).flatten().tolist())
  for model in models]

outputs = [outputs_A, outputs_A, outputs_A,
           outputs_E, outputs_E, outputs_E,
           outputs_EU, outputs_EU, outputs_EU]

###############################################
# Analysis of weights and changing parameters #
###############################################

##
# For all - Simple RNN
##
i = 0
model = models[i]
output = outputs[i]
## Description of the Elman network
# h_0 = [0, 0, 0]
# h_t = tanh(W_h * x_t + U_h * h_{t-1} + b_h)  (A)
# y_t = sigmoid(W_y * h_t + b_y)               (B)
W_h = model.get_weights()[0]
U_h = model.get_weights()[1]
b_h = model.get_weights()[2]
W_y = model.get_weights()[3]
b_y = model.get_weights()[4]
W_h = np.array([[2.52]]) # 2.5 does not work.
U_h = np.array([[2.5]])
b_h = np.array([-2.5])
W_y = np.array([[20]])
b_y = np.array([0])
model.set_weights([W_h, U_h, b_h, W_y, b_y])
model.fit(inputs, outputs[i], epochs = 1, batch_size = 32)

## From h to y
# From (B), we have: y_t = sigmoid(W_y' * h_t + b_y) = sigmoid(20 * h_t)
def y_from_h(h, W_y, b_y):
  return np.around([sigmoid(x) for x in np.dot(W_y.transpose(), h) + b_y], 2)
y_from_h([-1], W_y, b_y) # 0
y_from_h([0], W_y, b_y) # 0.5
y_from_h([1], W_y, b_y) # 1
# Since h_t is in [-1, 1], we have :
# * y_t == 1 iff h_t = 1, and 
# * y_t == 0 iff h_t = -1.

## From h to h
# From (A), we have: h_t = tanh(W_h' * x_t + U_h' * h_{t-1} + b_h)
#                    h_t = tanh(2.52 * x_t + 2.5 * h_{t-1} - 2.5)
def h_from_h(h, x, W_h, U_h, b_h):
  return np.around([math.tanh(x) for x in np.dot(W_h.transpose(), x) + 
                                          np.dot(U_h.transpose(), h) + 
                                          b_h], 2)
h_from_h(h = [-1], x = [0], W_h = W_h, U_h = U_h, b_h = b_h) # -1
h_from_h(h = [-1], x = [1], W_h = W_h, U_h = U_h, b_h = b_h) # -0.99
h_from_h(h = [0], x = [0], W_h = W_h, U_h = U_h, b_h = b_h) # -0.99
h_from_h(h = [0], x = [1], W_h = W_h, U_h = U_h, b_h = b_h) # 0.02
h_from_h(h = [1], x = [0], W_h = W_h, U_h = U_h, b_h = b_h) # 0
h_from_h(h = [1], x = [1], W_h = W_h, U_h = U_h, b_h = b_h) # 0.99

# If we reach h == -1, we are stuck in h close to -1.
h_from_h(h = [-1], x = [1], W_h = W_h, U_h = U_h, b_h = b_h)
h_from_h(h = [-0.99], x = [1], W_h = W_h, U_h = U_h, b_h = b_h)
# This can be seen from the formula: 
# With h_{t-1} < -0.5, we have: h_t < tanh(2.52 * x_t - 2.5 * 0.8 - 2.5)
#                               h_t <  -0.84 (both for x_t = 1 and x_t = 0).

# If we encounter x_t = 0, we are going to h_t = -1 unless h_{t-1} is large.
# Cf from the formula:
# h_t = tanh(2.52 * 0 + 2.5 * h_{t-1} - 2.5) = tanh(2.5. [h_{t-1} - 1])

## It can be more difficult to learn one '0' appearing if we have seen many
# '1' before.
h_from_h(h = [0], x = [1], W_h = W_h, U_h = U_h, b_h = b_h) # to h = 0.02
h_from_h(h = [0.02], x = [1], W_h = W_h, U_h = U_h, b_h = b_h) # to h = 0.07
h_from_h(h = [0.07], x = [1], W_h = W_h, U_h = U_h, b_h = b_h) # to h = 0.19
h_from_h(h = [0.19], x = [1], W_h = W_h, U_h = U_h, b_h = b_h) # to h = 0.49
h_from_h(h = [0.49], x = [1], W_h = W_h, U_h = U_h, b_h = b_h) # to h = 0.85
# If we would like to consider larger sequences, it remove some learning.
h_from_h(h = [0.85], x = [1], W_h = W_h, U_h = U_h, b_h = b_h) # to h = 0.97
h_from_h(h = [0.97], x = [1], W_h = W_h, U_h = U_h, b_h = b_h) # to h = 0.99
h_from_h(h = [0.99], x = [0], W_h = W_h, U_h = U_h, b_h = b_h) # close to 0-
# and:
y_from_h([-0.00001], W_y, b_y) # 0.5-.
# We will still select for_all([1,1,1,1,...,1,1,0]) = 0, but it is more
# difficult here to learn it.

##
# For all - LSTM
##
i = 1
model = models[i]
output = outputs[i]
#
W_x = model.get_weights()[0]
W_h = model.get_weights()[1]
b_h = model.get_weights()[2]
W_y = model.get_weights()[3]
b_y = model.get_weights()[4]
#
W_x = np.array([[-5, 0,  0,  0]]) # W_ix / W_fx / W_Cx / W_ox
W_h = np.array([[20, 0,  0,  0]]) # W_ih / W_fh / W_Ch / W_oh
b_h = np.array([  1, 0, 20, 20])  # b_i  / b_f  / b_C  / b_o
W_y = np.array([[-20]])
b_y = np.array([ 6])
model.set_weights([W_x, W_h, b_h, W_y, b_y])
model.fit(inputs, outputs[i], epochs = 1, batch_size = 32)

## From h to y
# We have 
# * y_t == 1 for h_t in [-1, 0.1] 
# * y_t == 0 for h_t > 0.6

## From h to h
# For the forget gate, we have f_t = 0.5 (cf W_f = 0 and f_t = sigma(W_f)
# For the hatC_t gate, we have hatC_t = 1 (cf b_C = 20)
# For the output gate, we have o_t = 1 (cf b_o = 20)
# The input gate is sigmoid[-5 x_t + 20 h_{t-1} + 1].

# So C_t is updated with: 0.5 * C_t_minus_1 + i_t
# And h_t is tanh(C_t)

##
# For all - GRU
##
i = 2
model = models[i]
output = outputs[i]
#
W_x = model.get_weights()[0]
W_h = model.get_weights()[1]
b_h = model.get_weights()[2]
W_y = model.get_weights()[3]
b_y = model.get_weights()[4]
#
W_x = np.array([[0,  0, -2.7]]) # W_zx / W_rx / W_ox
W_h = np.array([[0,  0,  3]]) # W_zh / W_rh / W_oh
b_h = np.array([-20,  0,  2.6])  # b_z  / b_r  / b_o
W_y = np.array([[-20]])
b_y = np.array([0])

model.set_weights([W_x, W_h, b_h, W_y, b_y])
model.fit(inputs, outputs[i], epochs = 1, batch_size = 32)

## From h to h
# For the z gate, we have z_t = 0
# For the r gate, we have r_t = 0.5
# For the o gate, we have \tilde{h} = tanh[-2.7 x_t + 3 h_{t-1} + 2.6].
# And then h_t = \tilde{h} cf z_t = 0 (formula cf 1_math_structure_of_rnn.py,
# in the picture 1_E_GRU.png it is 1 - z_t).

##
# There exists
##
# Not done. It should be similar with for all for the 3 models.

##
# There exists a unique - LSTM
##
# We only focus on LSTM. The 2 other models need two units and are more
# difficult to interpret.
i = 7

model = models[i]
output = outputs[i]
#
W_x = model.get_weights()[0]
W_h = model.get_weights()[1]
b_h = model.get_weights()[2]
W_y = model.get_weights()[3]
b_y = model.get_weights()[4]
#
W_x = np.array([[  0,   0,   100, 100]]) # W_ix / W_fx / W_Cx / W_ox
W_h = np.array([[  0,   0,     0,  15]]) # W_ih / W_fh / W_Ch / W_oh
b_h = np.array([ 100, 100, -0.55,  10])  # b_i  / b_f  / b_C  / b_o
W_y = np.array([[-20]])
b_y = np.array([-7])
model.set_weights([W_x, W_h, b_h, W_y, b_y])
model.fit(inputs, outputs[i], epochs = 1, batch_size = 32)
# After changing parameters, we have a loss of 4e-8.

## From h to y
# We have 
# * y_t == 0 for h_t >= 0
# * y_t == 1 for h_t <= -0.5

## From h to h
# * input gate: sigmoid(100) = 1, so we fully take input in all cases
# (there is no preference for some configuration of x_t or h_{t-1}).
#
# * forget gate: sigmoid(100) = 1, so we don't forget anything in C_{t-1}
# (there is no forget for some configuration of x_t or h_{t-1}).
#
# * hatC_t gate, we have hatC_t = tanh(100 x - 0.55),
# so in the case x == 1, we obtain hatC_t = 1,
#    in the case x == 0, we obtain hatC_t = tanh(-0.55) = -0.5.
#
# * C_t is updated as follows (cf f_t = 1 and i_t = 1): 
#       C_t =  C_{t-1} + \hat{C}_t
#
# C_t will: 
# - increase by 1 each time seeing x=1, 
# - decrease by 0.5 each time seeing x=0.
#
# * o_t = sigmoid[100x + 15h + 10]

# If x = 1, o_t = 1
# If x = 0, o_t = sigmoid[15h + 10]
# If x = 0 and h = 0, o_t = 1
# If x = 0 and h = 1, o_t = 1
# If x = 0 and h = -1, o_t = 0
#
# Then h_t = o_t * tanh(C_t)

## Application on some inputs
# The model is difficult to understand even in this case, because it works
# only well with 5 length elements:
print(model.predict(np.array([[[0]]]))) # output close to 1, but should be 0
print(model.predict(np.array([[[0], [0]]]))) # very close to 1
print(model.predict(np.array([[[0], [0], [0]]]))) # close to 0
print(model.predict(np.array([[[0],[0],[0],[0],[0]]]))) # very close to 0.

##
# Conclusion
##
# The interpretation of learnt models is not straightforward, and it apparently 
# does not follow the math logic: Some models can work well with 5 size 
# elements but guess wrong for other sizes.

############################################################################
############################################################################
## B/ Modeling of logic functions with vectors of different size with GRU ##
############################################################################
############################################################################
# For LSTM with different sequence lengths, there is 2 ways to proceed:
# 1/ Padding sequences with masking,
# 2/ Doing batch after partitioning vectors of same length.
#
# We use this second method here, by separating learning for each size.
# See also https://stackoverflow.com/questions/44873387
# "If you don't want to use padding, you can separate your data in smaller 
# batches, each batch with a distinct sequence length"
# Code in https://stackoverflow.com/questions/46144191 , but not used here.

##########################################
# Sample data with different size vector #
##########################################
# All non-empty binary vectors will be computed up to size 'n',
# containing 2**(n+1)-1 elements.
# Binary vectors with larger size are sampled up to size 'n' and such that the
# whole data set has size 'sample_size
n = 5

##
# Original strategy for inputs
##
# Size of inputs is: 1, 2, 3, 4, 5.
#inputs_all = [combinations(i) for i in range(1, n+1)]

##
# 'Repeat more smaller ones' strategy
##
# Size of inputs is: 1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5.
# We use this strategy when it is harder to learn small ones.
inputs_all = [[combinations(i) for i in range(1, k+1)] for k in range(1, n+1)]
flatten = lambda l: [item for sublist in l for item in sublist]
inputs_all = flatten(inputs_all)

## Three outputs sets are computed, one for each function we will learn
outputs_A_all = [np.array([for_all(x) for x in inputs]) 
    for inputs in inputs_all]
outputs_E_all = [np.array([there_exists(x) for x in inputs]) 
    for inputs in inputs_all]
outputs_EU_all = [np.array([there_exists_a_unique(x) for x in inputs]) 
    for inputs in inputs_all]

## Reshape inputs and outputs (to be able to use Keras on them)
inputs_all = [inputs.reshape(inputs.shape + (1,)) for inputs in inputs_all]
# shape is (sample, T, dim_in) for each

outputs_A_all = [outputs_A.reshape(-1, 1) for outputs_A in outputs_A_all]
outputs_E_all = [outputs_E.reshape(-1, 1) for outputs_E in outputs_E_all]
outputs_EU_all = [outputs_EU.reshape(-1, 1) for outputs_EU in outputs_EU_all]
# shape is (sample, dim_out), since we want to do many-to-one.

#########################
# Train models with GRU #
#########################
repetitions = 10

# * Logic expected for 'for all'
# With 1 layer, with 1 unit, we can compute the function:
# Initially h = 0
# - If h=0, x=0, then we do: h=1.
# - If h=0, x=1, then we do: h=0.
# - If h=1, x=*, then we do: h=1. (* means anything, e.g. 0 or 1)
# Finally: y=0 if h=1 ; y=1 if h=0.
def create_model1():
    model=Sequential()    
    model = Sequential()
    dim_in = 1
    dim_out = 1
    model.add(GRU(input_shape=(None, dim_in), units=1))
    model.add(Dense(activation='sigmoid', units=dim_out))
    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', 
                  metrics=['accuracy'])
    return(model)

epochs = 100
model_AG = create_model1()
# Maybe not the best way to learn, but reach accuracy = 1 for all sizes.
for k in range(repetitions):
  for inputs, outputs in zip(inputs_all, outputs_A_all):
      model_AG.fit(inputs, outputs, epochs = epochs, batch_size = 32)

# * Logic expected for 'there exists'
# With 1 layer, with 1 unit, we can compute the function:
# Initially h = 0
# - If h=0, x=0, then we do: h=0.
# - If h=0, x=1, then we do: h=1.
# - If h=1, x=*, then we do: h=1. (* means anything, e.g. 0 or 1)
# Finally: y=1 if h=1 ; y=0 if h=0.
epochs = 100
model_EG = create_model1()
for k in range(repetitions):
  for inputs, outputs in zip(inputs_all, outputs_E_all):
      model_EG.fit(inputs, outputs, epochs = epochs, batch_size = 32)

# * Logic for 'there exists a unique'
# Initially, I though it was not possible to train this function with 1 layer
# and 2 nodes. But it will work, see in Part C.
# 
# Logic can be as follows
# Initially h = 0 and h' = 0
# - If h=0, h'=0, x=0, then we do: h=0, h'=0.
# - If h=0, h'=0, x=1, then we do: h=1, h'=0.
# - If h=1, h'=0, x=0, then we do: h=1, h'=0.
# - If h=1, h'=0, x=1, then we do: h=*, h'=1. (* means anything, e.g. 0 or 1)
# - If h=*, h'=1, x=*, then we do: h=*, h'=1.
# Finally: we deduce y from h and h'.
# On the whole: y=0 if h'=1 ; y=0 if h'=0 and h=0 ; y=1 if h'=0 and h=1.
def create_model2():
    # model with 2 hidden layers
    model=Sequential()    
    model = Sequential()
    dim_in = 1
    dim_out = 1
    model.add(GRU(input_shape=(None, dim_in), units=1,
                  return_sequences=True))
    model.add(GRU(input_shape=(None, dim_in), units=2))
    model.add(Dense(activation='sigmoid', units=dim_out))
    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', 
                  metrics=['accuracy'])
    return(model)

epochs = 100
model_EUG = create_model2()
for k in range(repetitions):
  for inputs, outputs in zip(inputs_all, outputs_EU_all):
      model_EUG.fit(inputs, outputs, epochs = epochs, batch_size = 32)

##
# Save models
##
create('models')
to_save = False
if(to_save):
    model_AG.save('models/3_B_forall_gru.h5')
    model_EG.save('models/3_B_exists_gru.h5')
    model_EUG.save('models/3_B_existsunique_gru.h5')
else:
    from keras.models import load_model
    model_AG = load_model('models/3_B_forall_gru.h5')
    model_EG = load_model('models/3_B_exists_gru.h5')
    model_EUG = load_model('models/3_B_existsunique_gru.h5')

##
# Check evolution of global loss (i.e. for all 63 elements)
##
def cross_entropy_func(y, y_hat):
    return(-np.mean((y*np.log(y_hat+1e-20) + (1-y)*np.log(1-y_hat+1e-20))))

def cross_entropy_whole(model, inputs_all, outputs_all):
    y_hat = np.array([])
    y = np.array([])
    for i in range(len(inputs_all)):
        y_hat_i = model.predict(inputs_all[i]).flatten()
        y_i = outputs_all[i].flatten()
        y_hat = np.concatenate([y_hat, y_hat_i])
        y = np.concatenate([y, y_i])
    cross_entropy = cross_entropy_func(y, y_hat)
    return(cross_entropy)

## During training of models, we can follow evolution of cross entropy.
# Here example with the previous model_EUG.

#model = create_model1()
#outputs_all = outputs_A_all
#model = create_model1()
#outputs_all = outputs_E_all
model = create_model2()
outputs_all = outputs_EU_all

cross_entropy_evol = []
for k in range(repetitions):
  for inputs, outputs in zip(inputs_all, outputs_all):
      model.fit(inputs, outputs, epochs = epochs, batch_size = 32)
      # Following only for cross entropy evolution
      cross_current = cross_entropy_whole(model, inputs_all, outputs_all)
      cross_entropy_evol.append(cross_current)
import matplotlib.pyplot as plt
plt.plot(cross_entropy_evol)
plt.xlabel('Iterations (one iteration = 100 epochs on a data set)')
plt.ylabel('Loss cross-entropy on the whole data sets')
plt.plot([math.log(x+1e-8, 10) for x in cross_entropy_evol])
plt.xlabel('Iterations (one iteration = 100 epochs on a data set)')
plt.ylabel('Loss cross-entropy on the whole data sets (log base 10)')

##########################
# Checking model results #
##########################
# Checking the model accuracy for size 1 to 10
# Models generalize to all size 1 to 10.

##
# For all
##
model = model_AG
for n in range(1, 10):
    inputs = combinations(n)
    inputs = inputs.reshape(inputs.shape + (1,))
    print(np.around(model.predict(inputs)).flatten().tolist())

##
# There exists
##
model = model_EG
for n in range(1, 10):
    inputs = combinations(n)
    inputs = inputs.reshape(inputs.shape + (1,))
    print(np.around(model.predict(inputs)).flatten().tolist())

##
# There exists a unique
##
model = model_EUG
for n in range(1, 13):
    inputs = combinations(n)
    inputs = inputs.reshape(inputs.shape + (1,))
    y_hat = np.around(model.predict(inputs)).flatten()
    y = np.array([there_exists_a_unique(x) for x in inputs])
    #print((y_hat - y).tolist())
    print(np.sum(np.abs(y_hat - y))) # 0 means OK, no differences
# Can generalize well (in a first try here, it could not generalize well).

#####################################
# Analysis of weights for 'for all' #
#####################################
model = model_AG

W_x = model.get_weights()[0] # (1, 1*3)
W_h = model.get_weights()[1] # (1, 1*3)
b_h = model.get_weights()[2] # (1*3,)
W_y = model.get_weights()[3] # (1, 1)
b_y = model.get_weights()[4] # (1,)

#coeff = 20
#W_x = coeff*np.array([[0,  0, -1]])
#W_h = coeff*np.array([[0,  1,  1]])
#b_h = coeff*np.array([-1,  0,  0.5])
#W_y = np.array([[-15]])
#b_y =  np.array([0]) 
# which is equivalent to:
coeff = 20
W_x = coeff*np.array([[ 0,  0,    1]])
W_h = coeff*np.array([[ 0, -1,    1]])
b_h = coeff*np.array([ -1,  0, -0.5])
W_y = np.array([[15]])
b_y =  np.array([0]) 
 
model.set_weights([W_x, W_h, b_h, W_y, b_y])

# Output of 0.0 means small difference between predicted and output
cross_entropy = np.array([])
for n in range(1, 13):
    inputs = combinations(n)
    inputs = inputs.reshape(inputs.shape + (1,))
    y_hat = model.predict(inputs).flatten()
    y = np.array([for_all(x) for x in inputs])
    cross_current = cross_entropy_func(y, y_hat)
    cross_entropy = np.append(cross_entropy, cross_current)
    print(sum(np.around(y_hat) - y).tolist())
print(sum(cross_entropy))

##
# Analysis
##
# From h to y:
# y = sigmoid(15h), i.e.:
# if h > 0, then y = 1,
# if h < 0, then y = 0.

# 'through_GRU_layer' must be loaded from '1_math_structure_of_rnn.py'
# I have not written an external module to keep sequential structure of this
# tutorial.
x_t = [0]
h_t_minus_1 = [0]
through_GRU_layer(x_t, h_t_minus_1, W_x, W_h, b_h)
# x=0 h=0 --> -1

x_t = [1]
h_t_minus_1 = [0]
through_GRU_layer(x_t, h_t_minus_1, W_x, W_h, b_h)
# x=1 h=0 --> 1

x_t = [0]
h_t_minus_1 = [1]
through_GRU_layer(x_t, h_t_minus_1, W_x, W_h, b_h)
# x=0 h=1 --> -1

x_t = [1]
h_t_minus_1 = [1]
through_GRU_layer(x_t, h_t_minus_1, W_x, W_h, b_h)
# x=1 h=1 --> 1

x_t = [0]
h_t_minus_1 = [-1]
through_GRU_layer(x_t, h_t_minus_1, W_x, W_h, b_h)
# x=0 h=-1 --> -1

x_t = [1]
h_t_minus_1 = [-1]
through_GRU_layer(x_t, h_t_minus_1, W_x, W_h, b_h)
# x=1 h=-1 --> -1

## Analysis:
# z_t = sigmoid(-20) = 0
# so:
# h_t = tilde_h_t = tilde_h_t = tanh(20x + 20hr - 10)
# (z replaced with 1-z in the Keras implementation)
#
# And:
# r_t = sigmoid(-20h) = {0 if h<0; 0.5 if h=0; 1 if h>0}
#
# So if h = 0:
# h_t = tanh(20x - 10)
#     =  1 if x = 1
#     = -1 if x = 0
#
#   if h = 1:
# h_t = tanh(20x + 20h -10)= tanh(20x -10) = 1
# 
#  if h = -1:
# h_t = tanh(20x - 20 -10)= tanh(20x -30) = -1
#
# So it works in all cases.
#
# Note: The initialization with tanh activation is difficult to interpret
# is this case (for example, r_t(0)=0.5 whereas r_t(0-)=0 and r_t(0+)=1).
# See the conclusion of this part B for discussion about replacing it with
# sigmoid activation.

##########################################
# Analysis of weights for 'there exists' #
##########################################
model = model_EG

W_x = model.get_weights()[0] # (1, 1*3)
W_h = model.get_weights()[1] # (1, 1*3)
b_h = model.get_weights()[2] # (1*3,)
W_y = model.get_weights()[3] # (1, 1)
b_y = model.get_weights()[4] # (1,)

coeff = 20
W_x = coeff*np.array([[0,   -1,   -1]])
W_h = coeff*np.array([[0,    0,    1]])
b_h = coeff*np.array([-1,  0.5,  0.5])
W_y = np.array([[-15]])
b_y =  np.array([0]) 
 
model.set_weights([W_x, W_h, b_h, W_y, b_y])

# Output of 0.0 means small difference between predicted and output
cross_entropy = np.array([])
for n in range(1, 13):
    inputs = combinations(n)
    inputs = inputs.reshape(inputs.shape + (1,))
    y_hat = model.predict(inputs).flatten()
    y = np.array([there_exists(x) for x in inputs])
    cross_current = cross_entropy_func(y, y_hat)
    cross_entropy = np.append(cross_entropy, cross_current)
    print(sum(np.around(y_hat) - y).tolist())
print(sum(cross_entropy))

##
# Analysis
##
# From h to y:
# y = sigmoid(-15h), i.e.:
# if h > 0, then y = 0,
# if h < 0, then y = 1.

# 'through_GRU_layer' must be loaded from '1_math_structure_of_rnn.py'
# I have not written an external module to keep sequential structure of this
# tutorial.
x_t = [0]
h_t_minus_1 = [0]
through_GRU_layer(x_t, h_t_minus_1, W_x, W_h, b_h)
# x=0 h=0 --> 1

x_t = [1]
h_t_minus_1 = [0]
through_GRU_layer(x_t, h_t_minus_1, W_x, W_h, b_h)
# x=1 h=0 --> -1

x_t = [0]
h_t_minus_1 = [1]
through_GRU_layer(x_t, h_t_minus_1, W_x, W_h, b_h)
# x=0 h=1 --> 1

x_t = [1]
h_t_minus_1 = [1]
through_GRU_layer(x_t, h_t_minus_1, W_x, W_h, b_h)
# x=1 h=1 --> -1

x_t = [0]
h_t_minus_1 = [-1]
through_GRU_layer(x_t, h_t_minus_1, W_x, W_h, b_h)
# x=0 h=-1 --> -1

x_t = [1]
h_t_minus_1 = [-1]
through_GRU_layer(x_t, h_t_minus_1, W_x, W_h, b_h)
# x=1 h=-1 --> -1

## Analysis:
# z_t = sigmoid(-20) = 0
# so:
# h_t = tilde_h_t = tilde_h_t = tanh(-20x + 20hr + 10)
# (z replaced with 1-z in the Keras implementation)
#
# And:
# r_t = sigmoid(-20x+10) = {1 if x<=0; 0 if x>0}
#
# So if x = 0:
# h_t = tanh(20h + 10)
#     = -1 if h = -1 # -1 is 'there exists', so keep it.
#     =  1 if h = 0  # If 0 or 'not exists', then go to 'not exists' i.e. 1
#     =  1 if h = 1
#
# and if x = 1:
# h_t = tanh(-20x + 10) = -1 # -1 is 'there exists'
#
# So it works in all cases.

###################################################
# Analysis of weights for 'there exists a unique' #
###################################################
model = model_EUG

# Layer 1 with 1 node
W_x = model.get_weights()[0] # (1, 1*3)
W_h = model.get_weights()[1] # (1, 1*3)
b_h = model.get_weights()[2] # (1*3,)

# Layer 2 with 2 nodes
W_xp = model.get_weights()[3] # (1, 2*3)
W_hp = model.get_weights()[4] # (2, 2*3)
b_hp = model.get_weights()[5] # (2*3,)

# Hidden to output
W_y = model.get_weights()[6] # (2, 1)
b_y = model.get_weights()[7] # (1,)

W_x = np.array([[-1.0, -1.0,  2.0]])
W_h = np.array([[ 1.0, -3.0, -1.0]])
b_h = np.array([ -0.7,  1.0, -0.7])
W_xp = np.array([[-0.7, -6.4,  2.7, -5.4, -7.0, -1.1]])
W_hp = np.array([[-1.0,  3.0,  1.5, -2.0,  1.2,  7.7],
                 [ 4.0, -3.0,  0.2, -0.8, -3.9,  7.2]])
b_hp = np.array([ -2.0,  1.0,  1.4,  1.0,  1.7,  3.2])
W_y = np.array([[-15], [15]])
b_y =  np.array([-15]) 
 
model.set_weights([W_x, W_h, b_h, W_xp, W_hp, b_hp, W_y, b_y])

# Output of 0.0 means small difference between predicted and output
cross_entropy = np.array([])
for n in range(1, 15):
    inputs = combinations(n)
    inputs = inputs.reshape(inputs.shape + (1,))
    y_hat = model.predict(inputs).flatten()
    y = np.array([there_exists_a_unique(x) for x in inputs])
    cross_current = cross_entropy_func(y, y_hat)
    cross_entropy = np.append(cross_entropy, cross_current)
    print(sum(np.around(y_hat) - y).tolist())
print(sum(cross_entropy))

# /!\ With this model, not sure if it will generalize well up to infinity.

##
# Analysis possible but boring.
##
# Model is not as simple as expected.
# In part because of 'h=0' at initialization with tanh (see conclusion).
# In the following part C, we retrain the model with sigma instead of tanh.

##############
# Conclusion #
##############
#     RNN models are working and give expected results.
#
#     The initialization with tanh activation is difficult to interpret
# in this part. It would be better to use sigmoid activation, and then
# h stands in [0,1], and 0 the initial value is one of the bounds.
#     This seems OK for 0/1 activation as here. In general, tanh works best
# over sigmoid. See cs224d.stanford.edu/lectures/CS224d-Lecture6.pdf
# and Aaron Schumacher in https://stackoverflow.com/questions/40761185
# "Sigmoid output is always non-negative; values in the state would only 
# increase." (not the other answer!).
#
#     Weights are manually reset to make the global network more 'logic'.
# Maybe this kind of set weights can be automated. In the following part C,
# we replace hard_sigmoid with Dirac function after training (called marble
# network in part C).
#    Another inspiration could be to use binary or ternary weights,
# see for example this article:
# "BinaryConnect: Training Deep Neural Networks with binary weights during 
# propagations"

####################################################################
####################################################################
## C/ Learning 'there exists a unique' with sigma instead of tanh ##
####################################################################
####################################################################

###########################################################
# Model with 1 layer, 2 units and hard_sigmoid activation #
###########################################################
def create_model3():
    model=Sequential()    
    model = Sequential()
    dim_in = 1
    dim_out = 1
    #model.add(GRU(input_shape=(None, dim_in), units=1,
    #              return_sequences=True, 
    #          activation='hard_sigmoid'))
    model.add(GRU(input_shape=(None, dim_in), units=2, 
                  activation='hard_sigmoid'))
    model.add(Dense(activation='sigmoid', units=dim_out))
    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', 
                  metrics=['accuracy'])
    return(model)

model = create_model3()
outputs_all = outputs_EU_all

epochs = 100 # do not select 1000 causing overfitting of size=2
cross_entropy_evol = []
repetitions = 50
for k in range(repetitions):
  for inputs, outputs in zip(inputs_all, outputs_all):
      model.fit(inputs, outputs, epochs = epochs, batch_size = 32)
      # Following only for cross entropy evolution
      cross_current = cross_entropy_whole(model, inputs_all, outputs_all)
      cross_entropy_evol.append(cross_current)

## Plotting cross-entropy
import matplotlib.pyplot as plt
plt.plot(cross_entropy_evol)
plt.xlabel('Iterations (one iteration = 100 epochs on a data set)')
plt.ylabel('Loss cross-entropy on the whole data sets')
plt.plot([math.log(x+1e-9, 10) for x in cross_entropy_evol])
plt.xlabel('Iterations (one iteration = 100 epochs on a data set)')
plt.ylabel('Loss cross-entropy on the whole data sets (log base 10)')

## Save model
to_save = False
if(to_save):
    model.save('models/3_C_existsunique_gru_sigmoid.h5')
else:
    from keras.models import load_model
    model = load_model('models/3_C_existsunique_gru_sigmoid.h5')

## Weights
# Layer 1 with 2 nodes
W_x = model.get_weights()[0] # (1, 2*3)
W_h = model.get_weights()[1] # (2, 2*3)
b_h = model.get_weights()[2] # (2*3,)

# Hidden to output
W_y = model.get_weights()[3] # (2, 1)
b_y = model.get_weights()[4] # (1,)

## Reset weights
coeff = 10
W_x = coeff*np.array([[-2,  0, -2,  0,  2,  0]])
W_h = coeff*np.array([[ 0, -2,  0,  2,  0, -2],
                      [ 0,  2, -2, -2,  0, -1]])
b_h = coeff*np.array([  1,  1,  1, -1,  0,  1])
W_y = np.array([[40], [-40]])
b_y = np.array([-20])
# This manual process needs to be automated! de-Regularization?
 
model.set_weights([W_x, W_h, b_h, W_y, b_y])

# Output of 0.0 means small difference between predicted and output
cross_entropy = np.array([])
for n in range(1, 15):
    inputs = combinations(n)
    inputs = inputs.reshape(inputs.shape + (1,))
    y_hat = model.predict(inputs).flatten()
    y = np.array([there_exists_a_unique(x) for x in inputs])
    cross_current = cross_entropy_func(y, y_hat)
    cross_entropy = np.append(cross_entropy, cross_current)
    print(sum(np.around(y_hat) - y).tolist())
print(sum(cross_entropy))
#
# z_t = sigma(-20x+10)
# z'_t = sigma(-20h+20h'+10)
# r_t = sigma(-20x-20h'+10)
# r'_t = sigma(20h-20h'-10)
# ~h_t = sigma(20x)
# ~h'_t = sigma(-20rh-10r'h'+10)
#
# h_t = z_t h_{t-1} + (1-z_t) ~h_t
# h'_t = z'_t h'_{t-1} + (1-z'_t) ~h'_t
#
# Different cases:
# x=0 (h_{t-1}=0, h'_{t-1}=0): h_t=h_{t-1} ; h'_t=h'_{t-1}
# x=1 (h_{t-1}=0, h'_{t-1}=0): h_t=1       ; h'_t=h'_{t-1}
# x=0 (h_{t-1}=1, h'_{t-1}=0): h_t=h_{t-1} ; h'_t=0
# x=1 (h_{t-1}=1, h'_{t-1}=0): h_t=1       ; h'_t=1
# x=0 (h_{t-1}=0, h'_{t-1}=1): h_t=h_{t-1} ; h'_t=h'_{t-1}
# x=1 (h_{t-1}=0, h'_{t-1}=1): h_t=1       ; h'_t=h'_{t-1}
# x=0 (h_{t-1}=1, h'_{t-1}=1): h_t=h_{t-1} ; h'_t=h'_{t-1}
# x=1 (h_{t-1}=1, h'_{t-1}=1): h_t=1       ; h'_t=h'_{t-1}
#
# In details:
# x=0 (h=0, h'=0): z_t=1 ; z'_t=1 ; r_t=1 ; r'_t=0 ; ~h_t=0.5
#                  rh=0;r'h'=0; ~h'_t=1; 
# x=1 (h=0, h'=0): z_t=0 ; z'_t=1 ; r_t=0 ; r'_t=0 ; ~h_t=1
#                  rh=0;r'h'=0; ~h'_t=1; 
# x=0 (h=1, h'=0): z_t=1 ; z'_t=0 ; r_t=1 ; r'_t=1 ; ~h_t=0.5
#                  rh=1;r'h'=0; ~h'_t=0; 
# x=1 (h=1, h'=0): z_t=0 ; z'_t=0 ; r_t=0 ; r'_t=1 ; ~h_t=1
#                  rh=0;r'h'=0; ~h'_t=1; 
# x=0 (h=0, h'=1): z_t=1 ; z'_t=1 ; r_t=0 ; r'_t=0 ; ~h_t=0.5
#                  rh=0;r'h'=0; ~h'_t=1; 
# x=1 (h=0, h'=1): z_t=0 ; z'_t=1 ; r_t=0 ; r'_t=0 ; ~h_t=1
#                  rh=0;r'h'=0; ~h'_t=1; 
# x=0 (h=1, h'=1): z_t=1 ; z'_t=1 ; r_t=0 ; r'_t=0 ; ~h_t=0.5
#                  rh=0;r'h'=0; ~h'_t=1; 
# x=1 (h=1, h'=1): z_t=0 ; z'_t=1 ; r_t=0 ; r'_t=0 ; ~h_t=1
#                  rh=0;r'h'=0; ~h'_t=1; 
#
# In summary:
# x=0 do nothing on hidden nodes: h_t=h_{t-1} ; h'_t=h'_{t-1}
# x=1 changes h to h=1
# x=1 with h=1 changes h' to h'=1.
#
# Conclusion:
# * h=1 iff there exists at least one x=1
# * h'=1 iff we have seen at least two x=1.
#
# Now going to y is simple:
# y = sigma(40h-40h'-20)
# If there is no 1:            y = sigma(-20)=0
# If there exists a unique 1:  y = sigma(40-20)=1
# If there is more than one 1: y = sigma(40-40-20)=0
# So it is working and can generalize to any size.
#
# Main conclusion: All is working and is fully interpretable as expected.

#############################################
# Marble the network to make it as an input #
#############################################
# Important: This will block any backpropagation.

import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def cross_entropy_for_Keras(y, y_hat):
    return(-K.mean((y*K.log(y_hat+1e-20) + (1-y)*K.log(1-y_hat+1e-20))))

def dirac(x):
    return (K.sign(x) + 1) / 2

def create_model_dirac():
    # This model CANNOT be trained.
    # It is to stick the model into 0/1 and check accuracy with it.
    # Alternative: Using ReLU instead of hard_sigmoid will also stick the model
    model=Sequential()    
    model = Sequential()
    dim_in = 1
    dim_out = 1
    model.add(GRU(input_shape=(None, dim_in), units=2, 
                  activation=dirac, recurrent_activation=dirac))
    model.add(Dense(activation=dirac, units=dim_out))
    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', 
                  metrics=['accuracy', cross_entropy_for_Keras])
    return(model)

model = create_model_dirac()
model_loaded = load_model('models/3_C_existsunique_gru_sigmoid.h5')
model.set_weights(model_loaded.get_weights())

outputs_all = outputs_EU_all

epochs = 1
cross_entropy_evol = []
repetitions = 1
for k in range(repetitions):
  for inputs, outputs in zip(inputs_all, outputs_all):
      model.fit(inputs, outputs, epochs = epochs, batch_size = 32)
      # Following only for cross entropy evolution
      cross_current = cross_entropy_whole(model, inputs_all, outputs_all)
      cross_entropy_evol.append(cross_current)
# Cross entropy is 0 for all 
# (loss given by Keras is the clipped one, not accurate here).
# For clipped loss, recall 
# 2_experimentations/1_linear_logistic_regressionis/logistic_regression.py

# Conclusion: The model works as expected, and we wish it can works for any
# size of inputs: It is exactly the function we wanted to learn.