import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, Activation
from keras.initializers import Zeros, RandomNormal
from keras.initializers import glorot_normal, glorot_uniform

# Simulate the MNIST input and set the nr of neurons to 256 (for the vis. of the weight initialization)
n_input = 784
n_dense = 256

# b=0 for the neurons of the hidden layers.
b_init = Zeros()

# Init the weights as standard normal distribution.
#w_init = RandomNormal(stddev=1.0)
#w_init = glorot_normal()
w_init = glorot_uniform()

# Design sample network to see the impact of the chosen weights. The activation function is added separately here.
model = Sequential()
model.add(Dense(n_dense,
                input_dim=n_input,
                kernel_initializer=w_init,
                bias_initializer=b_init))
model.add(Activation('sigmoid'))

# Random input for the net
x = np.random.random((1, n_input))

a = model.predict(x)

his = plt.hist(np.transpose(a))
plt.show()

