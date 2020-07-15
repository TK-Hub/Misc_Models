import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot as plt

(x_train, y_train), (x_valid, y_valid) = mnist.load_data()

plt.figure(figsize=(5,5))
for k in range(12):
    plt.subplot(3,4,k+1)
    plt.imshow(x_train[k], cmap='Greys')
    plt.axis('off')
plt.tight_layout()
plt.show()

# Reshaping the data to a one dimensional array of 784 numbers to fit the input layer geometry.
x_train = x_train.reshape(60000, 784).astype('float32')
x_valid = x_valid.reshape(60000, 784).astype('float32')

x_train /= 255
x_valid /= 255

# Done formatting the model inputs, now the labels y are converted from int to one-hot encodings.
n_classes = 10
y_train = keras.utils
