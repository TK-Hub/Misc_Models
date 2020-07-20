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
x_valid = x_valid.reshape(10000, 784).astype('float32')

x_train /= 255
x_valid /= 255

# Done formatting the model inputs, now the labels y are converted from int to one-hot encodings. This is similar to the NLP
# one hot encoding. Instead of the number 7, we have an array of length 10 (nr of classes) with binary values. In the case
# of a seven, this array has a 1 on the 7th place.
n_classes = 10
y_train = keras.utils.to_categorical(y_train, n_classes)
y_valid = keras.utils.to_categorical(y_valid, n_classes)

# Designing the neural network
model = Sequential()
model.add(Dense(64, activation='sigmoid', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))
model.summary()

# Training the model
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01), metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=128, epochs=200,
          verbose=1,
          validation_data=(x_valid, y_valid))