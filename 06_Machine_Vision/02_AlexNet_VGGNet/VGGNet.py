#==================================================================================================
#
#                       VGGNet in Keras
#
#==================================================================================================

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, BatchNormalization
from keras.optimizers import SGD

(x_train, y_train), (x_valid, y_valid) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')
x_valid = x_valid.reshape(10000, 28, 28, 1).astype('float32')

# Convert labels to one-hot encodings
n_classes = 10
y_train = keras.utils.to_categorical(y_train, n_classes)
y_valid = keras.utils.to_categorical(y_valid, n_classes)

# Define Model
model = Sequential()

# First convolutional layer. The 96 refers to the number of filters. This model is defined for a larger version of MNIST
# (224x224, RGB) and won't work for the standard Keras dataset. Reference architecture:
model.add(Conv2D(64, 3, activation='relu',
                 input_shape=(224, 224, 3)))
model.add(Conv2D(64, 3, activation='relu')
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization()))

# Second block of convolutional layers
model.add(Conv2D(128, 3, activation='relu'))
model.add(Conv2D(128, 3, activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())

# Third block of conv. layer, incl. pooling and batch normalization
model.add(Conv2D(256, 3, activation='relu'))
model.add(Conv2D(256, 3, activation='relu'))
model.add(Conv2D(256, 3, activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())

# Fourth block of conv. layer, incl. pooling and batch normalization
model.add(Conv2D(512, 3, activation='relu'))
model.add(Conv2D(512, 3, activation='relu'))
model.add(Conv2D(512, 3, activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())

# Fifth block of conv. layer, incl. pooling and batch normalization
model.add(Conv2D(512, 3, activation='relu'))
model.add(Conv2D(512, 3, activation='relu'))
model.add(Conv2D(512, 3, activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())

# Dense hidden layers, with dropout
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

# Output layer:
model.add(Dense(17, activation='softmax'))

model.summary()

#model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])

#model.fit(x_train, y_train,
#          batch_size=128, epochs=10,
#          verbose=1,
#          validation_data=(x_valid, y_valid))


