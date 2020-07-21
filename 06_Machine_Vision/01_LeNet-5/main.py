import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten
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

# First convolutional layer. The 32 refers to the number of filters.
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

# Second conv. layer, incl. pooling and dropout
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())

# Dense hidden layer, with dropout
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output layer:
model.add(Dense(n_classes, activation='softmax'))

model.summary()

#model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])

#model.fit(x_train, y_train,
#          batch_size=128, epochs=10,
#          verbose=1,
#          validation_data=(x_valid, y_valid))


