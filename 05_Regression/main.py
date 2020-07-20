from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras.callbacks import TensorBoard

(x_train, y_train), (x_valid, y_valid) = boston_housing.load_data()

model = Sequential()

model.add(Dense(32, input_dim=13, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(1, activation='linear'))
model.summary()

model.compile(loss='mean_squared_error', optimizer='adam')

tensorboard = TensorBoard('logs/deep-net')

model.fit(x_train, y_train,
          batch_size=8, epochs=32, verbose=1,
          validation_data=(x_valid, y_valid),
          callbacks=[tensorboard])

print(model.predict(np.reshape(x_valid[42], [1, 13])))