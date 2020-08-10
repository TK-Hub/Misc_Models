#==================================================================================================
#
#                       Natural Language Classification with a Recurrent Neural Network
#
#==================================================================================================

import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import SpatialDropout1D
from keras.layers import SimpleRNN
from keras.callbacks import ModelCheckpoint

import os
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import matplotlib.pyplot as plt

#==================================================================================================
#                       Hyperparameters

# Output directory name:
output_dir = 'model_output/rnn'

# Training parameters:
epochs = 16
batch_size = 128

# Vector-space embeddidng:
n_dim = 64
n_unique_words = 10000
max_review_length = 100
pad_type = trunc_type = 'pre'
drop_embed = 0.2

# RNN layer architecture
n_rnn = 256
drop_rnn = 0.2


#==================================================================================================
#                       Data Input (This is preprocessed by default in Keras)

(x_train, y_train), (x_valid, y_valid) = imdb.load_data(num_words=n_unique_words)
print(x_train[:6])

word_index = keras.datasets.imdb.get_word_index()
word_index = {k: (v+3) for k, v in word_index.items()}
word_index["PAD"] = 0
word_index["START"] = 1
word_index["UNK"] = 2
index_word = {v:k for k, v in word_index.items()}

print(' '.join(index_word[id] for id in x_train[0]))

(all_x_train, _), (all_x_valid, _) = imdb.load_data()

print(' '.join(index_word[id] for id in all_x_train[0]))

# Standardizing the Length of the Reviews
x_train = pad_sequences(x_train, maxlen=max_review_length,
                        padding=pad_type, truncating=trunc_type, value = 0)

x_valid = pad_sequences(x_valid, maxlen=max_review_length,
                        padding=pad_type, truncating=trunc_type, value = 0)

print(x_train[0:6])

#==================================================================================================
#                       Model Architecture

model = Sequential()
model.add(Embedding(n_unique_words, n_dim,
                    input_length=max_review_length))
model.add(SpatialDropout1D(drop_embed))
model.add(SimpleRNN(n_rnn, dropout=drop_rnn))
model.add(Dense(1, activation="sigmoid"))

model.summary()

model.compile(loss='binary_crossentropy', optimizer ='adam',
              metrics=['accuracy'])

#==================================================================================================
# Writing Checkpoints
modelcheckpoint = ModelCheckpoint(filepath=output_dir+"/weights.{epoch:02d}.hdf5")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model.fit(x_train, y_train,
          batch_size=batch_size, epochs=epochs, verbose=1,
          validation_data=(x_valid, y_valid),
          callbacks=[modelcheckpoint])

# This model doesn't really work well as SimpleRNNs only backpropagate for about 10 time steps -> LSTMs.
