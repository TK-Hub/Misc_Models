#==================================================================================================
#
#                       Natural Language Classification with Familiar Networks
#
#==================================================================================================

import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint

import os
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import matplotlib.pyplot as plt

#==================================================================================================
#                       Hyperparameters

# Output directory name:
output_dir = 'model_output/dense'

# Training parameters:
epochs = 4
batch_size = 128

# Vector-space embeddidng:
n_dim = 64
n_unique_words = 5000
n_words_to_skip = 50
max_review_length = 100
pad_type = trunc_type = 'pre'

# Network architecture
n_dense = 64
dropout = 0.5

#==================================================================================================
#                       Data Input (This is preprocessed by default in Keras)

(x_train, y_train), (x_valid, y_valid) = imdb.load_data(num_words=n_unique_words, skip_top=n_words_to_skip)
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
#                       Dense Model Architecture

model = Sequential()
model.add(Embedding(n_unique_words, n_dim,
                    input_length=max_review_length))
model.add(Flatten())
model.add(Dense(n_dense, activation='relu'))
model.add(Dropout(dropout))
#model.add(Dense(n_dense, activation='relu'))
#model.add(Dropout(dropout))

# Single output neuron since this is a binary classification task.
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer ='adam',
              metrics=['accuracy'])

# Writing Checkpoints
"""modelcheckpoint = ModelCheckpoint(filepath=output_dir+"/weights.{epoch:02d}.hdf5")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model.fit(x_train, y_train,
          batch_size=batch_size, epochs=epochs, verbose=1,
          validation_data=(x_valid, y_valid),
          callbacks=[modelcheckpoint])"""


#==================================================================================================
#                       Reload our Checkpoints and plot the results

model.load_weights(output_dir + "/weights.02.hdf5")
y_hat = model.predict_proba(x_valid)

print(' '.join(index_word[id] for id in all_x_valid[0]))

plt.hist(y_hat)
plt.axvline(x=0.5, color='orange')

#plt.show()


#==================================================================================================
#                       Calculate ROC AUC metrics
pct_auc = roc_auc_score(y_valid, y_hat)*100.0
print("{:0.2f}".format(pct_auc))


#==================================================================================================
#                       DataFrame of all reviews + 10 Examples with a high and low score
float_y_hat = []
for y in y_hat:
    float_y_hat.append(y[0])
ydf = pd.DataFrame(list(zip(float_y_hat, y_valid)),
                   columns=['y_hat', 'y'])
print(ydf)
print(ydf[(ydf.y == 0) & (ydf.y_hat > 0.9)].head(10))
print(ydf[(ydf.y == 1) & (ydf.y_hat < 0.1)].head(10))
