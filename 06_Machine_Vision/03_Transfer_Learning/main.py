from keras.applications.vgg19 import VGG19
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator

# Load the pre-trained VGG19 model
vgg19 = VGG19(include_top=False,
              weights='imagenet',
              input_shape=(224, 224, 3),
              pooling=None)

# Freeze all layers in the base VGGNet19 model:
for layer in vgg19.layers:
    layer.trainable = False

# Adding our model
model = Sequential()
model.add(vgg19)

model.add(Flatten(name='flattened'))
model.add(Dropout(0.5, name='dropout'))
model.add(Dense(2, activation='softmax', name='predictions'))

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Preprocess data
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    data_format='channels_last',
    rotation_range=30,
    horizontal_flip=True,
    fill_mode='reflect'
)

valid_datagen = ImageDataGenerator(
    rescale=1.0/255,
    data_format='channels_last'
)

batch_size = 32

train_generator = train_datagen.flow_from_directory(
    directory='./hot-dog-not-hot-dog/train',
    target_size=(224, 224),
    classes = ['hot_dog', 'not_hot_dog'],
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle = True,
    seed=42
)

valid_generator = train_datagen.flow_from_directory(
    directory='./hot-dog-not-hot-dog/test',
    target_size=(224, 224),
    classes = ['hot_dog', 'not_hot_dog'],
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle = True,
    seed=42
)

model.fit_generator(train_generator, steps_per_epoch=15,
                    epochs=16, validation_data=valid_generator,
                    validation_steps=15)