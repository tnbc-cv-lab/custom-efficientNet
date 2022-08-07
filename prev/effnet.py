from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from IPython.display import Image
from efficientnet_keras_transfer_learning.efficientnet import EfficientNetB0 as Net
from efficientnet_keras_transfer_learning.efficientnet import center_crop_and_resize, preprocess_input


batch_size = 48

width = 150
height = 150
epochs = 20
NUM_TRAIN = 1136
NUM_TEST = 576
dropout_rate = 0.2
input_shape = (height, width, 3)

conv_base = Net(weights='imagenet', include_top=False, input_shape=input_shape)

print("I am running?")
path = "/Users/mraoaakash/Desktop/TNBC/Models/EfficientNet/TumorImages"
wsimgs = glob.glob(os.path.join(path,"whiteSpace", "*tif"))
csimgs = glob.glob(os.path.join(path,"cellSpace", "*tif"))

print("total whiteSpace images: {}\n\rtotal cellSpace images: {}".format(len(wsimgs), len(csimgs)))


# The directory where we will
# store our smaller dataset
os.makedirs(path, exist_ok=True)

# Directories for our training,
# validation and test splits
train_dir = os.path.join(path, 'train')
os.makedirs(train_dir, exist_ok=True)
validation_dir = os.path.join(path, 'validation')
os.makedirs(validation_dir, exist_ok=True)
test_dir = os.path.join(path, 'test')
os.makedirs(test_dir, exist_ok=True)

# Directory with our training cat pictures
train_white_dir = os.path.join(train_dir, 'white')
os.makedirs(train_white_dir, exist_ok=True)

# Directory with our training dog pictures
train_cell_dir = os.path.join(train_dir, 'cell')
os.makedirs(train_cell_dir, exist_ok=True)

# Directory with our validation cat pictures
validation_white_dir = os.path.join(validation_dir, 'white')
os.makedirs(validation_white_dir, exist_ok=True)

# Directory with our validation dog pictures
validation_cell_dir = os.path.join(validation_dir, 'cell')
os.makedirs(validation_cell_dir, exist_ok=True)

# Directory with our validation cat pictures
test_white_dir = os.path.join(test_dir, 'white')
os.makedirs(test_white_dir, exist_ok=True)

# Directory with our validation dog pictures
test_cell_dir = os.path.join(test_dir, 'cell')
os.makedirs(test_cell_dir, exist_ok=True)

fnames = wsimgs[:NUM_TRAIN//2]
for fname in fnames:
    dst = os.path.join(train_white_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)

offset = NUM_TRAIN//2
# Copy next NUM_TEST //2 cat images to validation_white_dir
fnames = wsimgs[offset:offset + NUM_TEST // 2]
for fname in fnames:
    dst = os.path.join(validation_white_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
offset = offset + NUM_TEST // 2
# Copy next NUM_TRAIN//2 cat images to test_white_dir
fnames = wsimgs[offset:offset + NUM_TEST // 2]
for fname in fnames:
    dst = os.path.join(test_white_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)



# Copy first NUM_TRAIN//2 dog images to train_dogs_dir
fnames = csimgs[:NUM_TRAIN//2]
for fname in fnames:
    dst = os.path.join(train_cell_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)

offset = NUM_TRAIN//2
# Copy next NUM_TEST // 2 dog images to validation_cell_dir
fnames = csimgs[offset:offset + NUM_TEST // 2]
for fname in fnames:
    dst = os.path.join(validation_cell_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
offset = offset + NUM_TEST // 2

# Copy next NUM_TEST // 2 dog images to test_cell_dir
fnames = csimgs[offset:offset + NUM_TEST // 2]
for fname in fnames:
    dst = os.path.join(test_cell_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)



print('total training cat images:', len(os.listdir(train_white_dir)))
print('total training dog images:', len(os.listdir(train_cell_dir)))
print('total validation cat images:', len(os.listdir(validation_white_dir)))
print('total validation dog images:', len(os.listdir(validation_cell_dir)))
print('total test cat images:', len(os.listdir(test_white_dir)))
print('total test dog images:', len(os.listdir(test_cell_dir)))



train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to target height and width.
        target_size=(height, width),
        batch_size=batch_size,
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical')




model = models.Sequential()
model.add(conv_base)
model.add(layers.GlobalMaxPooling2D(name="gap"))
# model.add(layers.Flatten(name="flatten"))
if dropout_rate > 0:
    model.add(layers.Dropout(dropout_rate, name="dropout_out"))
# model.add(layers.Dense(256, activation='relu', name="fc1"))
model.add(layers.Dense(2, activation='softmax', name="fc_out"))
model.summary()

print('This is the number of trainable layers '
      'before freezing the conv base:', len(model.trainable_weights))

conv_base.trainable = False

print('This is the number of trainable layers '
      'after freezing the conv base:', len(model.trainable_weights))


devices = tf.config.experimental.list_physical_devices('CPU')

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=2e-5),
              metrics=['acc'])


history = model.fit_generator(
      train_generator,
      steps_per_epoch= NUM_TRAIN //batch_size,
      epochs=epochs,
      validation_data=validation_generator,
      validation_steps= NUM_TEST //batch_size,
      verbose=1,
      use_multiprocessing=False,)



acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_x = range(len(acc))

plt.plot(epochs_x, acc, 'bo', label='Training acc')
plt.plot(epochs_x, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs_x, loss, 'bo', label='Training loss')
plt.plot(epochs_x, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()