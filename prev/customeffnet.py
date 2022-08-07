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

# Initialising the required global variables 
batch_size = 10
width = 150
height = 150
epochs = 15
NUM_TRAIN = 121
NUM_TEST = 60
dropout_rate = 0.2
input_shape = (height, width, 3)


# Initialising the base model
conv_base = Net(weights='imagenet', include_top=False, input_shape=input_shape)


# sanity check and importing all the images
path = "/Users/mraoaakash/Desktop/TNBC/Models/EfficientNet/TumImg/data"
stroma = glob.glob(os.path.join(path,"Stroma", "*tif"))
stromatils = glob.glob(os.path.join(path,"StromaTils", "*tif"))
tumor = glob.glob(os.path.join(path,"Tumor", "*tif"))
tumortils = glob.glob(os.path.join(path,"TumorTils", "*tif"))
wspace = glob.glob(os.path.join(path,"WhiteSpace", "*tif"))

print("total Stroma images: {}\n\rtotal StromaTils images: {}\n\rtotal Tumor images: {}\n\rtotal TumorTils images: {}\n\rtotal WhiteSpace images: {}".format(len(stroma), len(stromatils), len(tumor), len(tumortils), len(wspace)))

path = "/Users/mraoaakash/Desktop/TNBC/Models/EfficientNet/TumImg/modeldata"
os.makedirs(path, exist_ok=True)

# Directories for our training,
# validation and test splits
train_dir = os.path.join(path, 'train')
os.makedirs(train_dir, exist_ok=True)
validation_dir = os.path.join(path, 'validation')
os.makedirs(validation_dir, exist_ok=True)
test_dir = os.path.join(path, 'test')
os.makedirs(test_dir, exist_ok=True)


# Directory with our training stroma pictures
train_stroma_dir = os.path.join(train_dir, 'stroma')
os.makedirs(train_stroma_dir, exist_ok=True)

# Directory with our training stromatils pictures
train_stromatils_dir = os.path.join(train_dir, 'stromatils')
os.makedirs(train_stromatils_dir, exist_ok=True)

# Directory with our training tumor pictures
train_tumor_dir = os.path.join(train_dir, 'tumor')
os.makedirs(train_tumor_dir, exist_ok=True)

# # Directory with our training tumortils pictures
# train_tumortils_dir = os.path.join(train_dir, 'tumortils')
# os.makedirs(train_tumortils_dir, exist_ok=True)

# Directory with our training wspace pictures
train_wspace_dir = os.path.join(train_dir, 'wspace')
os.makedirs(train_wspace_dir, exist_ok=True)


# Directory with our testing stroma pictures
test_stroma_dir = os.path.join(test_dir, 'stroma')
os.makedirs(test_stroma_dir, exist_ok=True)

# Directory with our testing stromatils pictures
test_stromatils_dir = os.path.join(test_dir, 'stromatils')
os.makedirs(test_stromatils_dir, exist_ok=True)

# Directory with our testing tumor pictures
test_tumor_dir = os.path.join(test_dir, 'tumor')
os.makedirs(test_tumor_dir, exist_ok=True)

# # Directory with our testing tumortils pictures
# test_tumortils_dir = os.path.join(test_dir, 'tumortils')
# os.makedirs(test_tumortils_dir, exist_ok=True)

# Directory with our testing wspace pictures
test_wspace_dir = os.path.join(test_dir, 'wspace')
os.makedirs(test_wspace_dir, exist_ok=True)



# Directory with our validation stroma pictures
validation_stroma_dir = os.path.join(validation_dir, 'stroma')
os.makedirs(validation_stroma_dir, exist_ok=True)

# Directory with our validation stromatils pictures
validation_stromatils_dir = os.path.join(validation_dir, 'stromatils')
os.makedirs(validation_stromatils_dir, exist_ok=True)

# Directory with our validation tumor pictures
validation_tumor_dir = os.path.join(validation_dir, 'tumor')
os.makedirs(validation_tumor_dir, exist_ok=True)

# # Directory with our validation tumortils pictures
# validation_tumortils_dir = os.path.join(validation_dir, 'tumortils')
# os.makedirs(validation_tumortils_dir, exist_ok=True)

# Directory with our validation wspace pictures
validation_wspace_dir = os.path.join(validation_dir, 'wspace')
os.makedirs(validation_wspace_dir, exist_ok=True)



# copying files for stroma samples to train/stroma
fnames = stroma[:NUM_TRAIN//2]
for fname in fnames:
    dst = os.path.join(train_stroma_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)

offset = NUM_TRAIN//2
# Copy next NUM_TEST //2 stroma images to validation_stroma_dir
fnames = stroma[offset:offset + NUM_TEST // 2]
for fname in fnames:
    dst = os.path.join(validation_stroma_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
offset = offset + NUM_TEST // 2
# Copy next NUM_TRAIN//2 stroma images to test_stroma_dir
fnames = stroma[offset:offset + NUM_TEST // 2]
for fname in fnames:
    dst = os.path.join(test_stroma_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)


# copying files for stromatils samples to train/stromatils
fnames = stromatils[:NUM_TRAIN//2]
for fname in fnames:
    dst = os.path.join(train_stromatils_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)

offset = NUM_TRAIN//2
# Copy next NUM_TEST //2 stromatils images to validation_stromatils_dir
fnames = stromatils[offset:offset + NUM_TEST // 2]
for fname in fnames:
    dst = os.path.join(validation_stromatils_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
offset = offset + NUM_TEST // 2
# Copy next NUM_TRAIN//2 stromatils images to test_stromatils_dir
fnames = stromatils[offset:offset + NUM_TEST // 2]
for fname in fnames:
    dst = os.path.join(test_stromatils_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)



# copying files for tumor samples to train/tumor
fnames = tumor[:NUM_TRAIN//2]
for fname in fnames:
    dst = os.path.join(train_tumor_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)

offset = NUM_TRAIN//2
# Copy next NUM_TEST //2 tumor images to validation_tumor_dir
fnames = tumor[offset:offset + NUM_TEST // 2]
for fname in fnames:
    dst = os.path.join(validation_tumor_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
offset = offset + NUM_TEST // 2
# Copy next NUM_TRAIN//2 tumor images to test_tumor_dir
fnames = tumor[offset:offset + NUM_TEST // 2]
for fname in fnames:
    dst = os.path.join(test_tumor_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)



# # copying files for tumortils samples to train/tumortils
# fnames = tumortils[:NUM_TRAIN//2]
# for fname in fnames:
#     dst = os.path.join(train_tumortils_dir, os.path.basename(fname))
#     shutil.copyfile(fname, dst)

# offset = NUM_TRAIN//2
# # Copy next NUM_TEST //2 tumortils images to validation_tumortils_dir
# fnames = tumortils[offset:offset + NUM_TEST // 2]
# for fname in fnames:
#     dst = os.path.join(validation_tumortils_dir, os.path.basename(fname))
#     shutil.copyfile(fname, dst)
# offset = offset + NUM_TEST // 2
# # Copy next NUM_TRAIN//2 tumortils images to test_tumortils_dir
# fnames = tumortils[offset:offset + NUM_TEST // 2]
# for fname in fnames:
#     dst = os.path.join(test_tumortils_dir, os.path.basename(fname))
#     shutil.copyfile(fname, dst)


# copying files for wspace samples to train/wspace
fnames = wspace[:NUM_TRAIN//2]
for fname in fnames:
    dst = os.path.join(train_wspace_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)

offset = NUM_TRAIN//2
# Copy next NUM_TEST //2 wspace images to validation_wspace_dir
fnames = wspace[offset:offset + NUM_TEST // 2]
for fname in fnames:
    dst = os.path.join(validation_wspace_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
offset = offset + NUM_TEST // 2
# Copy next NUM_TRAIN//2 wspace images to test_wspace_dir
fnames = wspace[offset:offset + NUM_TEST // 2]
for fname in fnames:
    dst = os.path.join(test_wspace_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)



print("Model Data Info:")

print('total training stroma images:', len(os.listdir(train_stroma_dir)))
print('total testing stroma images:', len(os.listdir(test_stroma_dir)))
print('total validation stroma images:', len(os.listdir(validation_stroma_dir)))

print('total training stromatils images:', len(os.listdir(train_stromatils_dir)))
print('total testing stromatils images:', len(os.listdir(test_stromatils_dir)))
print('total validation stromatils images:', len(os.listdir(validation_stromatils_dir)))

print('total training tumor images:', len(os.listdir(train_tumor_dir)))
print('total testing tumor images:', len(os.listdir(test_tumor_dir)))
print('total validation tumor images:', len(os.listdir(validation_tumor_dir)))

# print('total training tumortils images:', len(os.listdir(train_tumortils_dir)))
# print('total testing tumortils images:', len(os.listdir(test_tumortils_dir)))
# print('total validation tumortils images:', len(os.listdir(validation_tumortils_dir)))

print('total training wspace images:', len(os.listdir(train_wspace_dir)))
print('total testing wspace images:', len(os.listdir(test_wspace_dir)))
print('total validation wspace images:', len(os.listdir(validation_wspace_dir)))



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
model.add(layers.Dense(4, activation='softmax', name="fc_out"))
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