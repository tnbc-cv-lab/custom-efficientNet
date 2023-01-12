from time import time
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
import shutil
from skimage.io import imread
from datetime import datetime

# Importing the network and related pipelines
from efficientnet import EfficientNetB0 as Net
from efficientnet import center_crop_and_resize, preprocess_input

# Importing the required supporter files
from supporters.supporters import bcolors, start, stop
from supporters.plotter import plottersaver

# Importing the test script
# from mastertestnet import testnet

# Importing the available data choices
from datachoice.choice1 import choice_1
from datachoice.choice2 import choice_2
from datachoice.choice3 import choice_3
 


def run_model(height=150, width = 150, epochs=20, NUM_TRAIN=1136, NUM_TEST=576, dropout_rate=0.2, data_mode = 2, layertrainable = True, flatten=False, dense=False):
    # sets the time the function was called and
    # creates a folder to store the results
    now = datetime.now()
    outputs = os.path.join("./data/outputs",now.strftime("%B%d_%H%M"))
    os.makedirs(outputs, exist_ok=True)
    # start(filename = os.path.join(outputs, "log.out"))



    # sets the input shape of the data based on the data mode
    input_shape = (height, width, 3)


    # initialises the network with default imagenet weights
    conv_base = Net(weights='imagenet', include_top=False, input_shape=input_shape)

    # Calls the relevant data choice function
    if data_mode == 1:
        data_out = choice_1()
    elif data_mode == 2:
        data_out = choice_2()
        NUM_TRAIN=121
        NUM_TEST=60
    elif data_mode == 3:
        data_out = choice_3()
        NUM_TRAIN=460
        NUM_TEST=230
    else:
        data_mode = choice_1()
    train_dir = data_out[0]
    validation_dir = data_out[1]
    test_dir = data_out[2]
    batch_size= data_out[3]
    classes=data_out[4]


    # Creates the data generators for the training phase
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



    # adds various custom layers to the network to conduct a form of transfer learning
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.GlobalMaxPooling2D(name="gap"))
    if flatten ==True:
        model.add(layers.Flatten(name="flatten"))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate, name="dropout_out"))
    if dense==True:
        model.add(layers.Dense(256, activation='relu', name="fc1"))
    model.add(layers.Dense(classes, activation='softmax', name="fc_out"))
    print(bcolors.OKGREEN)
    model.summary()
    print(bcolors.ENDC)

    print(bcolors.WARNING + 'This is the number of trainable layers '
        'before freezing the conv base:', len(model.trainable_weights))

    conv_base.trainable = False

    print(f"This is the number of trainable layers after freezing the conv base: {len(model.trainable_weights)} " + bcolors.ENDC)


    devices = tf.config.experimental.list_physical_devices('CPU')



    model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.RMSprop(learning_rate=2e-5),
                metrics=['acc'])


    history = model.fit(
        train_generator,
        steps_per_epoch= NUM_TRAIN //batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps= NUM_TEST //batch_size,
        verbose=1,
        use_multiprocessing=False,)
    print(history.history)
    plottersaver(history.history['acc'], history.history['val_acc'], history.history['loss'], history.history['val_loss'], range(len(history.history['acc'])), data_mode, outputs, frozen=True)

    if layertrainable ==True:
        conv_base.trainable = True

        set_trainable = False
        for layer in conv_base.layers:
            if layer.name == 'multiply_16':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False
        
        model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.RMSprop(lr=2e-5),
                metrics=['acc'])

        history = model.fit(
            train_generator,
            steps_per_epoch= NUM_TRAIN //batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps= NUM_TEST //batch_size,
            verbose=1,
            use_multiprocessing=False)
        plottersaver(history.history['acc'], history.history['val_acc'], history.history['loss'], history.history['val_loss'], range(len(history.history['acc'])), data_mode, outputs, frozen=False)
    model.save(os.path.join("./data/models", "model_choice"+str(data_mode)+".h5"))

    # testnet(data_mode=1, height=150, width=150)

    for dir in os.listdir("./data/images/modeldata"):
        shutil.rmtree(os.path.join("./data/images/modeldata",dir))

    # stop()
    return (history.history, outputs)

def main(data_mode=1, layertrainable=True, flatten=False, dense=False):
    history = run_model(data_mode=data_mode, layertrainable=layertrainable, flatten=flatten, dense=dense)
    outputs = history[1]
    history = history[0]
    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs_x = range(len(acc))

# print(bcolors.FAIL + "Setting: \n\r - Data_mode = 1 \n\r - Layersrainable = False \n\r - flatten layer = False \n\r - dense layer = False" + bcolors.ENDC)
# main(1, False, False, False)

print(bcolors.FAIL + "Setting: \n\r - Data_mode = 1 \n\r - Layersrainable = True \n\r - flatten layer = False \n\r - dense layer = False" + bcolors.ENDC)
main(1, True, False, False)

# print(bcolors.FAIL + "Setting: \n\r - Data_mode = 1 \n\r - Layersrainable = False \n\r - flatten layer = True \n\r - dense layer = False" + bcolors.ENDC)
# main(1, False, True, False)

# print(bcolors.FAIL + "Setting: \n\r - Data_mode = 1 \n\r - Layersrainable = True \n\r - flatten layer = True \n\r - dense layer = False" + bcolors.ENDC)
# main(1, True, True, False)

# print(bcolors.FAIL + "Setting: \n\r - Data_mode = 1 \n\r - Layersrainable = False \n\r - flatten layer = False \n\r - dense layer = True" + bcolors.ENDC)
# main(1, False, False, True)

# print(bcolors.FAIL + "Setting: \n\r - Data_mode = 1 \n\r - Layersrainable = True \n\r - flatten layer = False \n\r - dense layer = True" + bcolors.ENDC)
# main(1, True, False, True)

# print(bcolors.FAIL + "Setting: \n\r - Data_mode = 1 \n\r - Layersrainable = False \n\r - flatten layer = True \n\r - dense layer = True" + bcolors.ENDC)
# main(1, False, True, True)

# print(bcolors.FAIL + "Setting: \n\r - Data_mode = 1 \n\r - Layersrainable = True \n\r - flatten layer = True \n\r - dense layer = True" + bcolors.ENDC)
# main(1, True, True, True)

