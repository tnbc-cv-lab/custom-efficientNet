from efficientnet_keras_transfer_learning.efficientnet.layers import Swish, DropConnect
from efficientnet_keras_transfer_learning.efficientnet.model import ConvKernalInitializer
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from supporters.supporters import bcolors
import os
import numpy as np




def testnet(data_mode=1, height=150, width=150):
    get_custom_objects().update({
        'ConvKernalInitializer': ConvKernalInitializer,
        'Swish': Swish,
        'DropConnect':DropConnect
    })
    if data_mode==1:
        model = load_model("./data/models/model_choice1.h5")
        print(bcolors.OKGREEN + "Model Loaded" + bcolors.ENDC)
        path="./data/images/modeldata/test"
        cell = os.path.join(path, "cell")
        white = os.path.join(path, "white")
        predictions = []
        for img in os.listdir(cell):
            testimg = image.load_img(os.path.join(cell, img), target_size=(height, width))
            # Convert it to a Numpy array with target shape.
            x = image.img_to_array(testimg)
            # Reshape
            x = x.reshape((1,) + x.shape)
            result = model.predict([x])[0][0]
            if result > 0.5:
                animal = "cat"
            else:
                animal = "dog"
                result = 1 - result
            predictions.append(result)
        predictions = np.array(predictions)
        print(predictions)
        print(np.mean(predictions[:][1]))




    if data_mode==2:
        model = load_model("./data/models/model_choice2.h5")
        print(bcolors.OKGREEN + "Model Loaded" + bcolors.ENDC)
    if data_mode==3:
        model = load_model("./data/models/model_choice3.h5")
        print(bcolors.OKGREEN + "Model Loaded" + bcolors.ENDC)
    

    

testnet(data_mode=1)