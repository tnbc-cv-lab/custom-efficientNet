from string import whitespace
from efficientnet_keras_transfer_learning.efficientnet.layers import Swish, DropConnect
from efficientnet_keras_transfer_learning.efficientnet.model import ConvKernalInitializer
from efficientnet_keras_transfer_learning.efficientnet import EfficientNetB3 as Net
from efficientnet_keras_transfer_learning.efficientnet import center_crop_and_resize, preprocess_input
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import sklearn.metrics 
from supporters.supporters import bcolors
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from supporters.supporters import bcolors





def testnet(outputfolder, data_mode=1, height=150, width=150):
    print(bcolors.WARNING + bcolors.BOLD + bcolors.UNDERLINE + "Testing the network" + bcolors.ENDC)
    get_custom_objects().update({
        'ConvKernalInitializer': ConvKernalInitializer,
        'Swish': Swish,
        'DropConnect':DropConnect
    })

    dirs = ["test", "train", "validation"]
    y_true, y_pred = [], []
    image_size = (height, width, 3)

    if data_mode==1:
        model = load_model("./data/models/model_choice1.h5")
        print(bcolors.OKGREEN + "Model Loaded" + bcolors.ENDC)
        dirs = ["test", "train", "validation"]
        labels = ["cell", "white"]
        
    if data_mode==2:
        model = load_model("./data/models/model_choice2.h5")
        print(bcolors.OKGREEN + "Model Loaded" + bcolors.ENDC)
        labels = ["stroma", "stromatils", "tumor", "wspace"]

    if data_mode==3:
        model = load_model("./data/models/model_choice3.h5")
        print(bcolors.OKGREEN + "Model Loaded" + bcolors.ENDC)
        labels = ["c0", "c2", "c3"]
    
    for i in dirs:
        for j in labels:
            path="./data/images/modeldata/"+i+"/"+j
            for img in os.listdir(path):
                testimg = imread(os.path.join(path, img))
                image_size = model.input_shape[1]
                x = center_crop_and_resize(testimg, image_size=image_size)
                x = preprocess_input(x)
                x = np.expand_dims(x, 0)
                # make prediction and decode
                y = model.predict(x)[0]
                y=list(y)
                max_index = y.index(max(list(y)))
                y_pred.append(labels[max_index])
                y_true.append(j)

        mat = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None)
        print(mat)
        ax = sns.heatmap(mat, annot=True, cmap='Blues')

        ax.set_title('Confusion matrix on ' + i + ' set\n\n');
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');


        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)
        savepath= os.path.join(outputfolder, "confusion_matrix_"+i+".png")
        ## Display the visualization of the Confusion Matrix.
        plt.savefig(savepath)
        plt.clf()

        print(bcolors.OKGREEN + "Accuracy Report" + bcolors.ENDC)
        classification_report = sklearn.metrics.classification_report(y_true, y_pred, labels=labels)
        print(f"the classification report for the given dataset is \n{classification_report}")
        with open(os.path.join(outputfolder, "classification_report_"+i+".txt"), "w") as f:
            f.write(classification_report)


    print(bcolors.WARNING + bcolors.BOLD + bcolors.UNDERLINE + "Testing Complete" + bcolors.ENDC)

    

# testnet(outputfolder="./data/outputs/August10_172157", data_mode=1)