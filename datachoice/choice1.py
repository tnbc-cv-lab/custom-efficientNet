import os
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from IPython.display import Image

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def choice_1(NUM_TRAIN=1136, NUM_TEST=576):
    print(bcolors.BOLD + bcolors.OKBLUE + "Data choice 1" + bcolors.ENDC)
    print(bcolors.OKBLUE + "Here, you have chosen to run the model on two classes \n - WhiteSpace  \n - CellSpace"+ bcolors.ENDC)
    print(bcolors.UNDERLINE + bcolors.BOLD + "Please wait as the model runs"+ bcolors.ENDC)
    path = "./data/images/choice_1"
    wsimgs = glob.glob(os.path.join(path,"whiteSpace", "*tif"))
    csimgs = glob.glob(os.path.join(path,"cellSpace", "*tif"))

    print(bcolors.OKCYAN + "---------------------------------" + bcolors.ENDC)
    print(" - total whiteSpace images: {}\n\rtotal cellSpace images: {}".format(len(wsimgs), len(csimgs)))


    # The directory where we will
    # store our smaller dataset
    path = "./data/images/modeldata"
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


    print(bcolors.OKCYAN + "---------------------------------" + bcolors.ENDC)
    print("Dataset information:")
    print(bcolors.OKCYAN + "---------------------------------" + bcolors.ENDC)
    print(' - total training cat images:', len(os.listdir(train_white_dir)))
    print(' - total validation cat images:', len(os.listdir(validation_white_dir)))
    print(' - total test cat images:', len(os.listdir(test_white_dir)))
    print(bcolors.OKCYAN + "--------" + bcolors.ENDC)
    print(' - total training dog images:', len(os.listdir(train_cell_dir)))
    print(' - total validation dog images:', len(os.listdir(validation_cell_dir)))
    print(' - total test dog images:', len(os.listdir(test_cell_dir)))
    print(bcolors.OKCYAN + "---------------------------------" + bcolors.ENDC)
    print(bcolors.OKCYAN + "---------------------------------" + bcolors.ENDC)

    return (train_dir, validation_dir, test_dir, 48, 2)

    