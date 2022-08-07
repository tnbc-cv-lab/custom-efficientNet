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

def choice_3(NUM_TRAIN=460, NUM_TEST=230):
    print(bcolors.BOLD + bcolors.OKBLUE + "Data choice 3" + bcolors.ENDC)
    print(bcolors.OKBLUE + "Here, you have chosen to run the model on three of the four clusters generated earlier \n - c0 \n - c2 \n - c3"+ bcolors.ENDC)
    print(bcolors.UNDERLINE + bcolors.BOLD + "Please wait as the model runs"+ bcolors.ENDC)
    path = "./data/images/choice_3"
    c0 = glob.glob(os.path.join(path,"c0", "*tif"))
    c2 = glob.glob(os.path.join(path,"c2", "*tif"))
    c3 = glob.glob(os.path.join(path,"c3", "*tif"))

    print(bcolors.OKCYAN + "---------------------------------" + bcolors.ENDC)
    print(" - total c0 images: {}\n\rtotal c2Space images: {}".format(len(c0), len(c2)))


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

    # Directory with our training c0 pictures
    train_c0_dir = os.path.join(train_dir, 'c0')
    os.makedirs(train_c0_dir, exist_ok=True)
    # Directory with our training c2 pictures
    train_c2_dir = os.path.join(train_dir, 'c2')
    os.makedirs(train_c2_dir, exist_ok=True)
    # Directory with our training c2 pictures
    train_c3_dir = os.path.join(train_dir, 'c3')
    os.makedirs(train_c3_dir, exist_ok=True)


    # Directory with our validation c0 pictures
    validation_c0_dir = os.path.join(validation_dir, 'c0')
    os.makedirs(validation_c0_dir, exist_ok=True)
    # Directory with our validation c2 pictures
    validation_c2_dir = os.path.join(validation_dir, 'c2')
    os.makedirs(validation_c2_dir, exist_ok=True)
    # Directory with our validation c2 pictures
    validation_c3_dir = os.path.join(validation_dir, 'c3')
    os.makedirs(validation_c3_dir, exist_ok=True)


    # Directory with our validation c0 pictures
    test_c0_dir = os.path.join(test_dir, 'c0')
    os.makedirs(test_c0_dir, exist_ok=True)
    # Directory with our validation c2 pictures
    test_c2_dir = os.path.join(test_dir, 'c2')
    os.makedirs(test_c2_dir, exist_ok=True)
    # Directory with our validation c2 pictures
    test_c3_dir = os.path.join(test_dir, 'c3')
    os.makedirs(test_c3_dir, exist_ok=True)



    fnames = c0[:NUM_TRAIN//2]
    for fname in fnames:
        dst = os.path.join(train_c0_dir, os.path.basename(fname))
        shutil.copyfile(fname, dst)

    offset = NUM_TRAIN//2
    # Copy next NUM_TEST //2 c0 images to validation_c0_dir
    fnames = c0[offset:offset + NUM_TEST // 2]
    for fname in fnames:
        dst = os.path.join(validation_c0_dir, os.path.basename(fname))
        shutil.copyfile(fname, dst)
    offset = offset + NUM_TEST // 2
    # Copy next NUM_TRAIN//2 c0 images to test_c0_dir
    fnames = c0[offset:offset + NUM_TEST // 2]
    for fname in fnames:
        dst = os.path.join(test_c0_dir, os.path.basename(fname))
        shutil.copyfile(fname, dst)



    # Copy first NUM_TRAIN//2 c2 images to train_c2s_dir
    fnames = c2[:NUM_TRAIN//2]
    for fname in fnames:
        dst = os.path.join(train_c2_dir, os.path.basename(fname))
        shutil.copyfile(fname, dst)

    offset = NUM_TRAIN//2
    # Copy next NUM_TEST // 2 c2 images to validation_c2_dir
    fnames = c2[offset:offset + NUM_TEST // 2]
    for fname in fnames:
        dst = os.path.join(validation_c2_dir, os.path.basename(fname))
        shutil.copyfile(fname, dst)
    offset = offset + NUM_TEST // 2

    # Copy next NUM_TEST // 2 c2 images to test_c2_dir
    fnames = c2[offset:offset + NUM_TEST // 2]
    for fname in fnames:
        dst = os.path.join(test_c2_dir, os.path.basename(fname))
        shutil.copyfile(fname, dst)

    # Copy first NUM_TRAIN//2 c3 images to train_c3s_dir
    fnames = c3[:NUM_TRAIN//2]
    for fname in fnames:
        dst = os.path.join(train_c3_dir, os.path.basename(fname))
        shutil.copyfile(fname, dst)

    offset = NUM_TRAIN//2
    # Copy next NUM_TEST // 2 c3 images to validation_c3_dir
    fnames = c3[offset:offset + NUM_TEST // 2]
    for fname in fnames:
        dst = os.path.join(validation_c3_dir, os.path.basename(fname))
        shutil.copyfile(fname, dst)
    offset = offset + NUM_TEST // 2

    # Copy next NUM_TEST // 2 c3 images to test_c3_dir
    fnames = c3[offset:offset + NUM_TEST // 2]
    for fname in fnames:
        dst = os.path.join(test_c3_dir, os.path.basename(fname))
        shutil.copyfile(fname, dst)


    print(bcolors.OKCYAN + "---------------------------------" + bcolors.ENDC)
    print("Dataset information:")
    print(bcolors.OKCYAN + "---------------------------------" + bcolors.ENDC)
    print(' - total training c0 images:', len(os.listdir(train_c0_dir)))
    print(' - total validation c0 images:', len(os.listdir(validation_c0_dir)))
    print(' - total test c0 images:', len(os.listdir(test_c0_dir)))
    print(bcolors.OKCYAN + "--------" + bcolors.ENDC)
    print(' - total training c2 images:', len(os.listdir(train_c2_dir)))
    print(' - total validation c2 images:', len(os.listdir(validation_c2_dir)))
    print(' - total test c2 images:', len(os.listdir(test_c2_dir)))
    print(bcolors.OKCYAN + "--------" + bcolors.ENDC)
    print(' - total training c3 images:', len(os.listdir(train_c3_dir)))
    print(' - total validation c3 images:', len(os.listdir(validation_c3_dir)))
    print(' - total test c3 images:', len(os.listdir(test_c3_dir)))
    print(bcolors.OKCYAN + "---------------------------------" + bcolors.ENDC)
    print(bcolors.OKCYAN + "---------------------------------" + bcolors.ENDC)

    return (train_dir, validation_dir, test_dir, 26, 3)