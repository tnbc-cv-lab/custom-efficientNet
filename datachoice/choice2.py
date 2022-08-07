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

def choice_2(NUM_TRAIN=121, NUM_TEST=60):
    print(bcolors.BOLD + bcolors.OKBLUE + "Data choice 2" + bcolors.ENDC)
    print(bcolors.OKBLUE + "Here, you have chosen to run the model on five classes \n - Stroma  \n - StromaTils  \n - Tumor  \n - TumorTils  \n - WhiteSpace")
    print("Please wait as the model runs below"+ bcolors.ENDC)
    path = "./data/images/choice_2"
    stroma = glob.glob(os.path.join(path,"Stroma", "*tif"))
    stromatils = glob.glob(os.path.join(path,"StromaTils", "*tif"))
    tumor = glob.glob(os.path.join(path,"Tumor", "*tif"))
    tumortils = glob.glob(os.path.join(path,"TumorTils", "*tif"))
    wspace = glob.glob(os.path.join(path,"WhiteSpace", "*tif"))

    print(bcolors.OKCYAN + "---------------------------------" + bcolors.ENDC)
    print(" - total Stroma images: {}\n\r - total StromaTils images: {}\n\r - total Tumor images: {}\n\r - total TumorTils images: {}\n\r - total WhiteSpace images: {}".format(len(stroma), len(stromatils), len(tumor), len(tumortils), len(wspace)))
    print(bcolors.OKCYAN + "---------------------------------" + bcolors.ENDC)

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
    print(NUM_TRAIN//2)
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


    print(bcolors.OKCYAN + "---------------------------------" + bcolors.ENDC)
    print(bcolors.OKCYAN + "---------------------------------" + bcolors.ENDC)
    print("Model Data Info:")
    print(' - total training stroma images:', len(os.listdir(train_stroma_dir)))
    print(' - total testing stroma images:', len(os.listdir(test_stroma_dir)))
    print(' - total validation stroma images:', len(os.listdir(validation_stroma_dir)))
    print(bcolors.OKCYAN + "--------" + bcolors.ENDC)

    print(' - total training stromatils images:', len(os.listdir(train_stromatils_dir)))
    print(' - total testing stromatils images:', len(os.listdir(test_stromatils_dir)))
    print(' - total validation stromatils images:', len(os.listdir(validation_stromatils_dir)))
    print(bcolors.OKCYAN + "--------" + bcolors.ENDC)

    print(' - total training tumor images:', len(os.listdir(train_tumor_dir)))
    print(' - total testing tumor images:', len(os.listdir(test_tumor_dir)))
    print(' - total validation tumor images:', len(os.listdir(validation_tumor_dir)))
    print(bcolors.OKCYAN + "--------" + bcolors.ENDC)

    # print('total training tumortils images:', len(os.listdir(train_tumortils_dir)))
    # print('total testing tumortils images:', len(os.listdir(test_tumortils_dir)))
    # print('total validation tumortils images:', len(os.listdir(validation_tumortils_dir)))

    print(' - total training wspace images:', len(os.listdir(train_wspace_dir)))
    print(' - total testing wspace images:', len(os.listdir(test_wspace_dir)))
    print(' - total validation wspace images:', len(os.listdir(validation_wspace_dir)))
    print(bcolors.OKCYAN + "---------------------------------" + bcolors.ENDC)
    print(bcolors.OKCYAN + "---------------------------------" + bcolors.ENDC)


    return (train_dir, validation_dir, test_dir, 20, 4)