import matplotlib.pyplot as plt
import os
import numpy as np

def plottersaver(acc, val_acc, loss, val_loss, epochs_x,data_mode, outputs, frozen):

    # Plot training & validation accuracy values
    os.makedirs(outputs, exist_ok=True)
    plt.plot(epochs_x, acc, 'bo', label='Training acc')
    plt.plot(epochs_x, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    accname = 'acc_frozenConv.png' if frozen==True else 'acc_trainableConv.png'
    plt.savefig(os.path.join(outputs, accname))

    plt.figure()


    # Plot training & validation loss values
    plt.plot(epochs_x, loss, 'bo', label='Training loss')
    plt.plot(epochs_x, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    lossname = 'loss_frozenConv.png' if frozen==True else 'loss_trainableConv.png'
    plt.savefig(os.path.join(outputs, lossname))
    txtname = 'details_frozenConv.txt' if frozen==True else 'details_trainableConv.txt'

    # Save relevant details to a txt file for future reference 
    with open(os.path.join(outputs,txtname), 'w+') as f:
        f.write("Accuracy: " + str(acc) + "\n")
        f.write("Validation Accuracy: " + str(val_acc) + "\n")
        f.write("Loss: " + str(loss) + "\n")
        f.write("Validation Loss: " + str(val_loss) + "\n")
        f.write("Data Mode: " + str(data_mode) + "\n")
        f.write("Epochs: " + str(epochs_x) + "\n")