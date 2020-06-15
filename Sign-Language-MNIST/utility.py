#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    # read data
    df_test = pd.read_csv('Data/sign_mnist_test.csv')
    df_train = pd.read_csv('Data/sign_mnist_train.csv')

    #shuffle data
    df_train = df_train.sample(frac=1).reset_index(drop=True)

    y_train = df_train['label'].to_numpy().ravel()
    X_train = df_train.iloc[:, 1:].to_numpy() / 255

    y_test = df_test['label'].to_numpy().ravel()
    X_test = df_test.iloc[:, 1:].to_numpy() / 255

    return X_train, y_train, X_test, y_test


def signsLabels():
    signs_dic = { 0:'A', 
    1:'B', 
    2:'C', 
    3:'D', 
    4:'E', 
    5:'F', 
    6:'G', 
    7:'H', 
    8:'I', 
    #9:'J', 
    10:'K', 
    11:'L', 
    12:'M', 
    13:'N', 
    14:'O', 
    15:'P', 
    16:'Q', 
    17:'R', 
    18:'S', 
    19:'T', 
    20:'U', 
    21:'V', 
    22:'W', 
    23:'X', 
    24:'Y',  
    #25:'Z'
                }
    return signs_dic

def plot_accuracy(history , name):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    axs[0].plot(history.history['accuracy'], label='Train Accuracy')
    axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy')

    axs[0].legend()
    axs[0].set_title('Accuracy')

    axs[1].plot(history.history['loss'], label='Train Loss')
    axs[1].plot(history.history['val_loss'], label='Validation Loss')
    axs[1].legend()
    axs[1].set_title('Loss')
    plt.savefig('output/{0}.png'.format(name) , format="png")
    plt.show()