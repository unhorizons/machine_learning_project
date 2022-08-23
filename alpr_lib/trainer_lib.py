import pandas as pd
import numpy as np

from matplotlib import image, pyplot as plt
import seaborn as sns
import cv2
import os
import glob

from random import sample, randint

from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.resnet_v2 import ResNet50V2

import keras.optimizers as  optimizers

import easyocr

from lxml import etree

IMAGE_SIZE = 200

class AlprTrainer:
    def __init__(self, loss=None, optimizer=None):
        self.optimizer = optimizer
        self.loss = optimizer
        self.model = None
    def create_vgg19_model(self, loss='mean_squared_error' , optimizer=optimizers.Adam(), metrics=['accuracy']):
        
        model = Sequential()
        model.add(VGG19(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64,  activation="relu"))
        model.add(Dense(4,   activation="sigmoid"))
    

        model.layers[-7].trainable = False
        
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        
        self.model = model
        
        return self.model
     
    def save_model(self,name, format='h5'):
        #!mkdir -p saved_model
        self.model.save(f'saved_model/{name}.h5')
        
        

class OCR:
    def dect(src_path):
        reader = easyocr.Reader(['en'],gpu=False) # this needs to run only once to load the model into memory
        result = reader.readtext(src_path + '/images/Cars4.png')
        #for img in img_list: print(img)
        return result