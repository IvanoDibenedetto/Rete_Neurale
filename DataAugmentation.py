# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 13:35:13 2020

@author: Ivano Dibenedetto mat. 654678
"""

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tqdm import tqdm
import os
import numpy as np

pathPneumonia = "Covid19_dataset_global/Polmonite/"
pathCovid = "Covid19_dataset_global/Covid/"

data_gen = ImageDataGenerator(
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=10,
    height_shift_range=0.15,
    width_shift_range=0.15,
    brightness_range=[0.2, 1])


def data_augmentation():
    # genero le immagini nel dataset Polmonite
    for k in tqdm(os.listdir(pathPneumonia)):
        img = load_img(os.path.join(pathPneumonia, k), grayscale=True)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in data_gen.flow(x, batch_size=1, save_to_dir=pathPneumonia, save_prefix="pneumonia"):
            i += 1
            if i > 4:
                break

    # genero le immagini nel dataset Covid
    for k in tqdm(os.listdir(pathCovid)):
        img = load_img(os.path.join(pathCovid, k), grayscale=True)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in data_gen.flow(x, batch_size=1, save_to_dir=pathCovid, save_prefix="covid"):
            i += 1
            if i > 4:
                break


data_augmentation()
