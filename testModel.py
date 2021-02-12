# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 08:48:10 2020

@author: Ivano Dibenedetto mat.654678
"""

import tensorflow as tf
import cv2
import os
from os import path
from tqdm import tqdm
from keract import display_activations
from keract import get_activations

CATEGORIES = ["Covid", "Polmonite"]
DATADIR = "Test"


def prepare(filepath):
    IMG_SIZE = 250
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("model-3264128.model")

for img in (os.listdir(DATADIR)):
    name, extension = path.splitext(img)
    prediction = model.predict([prepare(path.join(DATADIR, img))])
    cv2img = cv2.imread(path.join(DATADIR, img), cv2.IMREAD_GRAYSCALE)
    #activations = get_activations(model, prepare(path.join(DATADIR, img)))
    #display_activations(activations, cmap="gray")
    print("Categoria: ", name, "==", CATEGORIES[int(prediction[0])])

