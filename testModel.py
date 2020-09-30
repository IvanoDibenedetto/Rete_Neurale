# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 08:48:10 2020

@author: Ivano Dibenedetto mat.654678
"""



import tensorflow as tf
import cv2

CATEGORIES = ["Covid", "Polmonite"]  


def prepare(filepath):
    IMG_SIZE = 450  
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) 
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) 
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("X-ray_CNN.model")

prediction = model.predict([prepare('test/PNEUMONIA_TEST.jpg')]) 
print(prediction)
print(CATEGORIES[int(prediction[0][0])])


prediction = model.predict([prepare('test/COVID_TEST4.jpeg')])
print(prediction)
print(CATEGORIES[int(prediction[0][0])])
