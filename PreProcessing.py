# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 09:32:14 2020

@author: Ivano Dibenedetto mat. 654678
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle


DATADIR = "/Covid19_dataset_global"

CATEGORIES = ["Covid","Polmonite"]
IMG_SIZE = 450
training_data = []


def create_training_data():
    
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        
        for img in tqdm(os.listdir(path)):
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
            training_data.append([new_array,class_num])


create_training_data()

print(len(training_data))
random.shuffle(training_data)

X = []
y = []


for features,label in training_data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()
