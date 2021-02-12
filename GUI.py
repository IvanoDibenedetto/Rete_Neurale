# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 22:38:07 2020

@author: Ivano Dibenedetto mat.654648
"""

from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import tensorflow as tf

root = Tk(className=' CLASSIFICATORE')
root.geometry("800x600+250+150")
root.resizable(width=False, height=False)

filename = ""
CATEGORIES = ["Covid", "Polmonite"]


def LoadImage():
    filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                          filetypes=(("jpeg files", "*.jpg"), ("png files", "*.png")))

    load = Image.open(filename)
    load = load.resize((300, 300), Image.ANTIALIAS)
    render = ImageTk.PhotoImage(load)
    img = Label(image=render, text=filename)
    img.image = render
    img.place(x=250, y=150)
    classifica(filename)


def classifica(filename):

    print(filename)
    model = tf.keras.models.load_model("X-ray_CNN.model")
    prediction = model.predict_proba([prepare(filename)])
    score = prediction[0]
    ris = "il paziente è affetto da " + CATEGORIES[int(prediction[0][0]) ]
   

    print(
        "This image is %.2f percent COVID and %.2f percent PNEUMONIA."
        % (100 * (1 - score), 100 * score))

    predLabel.config(text=ris)


def prepare(filepath):
    IMG_SIZE = 250 # dimensione
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) # scala di grigi
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


introLabel = ttk.Label(root,
                       text="Rete neurale che classifica l'immagine di una radiografia al torace in una delle possibili etichette [Polmonite, Covid-19]")
introLabel.place(relx=0.1, rely=0.05)

loadButton = ttk.Button(text="Carica immagine", command=LoadImage)
loadButton.place(relx=0.5, rely=0.15, anchor=CENTER)

predLabel = ttk.Label(root, text="")
predLabel.place(relx=0.5, rely=0.8, anchor=CENTER)

root.mainloop()
