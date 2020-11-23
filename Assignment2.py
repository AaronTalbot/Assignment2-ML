# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:32:38 2020

@author: aaron
"""
import pandas as pd
import numpy as np
import math
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt

def Task1():
    data = pd.read_csv("product_images.csv")
    Sneakers = data[data["label"]== 0]
    Ankle_boots = data[data["label"]==1]
    # print(Sneakers.head())
    # print("="*50)
    # print(Ankle_boot.head())
    
    # print("Sneakers    = " + str(Sneakers.shape[0]))
    # print("Ankle boots = " + str(Ankle_boot.shape[0]))
    Train_Sneakers = Sneakers.head(3000) # As the sneakers total = 7000
    Train_Ankle_boots = Ankle_boots.head(3000)
    
    print(Train_Sneakers.shape[0])
    print(Train_Ankle_boots.shape[0])

    test_Sneakers = Sneakers.tail(4000)
    test_Ankle_boots = Ankle_boots.tail(4000)
    
    Train_Sneakers_Pixels = Train_Sneakers.drop(Train_Sneakers.columns[0], axis = 1)
    Train_Ankle_Boots_pixels = Train_Ankle_boots.drop(Train_Ankle_boots.columns[0], axis = 1)
    
    First_Sneaker = Train_Sneakers_Pixels.iloc[0].to_numpy().reshape(28,28)
    First_Boot = Train_Ankle_Boots_pixels.iloc[0].to_numpy().reshape(28,28)
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(First_Sneaker, cmap='gray')
    ax[1].imshow(First_Boot, cmap='gray')
    return Train_Sneakers, Train_Ankle_boots, test_Sneakers, test_Ankle_boots

Task1()
