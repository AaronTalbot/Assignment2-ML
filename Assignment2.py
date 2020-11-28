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
from sklearn import linear_model
from sklearn import svm
import matplotlib.pyplot as plt
import time

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
    Labels = data.iloc[:,0]
    Train_Target = Labels
    Train_Data = data

    # print(Train_Sneakers.shape[0])
    # print(Train_Ankle_boots.shape[0])

    test_Sneakers = Sneakers.tail(4000)
    test_Ankle_boots = Ankle_boots.tail(4000)
    
    Train_Sneakers_Pixels = Train_Sneakers.drop(Train_Sneakers.columns[0], axis = 1)
    Train_Ankle_Boots_pixels = Train_Ankle_boots.drop(Train_Ankle_boots.columns[0], axis = 1)
    
    First_Sneaker = Train_Sneakers_Pixels.iloc[0].to_numpy().reshape(28,28)
    First_Boot = Train_Ankle_Boots_pixels.iloc[0].to_numpy().reshape(28,28)
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(First_Sneaker, cmap='gray')
    ax[1].imshow(First_Boot, cmap='gray')
    
    return Train_Data, data, Train_Target


def Task2():
    Size = 14000
    Train_Data_All, data, Train_Target_All = Task1()
    List_of_Splits = [2,3,4,5]
    List_of_N_Samples = [500,1120,1500]
    
    for i in List_of_Splits:
        
        for j in List_of_N_Samples:
            print("="*80)
            print("Splits = ", i)
            print("Training Samples = ", j)
            kf = model_selection.KFold(n_splits=i, shuffle=True)
            Train_Data = Train_Data_All.head(j)
            Train_Target = Train_Target_All.head(j)
            
            Test_data = Train_Data_All.tail(Size-j)
            Test_Target = Train_Target_All.tail(Size-j)
            
            
            for train_index, test_index in kf.split(Train_Data.values):
                
                print("-"*50)
                print("Perceptron")
                start = time.time()
                classifier = linear_model.Perceptron()
                
                classifier.fit(Train_Data.values[train_index],Train_Target.values[train_index])
                end = time.time()
                print("Training Time for samples " + str(j) + "of split size " + str(i) + "  = "+ str(end - start))
                
                startPred = time.time()
                prediction_Train = classifier.predict(Train_Data.values[test_index])
                finPred = time.time()
                
                print("Prediction time is for samples " + str(j) + "of split size " + str(i) + "  = " + str(finPred-startPred))
                
                clasScore = metrics.accuracy_score(Train_Target.values[test_index],prediction_Train)
                
                print("Accuraccy = ",clasScore)
                
                Test_Prediction = classifier.predict(Test_data)
                TestScore = metrics.accuracy_score(Test_Target,Test_Prediction)
                print("Test Accuracy = ", TestScore)
                print("-"*50)
                print("SVM radial based")
                clf2 = svm.SVC(kernel="rbf", gamma=1e-9)
                clf2.fit(Train_Data.values[train_index],Train_Target.values[train_index])
                startSVM = time.time()
                prediction2 = clf2.predict(Train_Data.values[test_index])
                endSVM = time.time()
                print("Prediction time of SVM is for samples " + str(j) + "of split size " + str(i) + "  = " + str(endSVM-startSVM))
                clasScore2 = metrics.accuracy_score(Train_Target.values[test_index],prediction2)
                
                print("Accuracy = ", clasScore2)
                
                
                print("-"*50)
                print("SVM linear based")
                
                clf3 = svm.SVC(kernel="linear")
                clf3.fit(Train_Data.values[train_index],Train_Target.values[train_index])
                startSVM2 = time.time()
                prediction3 = clf3.predict(Train_Data.values[test_index])
                endSVM2 = time.time()
                print("Prediction time of SVM is for samples " + str(j) + "of split size " + str(i) + "  = " + str(endSVM2-startSVM2))
                clasScore3 = metrics.accuracy_score(Train_Target.values[test_index],prediction3)
                print("Accuracy = ", clasScore3)
                
                
       
        
Task2()
        
        
        
        
        
        