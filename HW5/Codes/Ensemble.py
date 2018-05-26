# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 00:13:07 2018

@author: Administrator
"""

import scipy.io as sio
import sys
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def createDataSet():
    # creat a matrix: each row as a sample
    print("---Getting training dataset...")
    f = open(r"C://Users//Administrator//Desktop//Intrusion-Detection-ML-master//intrusion_train_raw.txt","r") 
    train_data = list()
    train_label = list()
    train_sets = f.readlines()
    for i in range(len(train_sets)):
       c_array = train_sets[i].split(",")
       train = c_array[4:41]
       train_data.append(train)
       label = c_array[-1]
       train_label.append(label[:-2])
    train_label[4996]='satan'
    return train_data,train_label

def createTestSet():
    print("---Getting test dataset...")
    f = open(r"C://Users//Administrator//Desktop//Intrusion-Detection-ML-master//intrusion_test_raw.txt","r") 
    test_data = list()
    test_label = list()
    test_sets = f.readlines()
    for i in range(len(test_sets)):
       c_array = test_sets[i].split(",")
       test = c_array[4:41]
       test_data.append(test)
       label = c_array[-1]
       if(label[:-2] == '1'):
           test_label.append('smurf')
       else:
           test_label.append(label[:-2])       
    test_label[1235]='satan'
    return test_data,test_label

print ('reading training and testing data...')  
traindata, trainlabel=createDataSet()
testdata, testlabel=createTestSet()
clf = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
clf.fit(traindata,trainlabel)
predicts = clf.predict(testdata)      
cout=0
for i in range(len(predicts)):
    if predicts[i] == testlabel[i] :
        cout=cout+1
accuracy=cout/len(predicts)
print('accuracy: %.2f%%' % (100 * accuracy))
