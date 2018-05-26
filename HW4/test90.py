# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 12:13:50 2018

@author: Administrator
"""

import scipy.io as sio
import numpy as np 
import matplotlib.pyplot as plt   
from sklearn import linear_model
from sklearn import svm
from sklearn.neural_network import MLPClassifier

def createDataSet():
    print("---Getting training dataset...")
    dataFile = 'week1.mat' 
    group1 = sio.loadmat(dataFile) 
    group1 = group1.get('week1')
    dataFile= 'week2.mat'
    group2 = sio.loadmat(dataFile) 
    group2 = group2.get('week2')
    dataFile= 'week3.mat'
    group3 = sio.loadmat(dataFile)
    group3 = group3.get('week3')
    dataFile= 'week4.mat'
    group4 = sio.loadmat(dataFile) 
    group4 = group4.get('week4')
    set1 = group1[:,1:5]
    set2 = group2[:,1:5]
    set3 = group3[:,1:5]
    set4 = group4[:,1:5]
    group = np.row_stack((set1,set2,set3,set4))
    set1 = []
    label = []
    for i in range(len(group)):
         if  group[i][2] > 0.9 :
            set1.append(group[i][0])
            set1.append(group[i][1])
            label.append(group[i][3])
    n = int(len(set1)/2)
    set1 = np.array(set1).reshape(n,2)
    print("---Getting training label...")
    
    return set1,label

#ws=(X.T*X).I*(X.T*Y)    
def linear_modelRid(data,label,predict_value):
    reg = linear_model.Ridge(alpha = 0.1)
    reg.fit(data,label)
    predicts = reg.predict(predict_value)
    return predicts

def createTestSet(dataFile):
    print("---Getting test dataset...")
    group = sio.loadmat(dataFile) 
    group = group.get('week5')
    set1 = group[:,1:3]
    label1 = group[:,4]
    label1 = label1.flatten()
    return set1,label1

def Svm(data,label,predict_value):
    clf = svm.SVR()
    clf.fit(data, label) 
    predicts = clf.predict(predict_value)
    return predicts

def Mlp(data,label,predict_value):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(data, label)                 
    predicts = clf.predict(predict_value)
    return predicts
 

    
train_data,train_label = createDataSet()
dataFile = 'week5.mat'
test_data,test_label = createTestSet(dataFile)
predicts = Mlp(train_data,train_label,test_data)
print(predicts)
print(test_label)
cout = 0
for i in range(len(predicts)):
    if abs(predicts[i] - test_label[i]) < 1 :
        cout=cout+1
accuracy=cout/len(predicts)
print('accuracy: %.2f%%' % (100 * accuracy))
