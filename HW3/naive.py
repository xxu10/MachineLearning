# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 17:05:15 2018

@author: Administrator
"""
import scipy.io as sio
from sklearn.ensemble import gradient_boosting
import numpy as np

def createDataSet():
    # creat a matrix: each row as a sample
    print("---Getting training dataset...")
    dataFile = 'C://Users//Administrator//Desktop//Training_data.mat' 
    group_dict = sio.loadmat(dataFile) 
    group = group_dict.get('X_train3')
    print("---Getting training label...")
    labelFile = 'C://Users//Administrator//Desktop//Training_label.mat' 
    labels_dict  = sio.loadmat(labelFile) 
    labels = labels_dict.get('Ytrain')
    return group, labels


print ('reading training and testing data...')  
train_x, train_y=createDataSet()
print("---Getting testing set...") 
testdataFile = 'C://Users//Administrator//Desktop//Test_data.mat' 
testX = sio.loadmat(testdataFile) 
test_x=testX.get('X_test')
print("---Getting testing label...") 
testlabelFile = 'C://Users//Administrator//Desktop//Test_label.mat' 
testY = sio.loadmat(testlabelFile)
test_y=testY.get('Ytest')
data = train_x
labels =train_y[0]
clf = gradient_boosting()
clf.fit(data,labels)
predicts = clf.predict(test_x)      
cout=0
for i in range(len(predicts)):
    if predicts[i] == test_y[0][i] :
        cout=cout+1
accuracy=cout/len(predicts)
print('accuracy: %.2f%%' % (100 * accuracy))
