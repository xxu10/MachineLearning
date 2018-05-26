# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 12:27:26 2018

@author: Administrator
"""

import scipy.io as sio
import numpy as np 
from sklearn import svm


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
    set1 = group1[:,0:4]
    set2 = group2[:,0:4]
    set3 = group3[:,0:4]
    set4 = group4[:,0:4]
    group = np.row_stack((set1,set2,set3,set4))
    print("---Getting training label...")
    label1 = group1[:,4]
    label2 = group2[:,4]
    label3 = group3[:,4]
    label4 = group4[:,4]
    label = np.row_stack((label1,label2,label3,label4))
    label = label.flatten()
    return group,label
   
def createTestSet(group):
    print("---Getting test dataset...")
    set1 = group[:,0:4]
    label1 = group[:,4]
    label1 = label1.flatten()
    return set1,label1

def Svm(data,label,predict_value):
    clf = svm.SVC(probability=True)
    clf.fit(data, label) 
    predicts = clf.predict(predict_value)
    '''
    Compute probabilities of possible outcomes for samples in X.
    '''
    proba = clf.predict_proba(predict_value)
    return predicts,proba

def findproba(proba):
     proba_value = []
     for i in range(len(proba)):
        proba_value.append(np.amax(proba[i]))
     return proba_value

def Accuracy(predicts):
    cout = 0
    for i in range(len(predicts)):
         if predicts[i] == test_label[i]:
                 cout=cout+1
    accuracy=cout/len(predicts)
    return   accuracy

    
    
'''
put 'week6.mat' in the file (the same just like week1,2,3,4,5 ), then
change to "datafile ='week6.mat' " and " group.get('week6') "
'''
dataFile = 'week5.mat'
group = sio.loadmat(dataFile) 
group = group.get('week5')
train_data,train_label = createDataSet()
test_data,test_label = createTestSet(group)
'''
 the outcomes of the SVM
'''
print("---The SVM method : \n")
predicts,proba= Svm(train_data,train_label,test_data)
proba_value = findproba(proba)
accuracy = np.mean(proba_value)
regaccuracy = Accuracy(predicts)
print('---the predicted class probabilities: \n', proba_value)
print("---test_label: \n" , test_label)
print("---predicts: \n" , predicts)
print('---average accuracy: %.2f%%' % (100 * accuracy))
'''
the average accuracy is the average possibility of find every label
the regression accuracy is the possibility of accurate classification of entire
test dataset(which means the total number of correct classification / the total 
number of test_label)
'''

print('---regression accuracy: %.2f%%' % (100 * regaccuracy))