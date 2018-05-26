# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 15:03:31 2018

@author: Administrator
"""

def Accuracy(predicts):
    cout = 0
    for i in range(len(predicts)):
         if predicts[i] == test_label[i]:
                 cout=cout+1
    accuracy=cout/len(predicts)
    return   accuracy

def Regression(data,label,predict_value):
    predict1 = Randomforest(data,label,predict_value)
    accuracy1 = Accuracy(predict1)
    predict2 = Svm(data,label,predict_value)
    accuracy2 = Accuracy(predict2)
    if accuracy1 > accuracy2 :
       return predict1,accuracy1
    else :
       return predict2,accuracy2 
   
    
    score = CM.score(data,label,sample_weight=None)
    
    
    if accuracy1 > accuracy2 :
       return predict1,proba1,accuracy1
    else :