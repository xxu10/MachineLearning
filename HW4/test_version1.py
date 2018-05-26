# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 14:10:40 2018

@author: Administrator
"""

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
knn = neighbors.KNeighborsClassifier()     
data = train_x
labels =train_y[0]
knn.fit(data,labels)   
predicts=knn.predict(test_x)
cout=0
for i in range(len(predicts)):
    if predicts[i] == test_y[0][i] :
        cout=cout+1
accuracy=cout/len(predicts)
print('accuracy: %.2f%%' % (100 * accuracy))
k=reg.coef_


train_data,train_label = createDataSet()
dataFile = 'week5.mat'
test_data,test_label = createTestSet(dataFile)
predicts = linear_modelRid(train_data,train_label,test_data)
print(predicts)
print(test_label)
print()
cout = 0
for i in range(len(predicts)):
    if abs(predicts[i] - test_label[i]) < 1 :
        cout=cout+1
accuracy=cout/len(predicts)
print('accuracy: %.2f%%' % (100 * accuracy))


 label1 = group1[:,4]
    label2 = group2[:,4]
    label3 = group3[:,4]
    label4 = group4[:,4]
    label = np.row_stack((label1,label2,label3,label4))
    label = label.flatten()



cout = 0
for i in range(len(predicts)):
    if abs(predicts[i] - test_label[i]) < 1 :
        cout=cout+1
accuracy=cout/len(predicts)
print('accuracy: %.2f%%' % (100 * accuracy))






