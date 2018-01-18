
# coding: utf-8

# In[26]:

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
#修改了引进的包
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
#model
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import os
os.chdir("F:/weiwe/project/")


# In[27]:

def load_dataset(feature_paths,label_paths):
    feature = np.ndarray(shape = (0,41))
    label = np.ndarray(shape = (0,1))
    for file in feature_paths:
        df = pd.read_table(file,delimiter = ",",na_values = "?",header = None)
        imp = Imputer(missing_values = "NaN",strategy = 'mean',axis = 0)
        imp.fit(df)
        df = imp.transform(df)
        feature = np.concatenate((feature,df))
    for file in label_paths:
        df = pd.read_table(file,header = None)
        label = np.concatenate((label,df))
    label = np.ravel(label)
    return feature,label    


# In[28]:

if __name__ == '__main__':
    feature_paths = ["A1.feature.txt","B1.feature.txt","C1.feature.txt","D1.feature.txt","E1.feature.txt"]
    label_paths = ["A1.label.txt","B1.label.txt","C1.label.txt","D1.label.txt","E1.label.txt"]
    x_train ,y_train = load_dataset(feature_paths[:4],label_paths[:4])
    x_test ,y_test = load_dataset(feature_paths[4:],label_paths[4:])
    x_train ,x_,y_train,y_ = train_test_split(x_train,y_train,test_size = 0.0)


# In[29]:

print("strat training knn")
knn = KNeighborsClassifier().fit(x_train,y_train)
print("Training done!")
answer_knn = knn.predict(x_test)
print("Prediction done!")


# In[30]:

print("Start training DT")
dt = DecisionTreeClassifier().fit(x_train,y_train)
print("Training done!")
answer_dt = dt.predict(x_test)
print("Prediction")


# In[31]:

print("Start training Bayes")
gnb = GaussianNB().fit(x_train,y_train)
print("Training done!")
answer_gnb = gnb.predict(x_test)
print("Prediction done")


# In[32]:

print('\n\n The classification report for knn:')
print(classification_report(y_test,answer_knn))
print('\n\n The classfication report for DT:')
print(classification_report(y_test,answer_dt))
print('\n\n The classification report for Bayes:')
print(classification_report(y_test,answer_gnb))

