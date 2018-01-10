
# coding: utf-8

import numpy as np
import sklearn.cluster as skc
from sklearn import metrics
import matplotlib.pyplot as plt


mac2id = dict()
onlinetimes = []
with open('time.txt') as f:
    for lines in f:
        mac = lines.split(',')[2]
        onlinetime = int(lines.split(',')[6])
        starttime = int(lines.split(',')[4].split(' ')[1].split(':')[0])
        if mac not in mac2id:
            mac2id[mac] = len(onlinetimes)  
            onlinetimes.append((starttime,onlinetime))
        else:
            onlinetimes[mac2id[mac]] = [(starttime,onlinetime)]
    real_X = np.array(onlinetimes).reshape((-1,2))
                                    
        
X = real_X[:,0:1]
db =  skc.DBSCAN(eps=0.01,min_samples=20).fit(X)
labels = db.labels_

print('Labels:')
print(labels)
ratio = len(labels[labels[:] == -1])/len(labels)
print('Noise ratio:',format(ratio,'.2%'))

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters : %d' % n_clusters_)
print('Silhouette Cofficient : %0.3f ' % metrics.silhouette_score(X,labels))

for i in range(n_clusters_):
    print('Cluster ',i ,':')
    print(list(X[labels == i].flatten()))


get_ipython().magic('matplotlib inline')
plt.hist(X,24)



Y = np.log(1+real_X[:,1:])
db_Y = skc.DBSCAN(eps=0.14,min_samples=10).fit(Y)
labels_Y = db_Y.labels_

print("Labels_Y:")
print(labels_Y)
ratio = len(labels_Y[labels_Y[:] == -1])/len(labels_Y)
print('Noise ratio:',format(ratio,'.2%'))


n_clusters_Y_ = len(set(labels_Y)) - (1 if -1 in labels_Y else 0 )

print('Estimated number of clusters: %d' % n_clusters_Y_)
print("Silhouuette Cofficient : %0.3f " % metrics.silhouette_score(Y,labels_Y))

for i in range(n_clusters_Y_):
    print('Cluster ',i,':')
    count = len(Y[labels_Y == i])
    mean = np.mean(real_X[labels_Y == i][:,1])
    std = np.std(real_X[labels_Y == i][:,1])
    print('\t number of sample:',count)
    print('\t mean of sample :',format(mean,'.1f'))
    print('\t std of sample :',format(std,'.1f'))

