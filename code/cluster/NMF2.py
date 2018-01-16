#module
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.datasets import fetch_olivetti_faces
from numpy.random import RandomState
#设置图像展示时的排列情况
n_row , n_col = 2,3
# 设置提取的特征数目
n_components = n_row * n_col
image_shape = (64,64)

dataset = fetch_olivetti_faces(shuffle = True,random_state = RandomState(0))
faces = dataset.data

def plot_gallery(title, images, n_col = n_col, n_row = n_row):
    plt.figure(figsize = (2. * n_col, 2.26 * n_row))
    plt.suptitle(title , size = 16) 
    
    for i, comp in enumerate(images):
        plt.subplot( n_row, n_col, i + 1)
        vmax = max(comp.max() , -comp.min())
        plt.imshow(comp.reshape(image_shape),camp=plt.cm.gray, interpolation= 'nearest',vmin = -vmax, vmax = vmax)
        plt.xticks(())
        plt.yticks(())
        
    plt.subplots_adjust(0.01,0.05,0.99,0.93,0.04,0.)
    
    #创建特征提取的对象NMF，使用PCA作为对比
estimators = [
    ('Eigenfaces - PCA using randomized SVD',decomposition.PCA(n_components = 6,whiten =True)),
    ('Non-negative components - NMF',decomposition.NMF(n_components= 6,init = 'nndsvda',tol=5e-3))]
    #降维后数据点的可视化
for name, estimator in estimators:
    
    estimator.fit(faces)
    components_ = estimator.components_
    plot_gallery(name,components_[:n_components])
plt.show() 
