# city



#import module
import numpy as np
from sklearn.cluster import KMeans

def loadData(filePath):  #define the read function
    fr = open(filePath,'r+')
    lines = fr.readlines()
    retData = []
    retCityName =[]
    for line in lines:
        items = line.strip().split(",")
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1,len(items))])

if __name__ == '__main__':
    data,cityName = loadData('city.txt')
    # print(data) #data 是有8个特征的一个集合
    km = KMeans(n_clusters=3)  #人为指定分为几类
    label = km.fit_predict(data)
    expenses = np.sum(km.cluster_centers_,axis=1)
    #print(km.cluster_centers_) #这个是什么,

    

CityCluster = [[],[],[]]
for i in range(len(cityName)):
    CityCluster[label[i]].append(cityName[i])
for i in range(len(CityCluster)):
    print("Ecpense:%.2f"%expenses[i])
    print(CityCluster[i])

> 结果  

    Ecpense:5113.54
    ['天津', '江苏', '浙江', '福建', '湖南', '广西', '海南', '重庆', '四川', '云南', '西藏']
    Ecpense:7754.66
    ['北京', '上海', '广东']
    Ecpense:3827.87
    ['河北', '山西', '内蒙古', '辽宁', '吉林', '黑龙江', '安徽', '江西', '山东', '河南', '湖北', '贵州', '陕西', '甘肃', '青海', '宁夏', '新疆']
    


