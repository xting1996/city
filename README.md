# city


```python
#import module
import numpy as np
from sklearn.cluster import KMeans
```


```python
def loadData(filePath):  #define the read function
    fr = open(filePath,'r+')
    lines = fr.readlines()
    retData = []
    retCityName =[]
    for line in lines:
        items = line.strip().split(",")
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1,len(items))])
    return retData,retCityName
```


```python
if __name__ == '__main__':
    data,cityName = loadData('city.txt')
    # print(data) #data 是有8个特征的一个集合
    km = KMeans(n_clusters=3)  #人为指定分为几类
    label = km.fit_predict(data)
    expenses = np.sum(km.cluster_centers_,axis=1)
    print(km.cluster_centers_) #这个是什么,
```

    [[ 1559.73176471   477.77941176   317.49176471   224.31411765   237.32
        470.22882353   353.51764706   187.48235294]
     [ 2287.12363636   491.29636364   424.62272727   231.39         345.48909091
        601.17909091   459.10090909   273.33818182]
     [ 3242.22333333   544.92         735.78         405.51333333   602.25
       1016.62         760.52333333   446.82666667]]
    


```python
CityCluster = [[],[],[]]
for i in range(len(cityName)):
    CityCluster[label[i]].append(cityName[i])
for i in range(len(CityCluster)):
    print("Ecpense:%.2f"%expenses[i])
    print(CityCluster[i])
```
> 结果  

    Ecpense:5113.54
    ['天津', '江苏', '浙江', '福建', '湖南', '广西', '海南', '重庆', '四川', '云南', '西藏']
    Ecpense:7754.66
    ['北京', '上海', '广东']
    Ecpense:3827.87
    ['河北', '山西', '内蒙古', '辽宁', '吉林', '黑龙江', '安徽', '江西', '山东', '河南', '湖北', '贵州', '陕西', '甘肃', '青海', '宁夏', '新疆']
    


```python

```


```python

```
