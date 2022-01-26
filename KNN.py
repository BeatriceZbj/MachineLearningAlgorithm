# KNN手写代码 方法一

from operator import ne
import numpy as np
import collections

dataset = np.array([[1,101],[5,89],[108,5],[115,8]]) # (4,2)
labels = ['A','B','C','D']

test_data = [102,20]

k = 3 # Hyperparameter

distance = np.sum((test_data - dataset)**2, axis = 1)**0.5 # 欧几里得距离
k_labels = [labels[index] for index in distance.argsort()[0:k]]
label = collections.Counter(k_labels).most_common(1)[0][0] #列表计数，出现最多的标签即为最终的类别
print(label)


# KNN手写代码 方法二

import pandas as pd
k = 3
dataset = np.array([[1,101],[5,89],[108,5],[115,8]]) 
df = pd.DataFrame(dataset, columns = ['X','Y'])
df['label'] = ['C','C','C','D']
df['distance'] = np.sum((df.iloc[:,[0,1]].values-test_data)**2 , axis = 1)**0.5
df.sort_values(by = 'distance')['label'][:k].mode()



# sklearn中的KNN
# parameters:
'''
n_neighbors: 选取多少个邻居（K值）
weights: 为了解决数据的不平衡，离的近的给高权重 （uniform:等权重， distance : 权重和距离呈反比， 自定义）
algorithm: 'auto','ball_tree','kd_tree','brute'(暴力查找)
leaf_size: 这个是算法里面的参数，多少分一个组
p: 距离方式，1 曼哈顿，2 欧几里得 3 闵式距离
n_jobs: 并行计算
'''

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X,y)
neigh.predict([1.1])


# 超参数优化

from sklearn.model_selection import GridSearchCV
params = {'weights':['uniform','distance'],'n_neighbors':range(2,22)}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid = params, cv = 10, verbose = 2, n_jobs = -1)
grid_search.fit(x_train, y_train)
grid_search.best_params_

