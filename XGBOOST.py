import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance



iris = load_iris()
X,y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 123)

# 用于分类， 用于回归同理 使用XGBRegressor
model = XGBClassifier(
    # 优化的参数
    max_depth = 3, # 单棵树的最大深度 取值范围为： 3~15
    learning_rate = 0.1, # 学习率    取值范围为：10-4-10-1
    n_estimators = 100, # 树的数量    取值范围为： 50-200
    
    # 需要设置的参数
    verbosity = 2,  # 输出日志信息
    objective = 'binary:logistic', # 二分类：binary:logistic, 多分类：multi:softmax
    n_jobs = -1, # 并行计算
    
    # 优化的参数，控制树的复杂度 0.001， 0.01， 0.1， 1， 10， 100
    gamma = 0,  # 空值叶子数量 类似于岭回归中的正则化参数
    reg_alpha = 0,  # L1正则，限制叶子的取值，一般比较少优化
    reg_alpha = 1, # L2正则 限制叶子的取值
    
    # 两个随机
    subsample = 1 # 建立每棵树，采用的比例，样本随机 0.5-1
    colsample_bytree = 1 # 建立单棵树的特征的随机 特征随机 0.5-1

)

model.fit(X_train, y_train)

model.score(X_test, y_test)

plot_importance(model) # 显示重要特征
plt.show()


# XGBOOST优化

params = {
    'n_estimators': range(100, 300, 50),
    'max_depth': range(5, 15, 2),
    'learning_rate': np.linspace(0.01, 0.2, 10)
}

grid_search = GridSearchCV(model, param_grid = params, cv = 10, verbose = 2, n_jobs = -1)
grid_search.fit(X_train, y_train)
grid_search.best_params_

grid_search.score(X_test, y_test)