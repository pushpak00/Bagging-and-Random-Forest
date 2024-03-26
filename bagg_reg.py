import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import os
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.pipeline import Pipeline

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Concrete Strength")
concrete = pd.read_csv("Concrete_Data.csv")
X = concrete.drop('Strength', axis=1)
y = concrete['Strength']

kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
lr = LinearRegression()
ridge = Ridge()
lasso = Lasso()
elastic = ElasticNet()
dtr = DecisionTreeRegressor(random_state=2022)

bagging = BaggingRegressor(random_state=2022, n_estimators=15)
print(bagging.get_params())
params = {'estimator':[lr,ridge,lasso,elastic,dtr]}
gcv = GridSearchCV(bagging, param_grid=params,
                   cv=kfold, scoring='r2')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

