import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import os
from sklearn.ensemble import RandomForestRegressor

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Medical Cost Personal")
insure = pd.read_csv("insurance.csv")
dum_ins = pd.get_dummies(insure, drop_first=True)
X = dum_ins.drop('charges', axis=1)
y = dum_ins['charges']

kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
rf = RandomForestRegressor(random_state=2022)
params = {'max_features':np.arange(2,9)}
gcv = GridSearchCV(rf, param_grid=params,
                   cv=kfold, scoring='r2')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)


###### Feature Importance Plot ############
import matplotlib.pyplot as plt
best_model = gcv.best_estimator_
imps = best_model.feature_importances_
i_sorted = np.argsort(-imps)
n_sorted = X.columns[i_sorted]
imp_sort = imps[i_sorted]
plt.barh(n_sorted, imp_sort)
plt.title("Sorted Feature Importances")
plt.show()
