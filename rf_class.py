import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import os
from sklearn.ensemble import RandomForestClassifier

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Bankruptcy")

brupt = pd.read_csv("Bankruptcy.csv", index_col=0)
X = brupt.drop(['D', 'YR'], axis=1)
y = brupt['D']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    train_size=0.7,
                                                    random_state=2022)

rf = RandomForestClassifier(random_state=2022)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = rf.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

################ Grid Search CV ####################
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
params = {'max_features':np.arange(3,15)}
gcv = GridSearchCV(rf, param_grid=params, cv=kfold,
                   verbose=3, scoring='roc_auc')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

###### Feature Importance Plot ############
import matplotlib.pyplot as plt
best_model = gcv.best_estimator_
imps = best_model.feature_importances_
plt.barh(X.columns, imps)
plt.show()

i_sorted = np.argsort(-imps)
n_sorted = X.columns[i_sorted]
imp_sort = imps[i_sorted]
plt.barh(n_sorted, imp_sort)
plt.title("Sorted Feature Importances")
plt.show()

################### HR ###############################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\human-resources-analytics")

hr = pd.read_csv("HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first=True)
X = dum_hr.drop('left', axis=1)
y = dum_hr['left']

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
params = {'max_features':np.arange(3,15)}
gcv = GridSearchCV(rf, param_grid=params, cv=kfold,
                   verbose=3, scoring='roc_auc')
gcv.fit(X,y)
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

################### Image Segmentation #######################
from sklearn.preprocessing import LabelEncoder
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Image Segmentation")

image_seg = pd.read_csv("Image_Segmention.csv")
X = image_seg.drop('Class', axis=1)
y = image_seg['Class']

le = LabelEncoder()
le_y = le.fit_transform(y)
print(le.classes_)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
params = {'max_features':np.arange(3,15)}
gcv = GridSearchCV(rf, param_grid=params, cv=kfold,
                   verbose=3, scoring='neg_log_loss')
gcv.fit(X,le_y)
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






