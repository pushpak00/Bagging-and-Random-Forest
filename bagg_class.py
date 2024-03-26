import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Bankruptcy")

brupt = pd.read_csv("Bankruptcy.csv", index_col=0)
X = brupt.drop(['D', 'YR'], axis=1)
y = brupt['D']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    train_size=0.7,
                                                    random_state=2022)

lr = LogisticRegression()
bagging = BaggingClassifier(estimator=lr,
                            random_state=2022, n_estimators=15, oob_score=True)

bagging.fit(X_train, y_train)
print("Out of Bag Score =", bagging.oob_score_)
y_pred = bagging.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = bagging.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

### SVM Linear
scaler = StandardScaler()
svm_l = SVC(kernel='linear')
pipe_l = Pipeline([('STD',scaler),('SVM',svm_l)])
bagging = BaggingClassifier(estimator=pipe_l,
                            random_state=2022, n_estimators=15)
bagging.fit(X_train, y_train)
y_pred_prob = bagging.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

### SVM Radial
scaler = StandardScaler()
svm_r = SVC(kernel='rbf')
pipe_r = Pipeline([('STD',scaler),('SVM',svm_r)])
bagging = BaggingClassifier(estimator=pipe_r,
                            random_state=2022, n_estimators=15)
bagging.fit(X_train, y_train)
y_pred_prob = bagging.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

### Linear DA
da = LinearDiscriminantAnalysis()
bagging = BaggingClassifier(estimator=da,
                            random_state=2022, n_estimators=15)
bagging.fit(X_train, y_train)
y_pred_prob = bagging.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

### D Tree
dtc = DecisionTreeClassifier(random_state=2022)
bagging = BaggingClassifier(estimator=dtc,
                            random_state=2022, n_estimators=15)
bagging.fit(X_train, y_train)
y_pred_prob = bagging.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

###################### Grid Search CV ###########################
kfold = StratifiedKFold(n_splits=5, random_state=2022, shuffle=True)
lr = LogisticRegression()
scaler = StandardScaler()
svm_l = SVC(kernel='linear')
pipe_l = Pipeline([('STD',scaler),('SVM',svm_l)])
svm_r = SVC(kernel='rbf')
pipe_r = Pipeline([('STD',scaler),('SVM',svm_r)])
da = LinearDiscriminantAnalysis()
dtc = DecisionTreeClassifier(random_state=2022)
bagging = BaggingClassifier(random_state=2022, n_estimators=15)
print(bagging.get_params())
params = {'estimator':[lr,pipe_l,pipe_r,da,dtc]}
gcv = GridSearchCV(bagging, param_grid=params, cv=kfold,
                   verbose=3, scoring='roc_auc')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
