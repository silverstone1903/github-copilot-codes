from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2021)


# Logistic Regression
print("Logistic Regresstion")
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
print(confusion_matrix(y_test, lr_preds))
print(classification_report(y_test, lr_preds))
print(accuracy_score(y_test, lr_preds))

# Random Forest
print("Random Forest")
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print(confusion_matrix(y_test, rf_preds))
print(classification_report(y_test, rf_preds))
print(accuracy_score(y_test, rf_preds))

# XGBoost
print("XGBoost")
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
print(confusion_matrix(y_test, xgb_preds))
print(classification_report(y_test, xgb_preds))
print(accuracy_score(y_test, xgb_preds))

# LightGBM
print("LightGBM")
lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)
lgbm_preds = lgbm.predict(X_test)
print(confusion_matrix(y_test, lgbm_preds))
print(classification_report(y_test, lgbm_preds))
print(accuracy_score(y_test, lgbm_preds))

# CatBoost
print("CatBoost")
cat = CatBoostClassifier()
cat.fit(X_train, y_train)
cat_preds = cat.predict(X_test)
print(confusion_matrix(y_test, cat_preds))
print(classification_report(y_test, cat_preds))
print(accuracy_score(y_test, cat_preds))