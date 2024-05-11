
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('C:/Users/Ravi/Downloads/decision tree/Datasets_DTRF/Fraud_check.csv')
df.columns
df.isnull().sum()
df.isna().sum()
df.dropna()
df.drop_duplicates()
df.dtypes
df.shape
plt.boxplot(df.iloc[:,2:5])

df["income"]="<=30000"
df.loc[df["Taxable_Income"]>=30000,"income"]="Good"
df.loc[df["Taxable_Income"]<=30000,"income"]="Risky"

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for column_name in df.columns:
    if df[column_name].dtype == object:
        df[column_name] = le.fit_transform(df[column_name])
    else:
        pass

import seaborn as sns
sns.pairplot(data=df, hue = 'income')
##Splitting the data into featuers and labels
features = df.iloc[:,0:5]
labels = df.iloc[:,5]
## Collecting the column names
colnames = list(df.columns)
predictors = colnames[0:5]
target = colnames[5]


from sklearn.model_selection import train_test_split
train,test=train_test_split(df,test_size=0.3)

from sklearn.tree import DecisionTreeClassifier as DT


model = DT(criterion = 'entropy',max_depth=5)
model.fit(train[predictors], train[target])

from sklearn import tree
import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
tree.plot_tree(model,filled=True)



# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy



####grid searchcv


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)

rf_clf.fit(train[predictors],train[target])

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(test[target], rf_clf.predict(test[predictors]))
accuracy_score(test[target], rf_clf.predict(test[predictors]))

# Evaluation on Training Data
confusion_matrix(train[target], rf_clf.predict(train[predictors]))
accuracy_score(train[target], rf_clf.predict(train[predictors]))


from sklearn.model_selection import GridSearchCV


rf_clf_grid = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=20)

param_grid = {"max_features": [4, 5, 6, 7, 8, 9, 10], "min_samples_split": [2, 3, 10]}

grid_search = GridSearchCV(rf_clf_grid, param_grid, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(train[predictors], train[target])

grid_search.best_params_

cv_rf_clf_grid = grid_search.best_estimator_


from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(test[target], cv_rf_clf_grid.predict(test[predictors]))
accuracy_score(test[target], cv_rf_clf_grid.predict(test[predictors]))

# Evaluation on Training Data
confusion_matrix(train[target], cv_rf_clf_grid.predict(train[predictors]))
accuracy_score(train[target], cv_rf_clf_grid.predict(train[predictors]))


# RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV

model = DT(criterion = 'entropy')

param_dist = {'min_samples_leaf': list(range(1, 20)),
              'max_depth': list(range(2, 30)),
              'max_features': ['sqrt']}

n_iter = 70

model_random_search = RandomizedSearchCV(estimator = model,
                                         param_distributions = param_dist,
                                         n_iter = n_iter)

model_random_search.fit(train[predictors], train[target])

model_random_search.best_params_

dT_random = model_random_search.best_estimator_

#prediciton on test data 
pred_random = dT_random.predict(test[predictors])
pd.crosstab(test[target], pred_random, rownames=['Actual'], colnames=['Predictions'])

np.mean(pred_random == test[target])

#predicition on train data 
pred_random = dT_random.predict(train[predictors])
pd.crosstab(train[target], pred_random, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(pred_random == train[target])




import altair as alt
import altair.vegalite.v4
#pip install tensorflow

import streamlit as st
import tensorflow as tf

import pickle
model_pkl_file = open("rf_clf.pkl","wb" )
pickle.dump(rf_clf, model_pkl_file)
model_pkl_file.close()
