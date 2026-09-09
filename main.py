import pandas as pd
import numpy as np

df=pd.read_csv("Titanic-Dataset - Titanic-Dataset.csv")

df.head()

df.tail()

df.info()

df.shape

df.isnull().sum()

df['Survived'].value_counts()

df['SibSp'].value_counts()

df['Parch'].value_counts()

df.duplicated()

df['Embarked']=df['Embarked'].fillna(df['Embarked'].ffill())

df.isnull().sum()

df['Embarked'].isna().sum()

df['Gender']=df["Gender"].map({'male':0, 'female':1})

df['Embarked']=df['Embarked'].map({'S':0,'C':1,'Q':2})

df.info()

x=df[['Pclass','Gender','SibSp','Parch','Embarked']]

y=df['Survived']

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

default_model = DecisionTreeClassifier(random_state=42)

default_model.fit(x_train,y_train)

y_pred_default= default_model.predict(x_test)

print(y_pred_default)

from sklearn.metrics import accuracy_score,confusion_matrix,r2_score

default_acc=accuracy_score(y_test,y_pred_default)

print(default_acc)

default_confusion = confusion_matrix(y_test,y_pred_default)

print(default_confusion)

"""dtc parameters that can be tuned
1 ) criterian (possible values)  (gini,entropy)
2) max_depth  (max depth) (2 ,3,4, all possibilites  can be single , none = default= max and multiple)
3) min _samples_split (2,8,,10)
4) min_samples_leaf(2,5,6) remove the less leaf samples value

GridSearchCV
"""

grid_param={
    'criterion':['gini','entropy'],
     'max_depth': [3,5,7, None],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,3,5]
    }

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=grid_param,
    cv=5,
    scoring='accuracy')

grid_search.fit(x_train, y_train)

from sklearn.metrics import accuracy_score



print(grid_search.best_params_)

tuned_model=grid_search.best_estimator_

print(tuned_model)

y_pred_tuned=tuned_model.predict(x_test)
print(y_pred_tuned)

tuned_acc=accuracy_score(y_test,y_pred_tuned)
print(tuned_acc)

