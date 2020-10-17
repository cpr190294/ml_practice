import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv('C:/Users/38745/Documents/TEST_DATA/diabetes.csv')

x = df.iloc[:,:-1]
y = df.iloc[:,8]

x_train,x_test,y_train,y_test = train_test_split(x,y)

#r = DecisionTreeClassifier()
r = DecisionTreeClassifier(criterion='entropy',max_depth=3)
r.fit(x_train,y_train)
y_pred = r.predict(x_test)
print("Accuracy :  ",metrics.accuracy_score(y_test,y_pred))
