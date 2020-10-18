from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv('C:/Users/38745/Documents/TEST_DATA/iris.csv')
df.head()
x = df.iloc[:,:-1]
y = df.iloc[:,4]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
c = KNeighborsClassifier(n_neighbors=5)
c.fit(x_train,y_train)
y_pred = c.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
