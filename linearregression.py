#LinearRegression example 1 - 
#code below
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
s = pd.read_csv('C:Documents/TEST_DATA/test.csv')
s.head()
s.shape
s.plot(x='exp',y='salary',style='o')
plt.show()
X = s.iloc[:, :-1].values
Y = s.iloc[:,2].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
r = LinearRegression()
r.fit(X_train,Y_train)
y_pred = r.predict(X_test)
df = pd.DataFrame({"Actual":Y_test, "Predict":y_pred})
