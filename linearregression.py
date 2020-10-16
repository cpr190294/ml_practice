#LinearRegression example 1 - 
#y = mx + c -->
# ----- y = y axis, x = x axis, m = sploe of the line, c = y inercept of the lone -----
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


#2nd problem of LinearRegression
import numpy as np
import pandas as pd
df = pd.read_csv('C:/Users/38745/Documents/TEST_DATA/Realestate.csv')
df.head()
x = df.iloc[:,:-1]
y = df.iloc[:,7]
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
r = LinearRegression()
r.fit(x_train,y_train)
y_pred = r.predict(x_test)
df = pd.DataFrame({'Actual = ':y_test,'Prediction : ':y_pred})
df

#multi linear regression
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

sm = {'Year': [2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016],
                'Month': [12, 11,10,9,8,7,6,5,4,3,2,1,12,11,10,9,8,7,6,5,4,3,2,1],
                'Interest_Rate': [2.75,2.5,2.5,2.5,2.5,2.5,2.5,2.25,2.25,2.25,2,2,2,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75],
                'Unemployment_Rate': [5.3,5.3,5.3,5.3,5.4,5.6,5.5,5.5,5.5,5.6,5.7,5.9,6,5.9,5.8,6.1,6.2,6.1,6.1,6.1,5.9,6.2,6.2,6.1],
                'Stock_Index_Price': [1464,1394,1357,1293,1256,1254,1234,1195,1159,1167,1130,1075,1047,965,943,958,971,949,884,866,876,822,704,719]        
                }
df = pd.DataFrame(sm)
plt.scatter(df['Interest_Rate'],df['Stock_Index_Price'],color='red')
plt.show()
#df.head()
x = df[['Interest_Rate','Unemployment_Rate']]
y=df['Stock_Index_Price']
r = LinearRegression()
r.fit(x,y)
print("this is prediction  --  :  ",r.predict([[2.5,5.3]]))



