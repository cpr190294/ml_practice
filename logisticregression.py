#LogisticRegression First ExaMPLE
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot  as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
df = pd.read_csv('C:/Users/38745/Documents/TEST_DATA/bank.csv',header=0)
df = df.dropna()
df['education'] = np.where(df['education']=='basic.4y','Basics',df['education'])
df['education'] = np.where(df['education']=='basic.9y','Basics',df['education'])
df['education'] = np.where(df['education']=='basic.6y','Basics',df['education'])
df.head()
df['y'].value_counts()
non_sub = len(df['y']==0)
sub = len(df['y']==1)
per_sub = sub/sub+non_sub
per_nonsub = non_sub/sub+non_sub

#df.head()
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    #print("cat_list : ",cat_list)
    cat_list = pd.get_dummies(df[var], prefix=var)
    #print("cat_list 2  : ",cat_list)
    #print("cat_list columns  :  ",cat_list.columns)
    #print("df columns   :   ",df.columns)
    data1=df.join(cat_list)
    df=data1
    #print(df)
df.head()
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
d_v = df.columns.tolist()
print(d_v)
to_keep = [i for i in d_v if i not in cat_vars]
df = df[to_keep]
x = df.iloc[:,:-1]
y = df.iloc[:,20]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
#print(cat_list)
r = LogisticRegression()
r.fit(x_train,y_train)
y_pred = r.predict(x_test)
#print(pd.DataFrame({'Actual':y_test,'prection' : y_pred}))
from sklearn.metrics import precision_score
print(precision_score(y_test, y_pred))

#Logistic Regression 2nd example
import seaborn as sb
import pandas as pd
import numpy as np
import sklearn.linear_model as ln
from sklearn.model_selection import train_test_split
from sklearn import metrics

can = {'gmat': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
              'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
              'work_experience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
              'admitted': [1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
              }

df = pd.DataFrame(can)
df.head()
x = df.iloc[:,:-1]
y = df.iloc[:,3]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
r = ln.LogisticRegression()
r.fit(x_train,y_train)
y_pred = r.predict(np.array([770,3.4,3]).reshape(1,-1))
#print(pd.DataFrame({'Actual :':y_test,'Prediction :':y_pred}))
print(y_pred)
print(np.array([[780,4,3]]))
