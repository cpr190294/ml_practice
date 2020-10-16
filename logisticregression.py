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
