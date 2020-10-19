import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('C:/Users/38745/Documents/TEST_DATA/train.csv')
test = pd.read_csv('C:/Users/38745/Documents/TEST_DATA/test.csv')
#train.head()
#test.head()
print(train.columns.values)
#train.isnull().head()
train.fillna(train.mean(),inplace=True)
test.fillna(test.mean(),inplace=True)
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Sex', 'Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived', ascending=False)
train = train.drop(['Name','Ticket','Cabin','Embarked'],axis=1)
test = test.drop(['Name','Ticket','Cabin','Embarked'],axis=1)
le = LabelEncoder()
le.fit(train['Sex'])
le.fit(test['Sex'])
train['Sex'] = le.transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])
x = np.array(train.drop(['Survived'],1).astype(float))
y =  np.array(train['Survived'])

km = KMeans(n_clusters=2)
km.fit(x)
correct = 0
#print(x[2])
#predict_me = np.array(x[2].astype(float)),
#print('length',len(predict_me))
#predict_me = predict_me.reshape(-1, len(predict_me))
print(predict_me)
for i in range(len(x)):
    predict_me = np.array(x[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = km.predict(predict_me)
    #print('prediction   :  ',prediction)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(x))


