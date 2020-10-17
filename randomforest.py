import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
print(np.random.seed(0))
iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
#df.head()
df['species'] = pd.Categorical.from_codes(iris.target,iris.target_names)
df.head()
df['is_train'] = np.random.uniform(0,1,len(df))<=.75
df.head()
#len(df)
train,test = df[df['is_train']==True], df[df['is_train']==False]
print(len(train))
print(len(test))

#df.head()
features = df.columns[:4]
#print(features)
y = pd.factorize(train['species'])
#print(y)
z = y[0]
#print('---------------------------')
#print(z)
r = RandomForestClassifier()
r.fit(train[features],z)
q = r.predict(test[features])



