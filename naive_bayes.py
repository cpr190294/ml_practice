from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import numpy as np
data = fetch_20newsgroups()
#print(data.target_names)
categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 
              'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
              'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 
              'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

train = fetch_20newsgroups(subset='train',categories=categories)
test = fetch_20newsgroups(subset='test',categories=categories)
#print(train.data[5])
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
model = make_pipeline(TfidfVectorizer(),MultinomialNB())
#print(train.data[5])
#print(train.target)
model.fit(train.data,train.target)
#model.predict(test.data)

def predict_category(s,train=train,model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]
#print(predict_category('Jesus_christ'))
#print(predict_category('save this car'))
#print(predict_category('nasa'))


#2nd problem using gaussianNB
from sklearn.naive_bayes import GaussianNB



weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
we = le.fit_transform(weather)
print(we)
tmp = le.fit_transform(temp)
#print(tmp)
ply = le.fit_transform(play)
p = ply.reshape(1,-1)
features = zip(we,tmp)
print(' featutres :  ',features)
f = features
model = GaussianNB()
model.fit(f,p)
predicted = model.predict([[0,2]])
print('predicted : ',predicted)





