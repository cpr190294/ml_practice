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


