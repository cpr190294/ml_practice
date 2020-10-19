import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
c = load_breast_cancer()
df = pd.DataFrame(c['data'],columns=c['feature_names'])
df.head()

s = StandardScaler()
s.fit(df)
sdata = s.transform(df)
pca = PCA(n_components=3)
pca.fit(sdata)
x = pca.transform(sdata)
x.shape
#df.shape

