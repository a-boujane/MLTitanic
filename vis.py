# This program read the output of main.py.
# It is used for validation only

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
import score as sc

df=pd.read_csv('./data/output/trainClean.csv')
df0=df.loc[df['Survived']==0]
df1=df.loc[df['Survived']==1]
# print df

y=df["Survived"].as_matrix()
X=df.drop("Survived",axis=1).as_matrix()
clf=lm.LogisticRegression(penalty='l2',C=0.1,max_iter=10000)

clf.fit(X,y)

print y.shape
hy=clf.predict(sc.getX());
# print hy
print hy.shape
# print clf.score(sc.getX(),sc.getY());
print sc.getY().shape

fuckups=np.abs(sc.getY()-hy)
fuckups=fuckups.sum()/2
print fuckups
# print df.info()


