# This program reads the output of cleanUp

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
from sklearn.model_selection import KFold 
import cvalidate as cv

#Prepare the Data

df=pd.read_csv('./data/output/trainClean.csv')

y=df["Survived"].as_matrix()
X=df.drop("Survived",axis=1).as_matrix()

def classify(C,X,y,cvX,cvy):
    clf=lm.LogisticRegression(penalty='l2',C=C,max_iter=10000)
    clf.fit(X,y)
    return cv.validate(clf,cvX,cvy,1)
# print y.shape
# hy=clf.predict(sc.getX());
# print hy
# print hy.shape
# print clf.score(sc.getX(),sc.getY());
# print sc.getY().shape

# fuckups=np.abs(sc.getY()-hy)
# fuckups=fuckups.sum()/2
# print fuckups
# print df.info()

C=np.linspace(0.5,1.5,1000)



kf = KFold(10,shuffle=True);

def crossValidateC(C):
    mac=np.array([])
    for train,test in kf.split(X):
        mac=np.append(mac,classify(C,X[train],y[train],X[test],y[test]))
    return mac.mean(),mac.std()

mean=np.array([])
std=np.array([])

for c in C:
    a,b=crossValidateC(c)
    mean=np.append(mean,a)
    std=np.append(std,b)
    if c in [1,2,3,4,5,6,7,8,9]:
        print c


plt.subplot(211)
plt.title("Mean")
plt.plot(C,mean)
plt.subplot(212)
plt.title("STD")
plt.plot(C,std)

plt.show()
# np.savetxt("./data/output/csresult",cvresult)
# k=9
# xaxis=[]
# yaxis=[]
# for i in range(0,k-1):
#     start=i*99
#     finish=(i+1)*99-1
#     classify(C,X[start:finish],)