# This program reads the output of cleanUp

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
from sklearn.model_selection import KFold 
from sklearn.neural_network import MLPClassifier
import cvalidate as cv

#Prepare the Data

df=pd.read_csv('./data/output/trainClean.csv')

y=df["Survived"].as_matrix()
X=df.drop("Survived",axis=1).as_matrix()

def classify_validate(C,X,y,cvX,cvy,penalty='l1'):
    clf=classify(C,X,y,penalty)
    return cv.validate(clf,cvX,cvy,1)



def classify(C,X=X,y=y,penalty='l1'):
    clf=lm.LogisticRegression(penalty=penalty,C=C,max_iter=10000)
    clf.fit(X,y)
    return clf

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

def nnClassify_validate(X,y,cvX,cvy,alpha):
    clf=nnClassify(X,y,alpha)
    return cv.validate(clf,cvX,cvy,1)


def nnClassify(X=X,y=y,alpha=1e-5):
    clf=MLPClassifier(solver='lbfgs',alpha=alpha,hidden_layer_sizes=(5,2),random_state=1)
    clf.fit(X,y)
    return clf




alpha=np.logspace(1e-30,0.3,50)



kf = KFold(10,shuffle=True);

def crossValidateC(alpha):
    mac=np.array([])
    for train,test in kf.split(X):
        mac=np.append(mac,nnClassify_validate(X[train],y[train],X[test],y[test],alpha))
    print mac
    return mac.mean(),mac.std()
def runCrossValidation():
        
    mean=np.array([])
    std=np.array([])

    for al in alpha:
        a,b=crossValidateC(al)
        mean=np.append(mean,a)
        std=np.append(std,b)

        
    plt.subplot(211)
    plt.title("Mean")
    plt.axis([0.,0.1,0.5,1])
    plt.xscale('log')
    plt.plot(alpha,mean)
    plt.subplot(212)
    plt.title("STD")
    plt.xscale('log')
    plt.plot(alpha,std)

    plt.show()
# np.savetxt("./data/output/csresult",cvresult)
# k=9
# xaxis=[]
# yaxis=[]
# for i in range(0,k-1):
#     start=i*99
#     finish=(i+1)*99-1
#     classify_validate(C,X[start:finish],)
#runCrossValidation()
crossValidateC(1e-30)
