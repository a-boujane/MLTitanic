# This program reads the output of cleanUp

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
from sklearn.model_selection import KFold 
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import cvalidate as cv
import pprint

#Prepare the Data

df=pd.read_csv('./data/output/trainClean.csv')

y=df["Survived"].as_matrix()
X=df.drop("Survived",axis=1).as_matrix()


'''
Logistic Regression classification and validation methods
'''


def classify_validate(X,y,cvX,cvy,C,penalty='l1'):
    clf=classify(X,y,C,penalty)
    return cv.validate(clf,cvX,cvy,1)



def classify(X=X,y=y,C=1.5,penalty='l1'):
    clf=lm.LogisticRegression(penalty=penalty,C=C,max_iter=10000)
    clf.fit(X,y)
    return clf
'''
**********************************
'''

'''
Neural Networks classification and validation methods
'''

def nnClassify_validate(X,y,cvX,cvy,i,j):
    clf=nnClassify(X,y,i,j)
    return cv.validate(clf,cvX,cvy,1)


def nnClassify(X=X,y=y,i=50,j=2,alpha=1e-5):
    clf=MLPClassifier(solver='lbfgs',alpha=alpha,hidden_layer_sizes=(i,j),random_state=1)
    clf.fit(X,y)
    return clf

'''
**********************************
'''

'''
Random Forest Classifier
'''

def rfClassify_validate(X,y,cvX,cvy,n_estimators):
    clf=nnClassify(X,y,n_estimators)
    return cv.validate(clf,cvX,cvy,1)


def rfClaffisy(X=X,y=y,n_estimators=10):
    clf=RandomForestClassifier(n_estimators=n_estimators,n_jobs=-1)
    clf.fit(X,y)
    return clf
'''
***********************************
'''

alpha=np.logspace(1e-30,0.3,50)




def crossValidate(method,*args):
    # print "Using %s for classification" % method.func_name
    mac=np.array([])
    kf = KFold(10,shuffle=True);
    for train,test in kf.split(X):
        mac=np.append(mac,method(X[train],y[train],X[test],y[test],*args))
    # print mac
    return (mac.mean(),mac.std())

def runCrossValidation(method,*param):
        
    mean={}
    # std=np.array([])

    for par1 in param[0]:
        for par2 in param[1]:
            mean[(par1,par2)]=crossValidate(method,par1,par2)
            print (par1,par2) , mean[(par1,par2)]
            # mean=np.append(mean,a)
            # std=np.append(std,b)
        
    # for par1 in param[0]:
    #     mean[(par1,)]=crossValidate(method,par1)
    #     print (par1,) , mean[(par1,)]
    #         # mean=np.append(mean,a)
    #         # std=np.append(std,b)
        

    
    # plt.subplot(211)
    # plt.title("Mean")
    # plt.plot()
    # plt.subplot(212)
    # plt.title("STD")
    # plt.plot(paramAxes3D.plot_wireframe(X, Y, Z[0],std)
    # Axes3D.plot_wireframe(X, Y, Z
    # plt.show()
    return mean

# crossValidate(classify_validate,0.1,'l1')
# crossValidate(nnClassify_validate,1e-5)
# pprint.pprint(runCrossValidation(rfClassify_validate,np.arange(17,18)))

i=np.arange(5,25,1)
j=np.arange(5,30,4)


# pprint.pprint(runCrossValidation(nnClassify_validate,i,j))
'''
For Logistic Regression: C=0.23
'''
C=np.linspace(0.01,10,100)
# pprint.pprint(runCrossValidation(classify_validate,C))

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



# np.savetxt("./data/output/csresult",cvresult)
# k=9
# xaxis=[]
# yaxis=[]
# for i in range(0,k-1):
#     start=i*99
#     finish=(i+1)*99-1
#     classify_validate(C,X[start:finish],)
#runCrossValidation()