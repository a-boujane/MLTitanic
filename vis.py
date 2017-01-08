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
import utl


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

def nnClassify_validate(X,y,cvX,cvy,alpha):
    clf=nnClassify(X,y,alpha)
    return cv.validate(clf,cvX,cvy,1)


def nnClassify(X=X,y=y,alpha=1.e-14,i=4,j=15):
    clf=MLPClassifier(solver='lbfgs',alpha=alpha,hidden_layer_sizes=(i,j),random_state=1,activation="logistic")
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




def runSaved(boo,name="mean"):
    if boo:
        mean=utl.load(name)
    else:
        mean=runCrossValidation(nnClassify_validate,alpha)
        utl.save(mean,name)
    return mean

def crossValidate(method,*args):
    # print "Using %s for classification" % method.func_name
    mac=np.array([])
    kf = KFold(10,shuffle=True);
    for train,test in kf.split(X):
        mac=np.append(mac,method(X[train],y[train],X[test],y[test],*args))
    # print mac
    return mac.mean()

def runCrossValidation(method,*param):
        
    mean={}
    index=0.;
    # std=np.array([])
    

    # total=param[1].size*param[0].size
    # for par1 in param[0]:
    #     for par2 in param[1]:
    #         index+=1;
           
    #         mean[(par1,par2)]=crossValidate(method,par1,par2)
    #         utl.progress(index,total)
    #         # print (par1,par2) , mean[(par1,par2)]
            
    #         # mean=np.append(mean,a)
    #         # std=np.append(std,b)


    total=param[0].size    
    for par1 in param[0]:
        index+=1
        mean[(par1,)]=crossValidate(method,par1)
        utl.progress(index,total)
        # print (par1,) , mean[(par1,)]
            # mean=np.append(mean,a)
            # std=np.append(std,b)
        


    print ""
    return mean

# crossValidate(classify_validate,0.1,'l1')
# crossValidate(nnClassify_validate,1e-5)
# pprint.pprint(runCrossValidation(rfClassify_validate,np.arange(17,18)))


alpha=np.logspace(-50,1,3000)



# mean=runSaved(False,"AlphaMean")
# utl.plot2d(alpha,mean)

# print alpha
# i=np.arange(2,40,1)
# j=np.arange(2,40,1)
# pprint.pprint(mean)
# utl.plot3d(i,j,mean)


'''
For Logistic Regression: C=0.23
'''
C=np.linspace(0.01,10,100)
# pprint.pprint(runCrossValidation(classify_validate,C))

