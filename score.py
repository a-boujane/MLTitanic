import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
import initialCleanUp as icu

def getX():
    darray=['PassengerId', 'Name', 'Ticket',
                'Fare', 'Cabin','SibSp','Parch','Embarked']
    X=icu.cleanUp("./data/test.csv",darray).as_matrix()
    return X

def getY():
    df=pd.read_csv("./data/gendermodel.csv")
    y=df["Survived"].as_matrix()
    return y;