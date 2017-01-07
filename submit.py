import pandas as pd 
import numpy as np
import vis as vs
import score as sc

def submit(hy):
    df=pd.read_csv("./data/gendermodel.csv")
    df=df.drop("Survived",axis=1);
    df["Survived"]=hy
    df.to_csv("./data/output/submit.csv",index=False)

clf = vs.classify(C=0.9,penalty='l2')
submit(clf.predict(sc.getX()))