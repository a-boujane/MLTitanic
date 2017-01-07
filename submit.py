import pandas as pd 
import numpy as np
import vis as vs
import score as sc

def submit(hy):
    df=pd.read_csv("./data/gendermodel.csv")
    df=df.drop("Survived",axis=1);
    df["Survived"]=hy
    df.to_csv("./data/output/submit.csv",index=False)

clf = vs.nnClassify(i=5,j=28)
# clf = vs.classify(C=1.5)
# clf=vs.rfClaffisy(n_estimators=17)
submit(clf.predict(sc.getX()))