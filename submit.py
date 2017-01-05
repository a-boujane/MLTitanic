import pandas as pd 
import numpy as np


def submit(hy):
    df=pd.read_csv("./data/gendermodel.csv")
    df=df.drop("Survived",axis=1);
    df["Survived"]=hy
    df.to_csv("./data/output/submit.csv",index=False)