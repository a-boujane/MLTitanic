# This program is the main program that reads the data
# and cleans it up

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


def cleanUp(path, array):
    # Reading the Data
    df = pd.read_csv(path)
    # Dropping irrelevant information
    df = df.drop(array, axis=1)
    
    df.Sex = df.Sex.map({'female': 0, 'male': 1}).astype(int)
    mAge = df["Age"].mean()
    df["Age"] = df["Age"].fillna(np.random.normal(loc=mAge, scale=5))
    min_max_scaler = preprocessing.MinMaxScaler()
    df["Age"]=min_max_scaler.fit_transform(df["Age"])
    return df;



