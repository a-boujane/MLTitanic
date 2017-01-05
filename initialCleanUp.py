# This program is the main program that reads the data
# and cleans it up

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def cleanUp(path):
    # Reading the Data
    df = pd.read_csv(path)
    n = len(df)
    # Dropping irrelevant information
    df = df.drop(['PassengerId', 'Name', 'Ticket',
                'Fare', 'Cabin', 'Parch','SibSp','Embarked'], axis=1)
    
    df.Sex = df.Sex.map({'female': 0, 'male': 1}).astype(int)
    mAge = df["Age"].mean()
    df["Age"] = df["Age"].fillna(np.random.normal(loc=mAge, scale=5))

    return df;



