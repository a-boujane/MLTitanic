# This program is the main program that reads the data
# and cleans it up

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Reading the Data
df = pd.read_csv('./data/train.csv')
n = len(df)
# Dropping irrelevant information
df = df.drop(['PassengerId', 'Name', 'Ticket',
              'Fare', 'Cabin', 'Parch'], axis=1)

# Mapping Sex: Female to 0 and Male to 1
df.Sex = df.Sex.map({'female': 0, 'male': 1}).astype(int)
# Converting Embarked to floats and filling up NAs with a uniform distribution
df["Embarked"] = df["Embarked"].map({'S': 0, 'C': 1})
df["Embarked"] = df["Embarked"].fillna(np.random.randint(0, high=2))

# using a Standard Deviation of 5 anda mean of mAge:
# Filling up NAs in the Age (Gaussian Distribution)
mAge = df["Age"].mean()
df["Age"] = df["Age"].fillna(np.random.normal(loc=mAge, scale=5))

#At this point, we shouldn't have any NAs left in the Data
print df.describe()


# df['Embarked]'].where(df['Embarked'].isnull(),np.random.randint(0,high=2),axis='columns')
# Converting Embarked S to 0 and C to 1
# df["Embarked"].isnull=0
# df["Embarked"]=df["Embarked"].map({'S':0,'C':1}).astype(int)
# Deleting the Name Column

# To BE RE_ENABLED
# df0=df.loc[df['Survived']==0]
# df1=df.loc[df['Survived']==1]
# print df.info()

# f=open('./data/output/trainClean.csv','wb')
# f.close()

# df.to_csv('./data/output/trainClean.csv', index=False)
