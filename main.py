#This program is the main program that reads the data
# and cleans it up 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df= pd.read_csv('./data/train.csv')
n=len(df)
# Converting Female to 0 and Male to 1
df=df.drop(['PassengerId','Name','Ticket','Fare','Cabin'],axis=1)

df.Sex=df.Sex.map({'female':0,'male':1}).astype(int)
print '''Before'''
print "*************************************"
print df.loc[df["Embarked"].isnull()]
print "*************************************"
print "\n"
print "\n"
print "\n"
df["Embarked"]=df["Embarked"].fillna(np.random.randint(0,high=2))
print '''AFTER'''
print "*************************************"
print df.loc[df["Embarked"].isnull()]
print "*************************************"

print "\n"
print "\n"
print "\n"
df["Embarked"]=df["Embarked"].map({'S':0,'C':1}).astype(int)














# df['Embarked]'].where(df['Embarked'].isnull(),np.random.randint(0,high=2),axis='columns')
# Converting Embarked S to 0 and C to 1
# df["Embarked"].isnull=0
# df["Embarked"]=df["Embarked"].map({'S':0,'C':1}).astype(int)
# Deleting the Name Column

#To BE RE_ENABLED
# df0=df.loc[df['Survived']==0]
# df1=df.loc[df['Survived']==1]
# print df.info()

# f=open('./data/output/trainClean.csv','wb')
# f.close()

# df.to_csv('./data/output/trainClean.csv', index=False)

