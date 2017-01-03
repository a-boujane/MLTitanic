import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df= pd.read_csv('./data/train.csv')

# Converting Female to 0 and Male to 1
df.Sex=df.Sex.map({'female':0,'male':1}).astype(int)
# print df.Sex.info()
# Deleting the Name Column
df0=df.loc[df['Survived']==0]
df1=df.loc[df['Survived']==1]
print df.info()
df=df.drop(['PassengerId','Pclass','Name','Ticket','Fare','Cabin'],axis=1)

# f=open('./data/output/trainClean.csv','wb')
# f.close()

df.to_csv('./data/output/trainClean.csv', index=False)
