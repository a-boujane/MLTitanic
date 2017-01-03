# This program read the output of main.py.
# It is used for validation only

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('./data/output/trainClean.csv')
df0=df.loc[df['Survived']==0]
df1=df.loc[df['Survived']==1]

print df.info()