import initialCleanUp as icu
path='./data/train.csv'
df=icu.cleanUp(path)
df.to_csv('./data/output/trainClean.csv', index=False)