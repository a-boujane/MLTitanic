import initialCleanUp as icu
path='./data/train.csv'
darray=['PassengerId', 'Name', 'Ticket',
                'Fare', 'Cabin', 'SibSp','Parch','Embarked']
df=icu.cleanUp(path,darray)
df.to_csv('./data/output/trainClean.csv', index=False)