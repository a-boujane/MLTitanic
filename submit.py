import vis

df=vis.pd.read_csv("./data/gendermodel.csv")
df=df.drop("Survived",axis=1);
df["Survived"]=vis.hy
df.to_csv("./data/output/submit.csv",index=False)