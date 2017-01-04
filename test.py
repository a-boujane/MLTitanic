import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.DataFrame(np.random.normal(loc=30,scale=5,size=100000))
df.hist(bins=100)
plt.show()