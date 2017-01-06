import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#takes a trained classifier and an X and y, and a distance (norme) and measures accuracy
def validate(clf,cvX,cvy,norm=1):
    inter=np.sum(np.power(np.abs(clf.predict(cvX)-cvy),norm))
    # return inter
    return 1-np.power(inter,1./norm)/cvy.size;
