import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time
import sys
import cPickle as pickle

start=time.time()

def save(mean,name="mean"):
    with open("./data/"+name+".pkl",'w') as file:
        pickle.dump(mean,file,protocol=2)

def load(name="mean"):
    with open("./data/"+name+".pkl",'r') as file:
        return pickle.load(file)

def plot(x,y,mean):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    zs = np.array([mean[(x,y)] for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    # ax.plot_wireframe(X, Y, Z)
    ax.plot_surface(X,Y,Z,rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def progress(index,total):
    percent=(100*index)/total;
    elapsed=time.time()-start;
    remaining=elapsed*100/percent-elapsed
    line1="--[%-50s] %.2f%% \n";
    line2="--Elapsed \t: %.0f s -- (%.0f min) -- (%.2f h)\n"
    line3="--Remaining \t: %.0f s -- (%.0f min) -- (%.2f h)"
    upTwoLines="\033[F\033[F"
    sys.stdout.write('\r')
    sys.stdout.write(line1 % ('='*int((percent/2)),percent))
    sys.stdout.write(line2 % (elapsed,elapsed/60,elapsed/3600))
    sys.stdout.write(line3 % (remaining,remaining/60,remaining/3600))
    sys.stdout.write(upTwoLines)
    sys.stdout.flush()
 