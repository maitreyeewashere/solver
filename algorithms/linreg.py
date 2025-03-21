import numpy as np
import matplotlib.pyplot as plt
import os

def estimate(x,y):

    if len(x) == 0 or len(y) == 0:
        return None
    
    n = np.size(x)

    mx = np.mean(x)
    my = np.mean(y)

    ss_xy = np.sum(x*y) - n*my*mx
    ss_xx = np.sum(x*x) - n*mx*mx
    
    b1 = ss_xy/ss_xx
    b0 = my - b1*mx

    return (b0,b1)



def regression(x,y):
    x = np.array(x)
    y = np.array(y)

    
    b = estimate(x,y)
    #print(b)
    return b


    