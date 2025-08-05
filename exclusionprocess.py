import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.linear_model import LinearRegression
#import statsmodels.api as sma
import pandas as pd
import seaborn as sns

def reject_outliers(data):
    """Takes in some data and returns a list of the data with outliers removed.
    Uses the IQR method to determine outliers."""
    Q3=np.percentile(data, 75)
    Q1=np.percentile(data, 25)
    IQR=Q3-Q1
    Up=Q3+(1.5*IQR)
    Low=Q1-(1.5*IQR)
    filtered = [e for e in data if (Low < e < Up)]
    return filtered

def PPP(lam, L):
    """
    1-dimensional Poisson point process of rate lam
    :param lam: Gives the rate of the PPP
    :param L: length of line segment
    :return:
    [0] - Returns a list of the locations of nodes for a 1D PPP
    [1] - no. of nodes
    """
    L = float(L)
    n_nodes = np.random.poisson(lam * L)  # Poisson number of points
    node_locations = sorted(np.random.uniform(0, L, n_nodes))  # Uniformly distributed along the line

    return node_locations, n_nodes

def TDPPP(lam, L, H):
    """
    2-dimensional Poisson point process of rate lam
    :param lam: Gives the rate of the PPP
    :param L: length of segment
    :param H: height of segment
    :return:
    [0] - Returns a list of the locations of nodes for a 2D PPP
    [1] - no. of nodes
    """
    L = float(L)
    H = float(H)
    n_nodes = np.random.poisson(lam * L * H)  # Poisson number of points
    node_X = sorted(np.random.uniform(0, L, n_nodes))
    node_Y = np.random.uniform(0, H, n_nodes)
    return node_X, node_Y, n_nodes

def TwoDProcess(lam, L, H, v):
    x=TDPPP(lam, L, H)
    N=x[-1] #Pulls out the number of nodes
    Locations = x[0] #Pulls out the locations
    
    Times = x[1] #Generates the trigger times for each origin
    Trig2 = [0]*N
    #Fix me, make sure to compare with all other points pls
    j=1
    if Times[1]<Times[0]-abs(Locations[0]-Locations[1])/v:
        Trig2[0]+=1
    if Times[N-2]<Times[N-1]-abs(Locations[N-1]-Locations[N-2])/v:
        Trig2[N-1]+=1    
    while j<N-2:
        if Times[j-1]<Times[j]-abs(Locations[j-1]-Locations[j])/v:
             Trig2[j]+=1
        if Times[j+1]<Times[j]-abs(Locations[j+1]-Locations[j])/v:
             Trig2[j]+=1
        j+=1
        
    Active = [] #This array will be filled with the active points
    k=0
    while k<N: #Create a new version of the locations but only those that are active
        if Trig2[k]==0: 
            Active.append(Locations[k])
        k+=1
    NoActive2=len(Active) #Count the number of active points
    Spacings2=[Active[i+1]-Active[i] for i in range(NoActive2-1)] 
    #This is a collection of the nearest neighbour spacings of active points
    #Claimed complexity O(n^2)
    return Spacings2, NoActive2, N

def AlgoOne(PPP, v):
    """
    This algorithm is a brute force method to find the active points in a 2D Poisson point process.
    :param PPP: A tuple containing the locations, times, and number of points in the PPP
    :param v: The speed of the replication fork in the exclusion process
    :return: A tuple containing the active points, number of active points, and total number of points
    """
    Locations, Times, N = PPP
    #Pulls out the number of nodes
    Locations = list(Locations) #Pulls out the locations
    Times = list(Times) #Generates the trigger times for each origin
    Trig2 = [0]*N

    j=0 
    while j<N:
        i=0
        while i<N:
            if Times[i]<Times[j]-abs(Locations[i]-Locations[j])/v:
                Trig2[j]+=1
                break
            i+=1
        j+=1
        
    Active = [] #This array will be filled with the active points
    k=0
    while k<len(Trig2): #Create a new version of the locations but only those that are active
        if Trig2[k]==0: 
            Active.append(Locations[k])
        k+=1
    NoActive2=len(Active) #Count the number of active points
    #Spacings2=[Active[i+1]-Active[i] for i in range(NoActive2-1)] 
    #This is a collection of the nearest neighbour spacings of active points

    return Active, NoActive2, N

def AlgoTwo(PPP, v):
    """
    This algorithm is a more efficient method to find the active points in a 2D Poisson point process.
    It uses a greedy approach to find the lowest point and checks if it is active.
    Loop through remaining active points
    For each procedural lowest active point, check if it is made passive by any of the active points below it
    If not it's active. Add it to the active list.
    Terminate when you run out of active points.
    :param PPP: A tuple containing the locations, times, and number of points in the PPP
    :param v: The speed of the replication fork in the exclusion process
    :return: A tuple containing the active points, number of active points, and total number of points
    """
    Locations, Times, N = PPP
    #Pulls out the number of nodes
    Locations = list(Locations) #Pulls out the locations
    Times = list(Times) #Generates the trigger times for each origin

    #Find the lowest point by y coordinate
    #Lowest point is active 
    y=min(Times)
    x=Locations[Times.index(y)]

    Times.remove(y)
    Locations.remove(x)

    Active = [] #This array will be filled with the active points
    AccTimes=[]

    Active.append(x)
    AccTimes.append(y)

    while len(Times)>0:
        y=min(Times)
        x=Locations[Times.index(y)]
        j=0
        ACount=0
        while j<len(Active):
            if AccTimes[j]<y-abs(Active[j]-x)/v:
                ACount+=1
                break
            j+=1
        if ACount==0:
            Active.append(x)
            AccTimes.append(y)
        #print(f"Counter ={ACount}")
        Times.remove(y)
        Locations.remove(x)
    
    NoActive2=len(Active) #Count the number of active points
    #Spacings2=[Active[i+1]-Active[i] for i in range(NoActive2-1)] 
    return Active, NoActive2, N
