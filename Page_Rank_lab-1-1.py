
# coding: utf-8

# In[1]:
# Worked with Eric, Davis, Kiana, David, Alan, Muhammad
#imports
import IPython
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from numpy import linalg as LA
import matplotlib.pyplot as plt
import scipy as sp
get_ipython().magic(u'matplotlib inline')

import time
startTime = time.time()


# In[2]:

## moving data from txt file into Adjacency Matrix


# importing, reading file
file = open('links.txt', 'r')

#keeping count of itterations
count = 0
Adj = np.zeros((10748,10748))

#parsing file
for line in file:
    line = line.strip('\n')
    value = line.split(", ")
    r_list = list(map(int, value))
    # col 2 in txt file, col 1 in txt file, col 3 in txt file: col 1 == citing journal, 
    # 2 == cited journal, 3 == num of citations
    Adj[r_list[1]][r_list[0]] = r_list[2]
    count += 1

#diagonalizing Adj matrix
np.fill_diagonal(Adj, 0)        
file.close()


# In[3]:

## Defining Constants: 
ALPHA = 0.85
EPSILON = 0.00001


# In[4]:

#function to normalize matrix, takes matrix and returns it normalzied
def norm1(matrix):    
    #SUM = matrix.sum(axis = 0)
    matrix = matrix / matrix.sum(axis=0)
    #matrix/SUM
    matrix[np.isnan(matrix)] = 0
    return(matrix)


# In[5]:

#normalizing Adj matrix
H = norm1(Adj)
print(H[0,10])


# In[6]:

# takes parameter of a matrix and computes and returns the dangleing node matrix 
def dangleNodes(matrix):
    """This function takes in a matrix and returns a vector of dangeling nodes"""
    index = []
    d= np.zeros(matrix.shape[0])
    index.append(np.where(~matrix.T.any(axis=1))[0])
    for i in index:
        np.put(d, [i], [1])
    return(d)


# In[7]:

# DangelingNode matrix for H
d = dangleNodes(H)
print(d)


# In[8]:

# constructs and returns the matching Article Matrix for matrix for H
def Article():
    """computes article vector"""
    matrix = np.ones(10748)
    total_sum = matrix.sum()
    matrix = matrix.T / total_sum
    return(matrix)


# In[9]:

# Article matrix a for H
a = Article()
print(a)


# In[10]:

# This fucntion takes in a matrix, calculates and returns the coresponding Initial Start Vector
def InitStartVector(matrix):
    length = len(matrix)
    a = np.empty(length)
    a.fill(1/length)
    a = np.matrix(a)
    return(a)


# In[11]:

# This creates the initial start vector for H, however, since there was only one reference per article, pi is the same
# the same vector as a (Article Vector)
pi = InitStartVector(H)
pi = pi.T #this is for vector multiplaction in finding Influence vector 
print(pi.shape)


# In[12]:

# Finding Influence Vector using sparse aproach and initial Start vecotr, Adj matrix, Dangling node, and Article
def findP(pi, H, a, d):
    #Initializing Residual and count
    residual = 2
    count = 1
    # To iterate and check residual
    while (EPSILON < residual):
        pi1 = (ALPHA*H*pi) + ((ALPHA*d@pi + (1-ALPHA)) * a).T
        # L1-norm
        residual = np.linalg.norm((pi1-pi), ord = 1)
        # keeping track of iterations for part C)
        count = count+1
        pi = pi1
        
    print('Number of iterations\n', count)
    return(pi) 


# In[13]:

# Respoding Influence Vector (P)
P = findP(pi, H, a, d)
print(P)


# In[14]:

# Function to calculate EigenFactor EF using Adj Vecotor, and Influence Vector
def Eigenfactor(H, P):
    dot = H @ P
    EF = np.array([dot / np.sum(dot)]) * 100
    return(EF)


# In[15]:


EF = Eigenfactor(H,P)
print(EF)


# In[16]:

# Squeezing EF to eliminate dimensions, makeing list 2
list2 = np.squeeze(EF)

# Printing top 20 Articles for part A)
for i in list2.argsort()[-20:][::-1]:print(str(i) + ":" + str(list2[i]))


# In[17]:
# part B) Time
print(time.time() - startTime)

# There are no editers on the lab machines that would run my code other than juypter notebooks.
# I have printed all of the questions and values here, and also included an html file to my Juypter notbook

#Questions:

# A)                        B) 32.4204 sec           C) Num of iterations 33
#4408:1.44811869068
#4801:1.41271864171
#36610:1.2350345744
#2056:0.679502357161
#6919:0.66487911857
#6667:0.634634841505
#4024:0.577232971674
#36523:0.480815116448
#8930:0.47777264656
#6857:0.439734802299
#5966:0.429717753647
#1995:0.386206520689
#1935:0.3851202634
#3480:0.379577603316
#4598:0.372789008691
#2880:0.330306282718
#3314:0.327507895223
#6569:0.319271668906
#5035:0.316779034882
#31212:0.311257045538
