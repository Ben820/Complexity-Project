# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 13:27:30 2021

@author: 44743
"""
"""
Ben Amroota; Complexity Project
CID: 01508466
"""
""" 
IMPORTANT: 
The variable 'data' refers to the Oslo model simulation run for 1.5 million 
grain additions, for 1 realisation. 

The variabel 'dataA' refers to the Oslo model simulation run for 100 thousand 
grain additions, for 10 realisations.
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def Drive():
    num_grains_added[0] = num_grains_added[0] + 1
    heights[0] = heights[0] + 1
    z[0] = z[0] + 1
    s[0] = 0
    s[1] = 0
    s[2] = 0

def Relaxation():
    while any(z > z_thres):
        if z[0] > z_thres[0]: 
            s[0] += 1
            z[0] -= 2
            z[1] += 1
            heights[0] -= 1
            heights[1] += 1
            z_thres[0] = np.random.choice([1,2], p = [prob,1-prob])
        for a in range(1,L-1):
            if z[a] > z_thres[a]: 
                s[1] += 1
                z[a] -= 2
                z[a+1] += 1
                z[a-1] += 1
                heights[a] -= 1
                heights[a+1] += 1
                z_thres[a] = np.random.choice([1,2], p = [prob,1-prob])
        if z[-1] > z_thres[-1]:
            tc[0] = True
            s[2] += 1
            z[-1] -= 1
            z[-2] += 1
            heights[-1] -= 1
            z_thres[-1] = np.random.choice([1,2], p = [prob,1-prob])
    
    # Avalanche size is the total number of relaxations per single grain added
    s_list.append(sum(s))
    height_1.append(heights[0])
    num_grains_pile.append(sum(heights))
    if tc[0] == True:
        # This condition is activated for the first time when the first grain 
        # leaves the system, therefore turns True at the cross-over time
        tc_list.append(sum(heights))


def Algrithm(Iterations):
    for b in range(Iterations):
        Drive()
        Relaxation()
#%%
""" Initialisation """

# Initialised with system size L and probability p 
L = 4
prob = 0.5 

tc = np.array([False])
s = np.array([0,0,0])
z_thres = np.zeros(L)
heights = np.zeros(L)
z = np.zeros(L)
x_pos = np.arange(1,L+1)

num_grains_added = np.array([0])
height_1 = []
num_grains_pile = []
s_list = []
tc_list = [] 

# Assigns a random value between 1 and 2 to each z_i as z_thres
for x in range(0,L):
    z_thres[x] = np.random.choice([1,2], p = [prob,1-prob])
#%%
data = {}
# R-1 is the number of Realisations 
# I is the number of single grain additions (i.e. total time)
R = 2 
I = 1500000
g = np.array([4,8,16,32,64,128,256])
for j in range(len(g)):
    for h in range(1,R):
        Algrithm(I)
        data[g[j],h] = [height_1, num_grains_pile, s_list, tc_list]
        
        # Condition to handle hitting the end of the list h
        if h < R-1:
            L = g[j]
        if h == R-1:
            if j >= len(g)-1:
                L = g[j]
            else:
                L = g[j+1]
        tc = np.array([False])
        s = np.array([0,0,0])
        z_thres = np.zeros(L)
        heights = np.zeros(L)
        z = np.zeros(L)
        x_pos = np.arange(1,L+1)
        
        num_grains_added = np.array([0])
        height_1 = []
        num_grains_pile = []
        s_list = []
        tc_list = [] 
        
        # Resets the threshold slopes after each realisation
        for x in range(0,L):
            z_thres[x] = np.random.choice([1,2], p = [prob,1-prob])
##%%
""" The cross-over time """

t_c_av = [[],[],[],[],[],[],[]]

for e in range(len(g)):
    for q in range(1,R,1):
        t_c_av[e].append(data[g[e],q][3][0])

""" The steady state """
# Creates a new array just including the information in the steady state,
# i.e. after the cross-over time.

dataTC_Complete = {}
List_Hei = []
List_Num = []
List_S = []

for e in range(len(g)):
    for i in range(len(data[128,1][0])):
        if i > t_c_av[e][0]:
            List_Hei.append(data[g[e],1][0][i])
            List_Num.append(data[g[e],1][1][i])
            List_S.append(data[g[e],1][2][i])
            dataTC_Complete[g[e],0] = List_Hei
            dataTC_Complete[g[e],1] = List_Num
            dataTC_Complete[g[e],2] = List_S
    List_Hei = []
    List_Num = []
    List_S = []

""" The average height of the system """

avgH = []
for e in range(len(g)):
    sum_H = sum(dataTC_Complete[g[e],0])
    av_H = sum_H/(I-t_c_av[e][0])
    avgH.append(av_H)

""" Standard Deviation of the heights """

avgHsq = []
for e in range(len(g)):
    sum_Hsq = sum(np.array(dataTC_Complete[g[e],0])**2)
    av_Hsq = sum_Hsq/(I-t_c_av[e][0])
    avgHsq.append(av_Hsq)

""" Height probability P(h;L) """

# Isolates the data in the steady state and calculates the probabilities of 
# the systems height and the height of the system in the steady state 
dataTC = {}
List_Val = []
for e in range(len(g)):
    for i in range(len(data[128,1][0])):
        if i > t_c_av[e][0]:
            List_Val.append(data[g[e],1][0][i])
            dataTC[g[e]] = List_Val
    List_Val = []

Y = np.array([np.size(dataTC[4]), np.size(dataTC[8]), np.size(dataTC[16]), 
              np.size(dataTC[32]), np.size(dataTC[64]), np.size(dataTC[128]),
              np.size(dataTC[256])])

Count = {}
from collections import Counter

for e in range(len(g)):
    Count[g[e],0] = dict(Counter(dataTC[g[e]]))

Prb = Count.copy()
for e in range(len(g)):
    for c in sorted(Count[g[e],0].keys()):
        Prb[g[e],0][c] = Count[g[e],0][c]/Y[e]

# Val is the height of the system and Prob is the height probability in the 
# steady state
Prob = [[],[],[],[],[],[],[]]
Val = [[],[],[],[],[],[],[]]

for e in range(len(g)):
    for r in sorted(Prb[g[e],0].keys()):
        Val[e].append(r)
        Prob[e].append(Prb[g[e],0][r])

# Calculates the average height, average height squared and standard deviation
# in the steady state 
avg_H = np.array(avgH)
avg_H_sq = np.array(avgHsq)
sigma = np.sqrt(avg_H_sq - np.square(avg_H))

# scalex and scaley are the scaling terms on the x and y axis for the data 
# collapse in Task 2g
scalex = [[],[],[],[],[],[],[],[]]
for z in range(7):
    scalex[z].append((Val[z]-avg_H[z])/sigma[z])

scaley = [[],[],[],[],[],[],[],[]]
for k in range(7):
    scaley[k].append(np.array(Prob[k])*sigma[k])

""" Avalanche size probability P(s; L) """

# Isolates the data in the steady state and calculates the probabilities of 
# the avalanche sizes and the avalanche sizes in the steady state- analogous 
# procedure to above

dataTCP = {}
S_list = []
for e in range(len(g)):
    for i in range(len(data[128,1][0])):
        if i > t_c_av[e][0]:
            S_list.append(data[g[e],1][2][i])
            dataTCP[g[e]] = S_list
    S_list = []

from collections import Counter

Count_S = {}
for e in range(len(g)):
    Count_S[g[e]] = dict(Counter(dataTCP[g[e]]))

Y_S = np.array([np.size(dataTCP[4]), np.size(dataTCP[8]), np.size(dataTCP[16]), 
              np.size(dataTCP[32]), np.size(dataTCP[64]), np.size(dataTCP[128]),
              np.size(dataTCP[256])])

Prb_S = Count_S.copy()
for e in range(len(g)):
    for c in sorted(Count_S[g[e]].keys()):
        Prb_S[g[e]][c] = Count_S[g[e]][c]/Y_S[e]

# Val_S is the avalanche size and Prob_S is the avalanche size probability in 
# the steady state
Prob_S = [[],[],[],[],[],[],[]]
Val_S = [[],[],[],[],[],[],[]]

for e in range(len(g)):#7):
    for r in sorted(Prb_S[g[e]].keys()):
        Val_S[e].append(r)
        Prob_S[e].append(Prb_S[g[e]][r])

""" Calculation of kth moment """
import math
K = 10

kth_mom = [[] for x in range(K)]
kth_moment = [[] for x in range(K)]
kth = {}

for e in range(len(g)):
    for k in range(1,K+1):
        s_k = []
        for b in range(len(dataTC_Complete[g[e],2])):
            s_k.append(math.pow(dataTC_Complete[g[e],2][b], k))
        kth_moment[k-1].append((np.sum(s_k))/len(dataTC_Complete[g[e],0]))
#%% # PLOTTING
""" TESTING: p = 0.5; 1st moment (average avalanche size) = L """
plt.figure()

plt.plot(g, kth_moment[0]/g)
plt.plot(g, kth_moment[0]/g, 'x')

plt.xlabel("System size, $L$", size = "15")
plt.ylabel(r' Scaled average avalanche size, $\langle s \rangle /L $', size = "15")
plt.grid()
plt.ylim(1,1)
plt.savefig("Testing 1.png", dpi = 1000)
plt.show()
#%% # PLOTTING 
""" TASK 2a: Height vs Time; Uncollapsed """
plt.figure()

time = np.arange(1, len(data[4,1][0][:100000])+1)

plt.plot(time, data[4,1][0][:100000], '.', label = "L = 4")
plt.plot(time, data[8,1][0][:100000], '.', label = "L = 8")
plt.plot(time, data[16,1][0][:100000], '.', label = "L = 16")
plt.plot(time, data[32,1][0][:100000], '.', label = "L = 32")
plt.plot(time, data[64,1][0][:100000], '.', label = "L = 64")
plt.plot(time, data[128,1][0][:100000], '.', label = "L = 128")
plt.plot(time, data[256,1][0][:100000], '.', label = "L = 256")

plt.xlabel("Time, $t$", size = "15")
plt.ylabel("Height of the pile, $h(t; L)$", size = "15")
plt.xlim(-8000,105000)
#plt.xscale("log")
#plt.yscale("log")
plt.legend()
plt.grid()
#plt.savefig("Task 2a.png", dpi = 1000)
plt.show()
#%% # PLOTTING
""" Task 2b: Average cross-over-time vs System size; Fitted """
plt.figure()

t_c_AV = [np.mean(np.array(t_c_av[0])), np.mean(np.array(t_c_av[1])), 
          np.mean(np.array(t_c_av[2])), np.mean(np.array(t_c_av[3])),
          np.mean(np.array(t_c_av[4])), np.mean(np.array(t_c_av[5])),
          np.mean(np.array(t_c_av[6]))]

def Plaw(x, m, k):
    y = m*(x**k)
    return y

arO = np.arange(4, 256, 0.01)

p0 = np.array([0.8, 2])
p, cov = opt.curve_fit(Plaw, g, t_c_AV, p0)
plt.plot(arO, Plaw(arO, p[0], p[1]), zorder=10,color = 'red')

plt.plot(g, t_c_AV, "x")

plt.xlabel("System size, $L$", size = "15")
plt.ylabel(r'Average cross-over time, $\langle t_c \rangle$', size = "15")
plt.xscale("log")
plt.yscale("log")
plt.grid()
#plt.savefig("Task 2b.png", dpi = 1000)
plt.show()

for c in zip(p, np.sqrt(np.diag(cov))):# zips root of diag of cov matrix with related value in curve fit
    print('%.15f pm %.4g' % (c[0], c[1]))# prints value and uncertainty, f is decimal places and G is sig figs
#%% # PLOTTING
""" Task 2d: Processed Height vs Time; Uncollapsed """
plt.figure()

height_proc_4 = (1/(R-1))*(np.array(dataA[4,1][0]) + np.array(dataA[4,2][0]) + np.array(dataA[4,3][0]) + np.array(dataA[4,4][0]) + np.array(dataA[4,5][0]))
height_proc_8 = (1/(R-1))*(np.array(dataA[8,1][0]) + np.array(dataA[8,2][0]) + np.array(dataA[8,3][0]) + np.array(dataA[8,4][0]) + np.array(dataA[8,5][0]))
height_proc_16 = (1/(R-1))*(np.array(dataA[16,1][0]) + np.array(dataA[16,2][0]) + np.array(dataA[16,3][0]) + np.array(dataA[16,4][0]) + np.array(dataA[16,5][0]))
height_proc_32 = (1/(R-1))*(np.array(dataA[32,1][0]) + np.array(dataA[32,2][0]) + np.array(dataA[32,3][0]) + np.array(dataA[32,4][0]) + np.array(dataA[32,5][0]))
height_proc_64 = (1/(R-1))*(np.array(dataA[64,1][0]) + np.array(dataA[64,2][0]) + np.array(dataA[64,3][0]) + np.array(dataA[64,4][0]) + np.array(dataA[64,5][0]))
height_proc_128 = (1/(R-1))*(np.array(dataA[128,1][0]) + np.array(dataA[128,2][0]) + np.array(dataA[128,3][0]) + np.array(dataA[128,4][0]) + np.array(dataA[128,5][0]))
height_proc_256 = (1/(R-1))*(np.array(dataA[256,1][0]) + np.array(dataA[256,2][0]) + np.array(dataA[256,3][0]) + np.array(dataA[256,4][0]) + np.array(dataA[256,5][0]))

time = np.arange(1, len(height_proc_4)+1)

plt.plot(time, height_proc_4, 'x', label = "L = 4")
plt.plot(time, height_proc_8, 'x', label = "L = 8")
plt.plot(time, height_proc_16, 'x', label = "L = 16")
plt.plot(time, height_proc_32, 'x', label = "L = 32")
plt.plot(time, height_proc_64, 'x', label = "L = 64")
plt.plot(time, height_proc_128, 'x', label = "L = 128")
plt.plot(time, height_proc_256, 'x', label = "L = 256")

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Time, $ t $", size = "15")
plt.ylabel(" $ \~h(t; L) $", size = "15")
plt.legend()
plt.grid()
plt.show()
#%% # PLOTTING
""" Task 2d: Processed Height vs Time; Collapsed """
plt.figure()

height_proc_4 = (1/(R-1))*(np.array(dataA[4,1][0]) + np.array(dataA[4,2][0]) + np.array(dataA[4,3][0]) + np.array(dataA[4,4][0]) + np.array(dataA[4,5][0]) + np.array(dataA[4,6][0]) + np.array(dataA[4,7][0]) + np.array(dataA[4,8][0]) + np.array(dataA[4,9][0]) + np.array(dataA[4,10][0]))
height_proc_8 = (1/(R-1))*(np.array(dataA[8,1][0]) + np.array(dataA[8,2][0]) + np.array(dataA[8,3][0]) + np.array(dataA[8,4][0]) + np.array(dataA[8,5][0]) + np.array(dataA[8,6][0]) + np.array(dataA[8,7][0]) + np.array(dataA[8,8][0]) + np.array(dataA[8,9][0]) + np.array(dataA[8,10][0]))
height_proc_16 = (1/(R-1))*(np.array(dataA[16,1][0]) + np.array(dataA[16,2][0]) + np.array(dataA[16,3][0]) + np.array(dataA[16,4][0]) + np.array(dataA[16,5][0]) + np.array(dataA[16,6][0]) + np.array(dataA[16,7][0]) + np.array(dataA[16,8][0]) + np.array(dataA[16,9][0]) + np.array(dataA[16,10][0]))
height_proc_32 = (1/(R-1))*(np.array(dataA[32,1][0]) + np.array(dataA[32,2][0]) + np.array(dataA[32,3][0]) + np.array(dataA[32,4][0]) + np.array(dataA[32,5][0]) + np.array(dataA[32,6][0]) + np.array(dataA[32,7][0]) + np.array(dataA[32,8][0]) + np.array(dataA[32,9][0]) + np.array(dataA[32,10][0]))
height_proc_64 = (1/(R-1))*(np.array(dataA[64,1][0]) + np.array(dataA[64,2][0]) + np.array(dataA[64,3][0]) + np.array(dataA[64,4][0]) + np.array(dataA[64,5][0]) + np.array(dataA[64,6][0]) + np.array(dataA[64,7][0]) + np.array(dataA[64,8][0]) + np.array(dataA[64,9][0]) + np.array(dataA[64,10][0]))
height_proc_128 = (1/(R-1))*(np.array(dataA[128,1][0]) + np.array(dataA[128,2][0]) + np.array(dataA[128,3][0]) + np.array(dataA[128,4][0]) + np.array(dataA[128,5][0]) + np.array(dataA[128,6][0]) + np.array(dataA[128,7][0]) + np.array(dataA[128,8][0]) + np.array(dataA[128,9][0]) + np.array(dataA[128,10][0]))
height_proc_256 = (1/(R-1))*(np.array(dataA[256,1][0]) + np.array(dataA[256,2][0]) + np.array(dataA[256,3][0]) + np.array(dataA[256,4][0]) + np.array(dataA[256,5][0]) + np.array(dataA[256,6][0]) + np.array(dataA[256,7][0]) + np.array(dataA[256,8][0]) + np.array(dataA[256,9][0]) + np.array(dataA[256,10][0]))

time = np.arange(1, len(height_proc_4)+1)

plt.plot(time/(4**2), height_proc_4/4, label = "L = 4")
plt.plot(time/(8**2), height_proc_8/8, label = "L = 8")
plt.plot(time/(16**2), height_proc_16/16, label = "L = 16")
plt.plot(time/(32**2), height_proc_32/32, label = "L = 32")
plt.plot(time/(64**2), height_proc_64/64, label = "L = 64")
plt.plot(time/(128**2), height_proc_128/128, label = "L = 128")
plt.plot(time/(256**2), height_proc_256/256, label = "L = 256")

plt.xscale("log")
plt.yscale("log")

plt.xlabel("$ t / L^2 $", size = "15")
plt.ylabel("$ \~h(t; L) / L $", size = "15")
plt.legend()
plt.grid()
#plt.savefig("Task 2d.png", dpi = 1000)
plt.show()
#%% # PLOTTING
""" Task 2e: Time averaged height vs System size """

avg_H = np.array(avgH)

plt.figure()

def Corr(x, a0, a1, w1):
    y = a0*x*(1-a1*(x**-w1))
    return y 

arO = np.arange(0, 256, 0.0001)
p0 = np.array([1.7, 1, 0.64])
p, cov = opt.curve_fit(Corr, g[3:], avg_H[3:], p0)
plt.plot(arO, Corr(arO, p[0], p[1], p[2]), zorder=10,color = 'red', 
         label = "First order corrections\n to scaling fit")

plt.plot(g, avg_H, 'x')

plt.xlabel("System size, $L$", size = "15")
plt.ylabel(r'Time averaged height, $\langle h(t; L) \rangle _t $', size = "15")
plt.grid()
plt.legend()
plt.savefig("Task 2e.png", dpi = 1000)
plt.show()

for c in zip(p, np.sqrt(np.diag(cov))): #zips root of diag of cov matrix with related value in curve fit
    print('%.15f pm %.4g' % (c[0], c[1])) #prints value and uncertainty, f is decimal places and G is sig figs
#%% # PLOTTING
""" Task 2f: Standard deviation of the Height vs System Size """
plt.figure()

avg_H = np.array(avgH)
avg_H_sq = np.array(avgHsq)
sigma = np.sqrt(avg_H_sq - np.square(avg_H))

def PLaw(x, m, k):
    y = m*(x**k)
    return y 

arO = np.arange(4, 256, 0.0001)
p0 = np.array([0.2, 0.58])
p, cov = opt.curve_fit(PLaw, g[3:], sigma[3:], p0)
plt.plot(arO, PLaw(arO, p[0], p[1]), zorder=10, color = 'red')

plt.plot(g, sigma, 'x')

plt.xlabel("System size, $L$", size = "14")
plt.ylabel(r'Standard deviation of the height, $\sigma_h(L) $', size = "13")
plt.xscale("log")
plt.yscale("log")
plt.grid()
#plt.savefig("Task 2f.png", dpi = 1000)
plt.show()

for c in zip(p, np.sqrt(np.diag(cov))): #zips root of diag of cov matrix with related value in curve fit
    print('%.15f pm %.4g' % (c[0], c[1])) #prints value and uncertainty, f is decimal places and G is sig figs       
#%% # PLOTTING
""" Task 2g: Probability of height h in system of size L vs System size; Uncollapsed """
plt.figure()

plt.plot(Val[0], Prob[0], label = "L = 4")
plt.plot(Val[1], Prob[1], label = "L = 8")
plt.plot(Val[2], Prob[2], label = "L = 16")
plt.plot(Val[3], Prob[3], label = "L = 32")
plt.plot(Val[4], Prob[4], label = "L = 64")
plt.plot(Val[5], Prob[5], label = "L = 128")
plt.plot(Val[6], Prob[6], label = "L = 256")

plt.xlabel("Height, $h$", size = "15")
plt.ylabel("Height probability, $P(h; L)$", size = "15")
plt.legend()
plt.grid()
#plt.savefig("Task 2g uncol.png", dpi = 1000)
plt.show()
#%% # PLOTTING
""" Task 2g: Probability of height h in system of size L vs System size; Collapsed """
plt.figure()

plt.plot(scalex[0][0], scaley[0][0], 'x', label = "L = 4")
plt.plot(scalex[1][0], scaley[1][0], 'x', label = "L = 8")
plt.plot(scalex[2][0], scaley[2][0], 'x', label = "L = 16")
plt.plot(scalex[3][0], scaley[3][0], 'x', label = "L = 32")
plt.plot(scalex[4][0], scaley[4][0], 'x', label = "L = 64")
plt.plot(scalex[5][0], scaley[5][0], 'x', label = "L = 128")
plt.plot(scalex[6][0], scaley[6][0], 'x', label = "L = 256")

import lmfit as lm
p0G = np.array([1, 0, 1])    # 1st amplitude; 2nd mu (center param); 3rd sigma
#pG, covG = opt.curve_fit(lm.models.gaussian, scalex, scaley, p0G)
x2 = np.arange(-4, 6, 0.00001) 

plt.plot(x2, lm.models.gaussian(x2, p0G[0], p0G[1], p0G[2]), linewidth=2, 
         color = "black", linestyle='dashed', label = "Gaussian fit\n $\mu$ = 0 ;$\sigma$ = 1")

plt.xlabel(r'$ (h - \langle h(t; L) \rangle _t )/\sigma_h(L) $', size = "15")
plt.ylabel("$ P(h; L)\sigma_h(L) $", size = "15")
plt.legend()
plt.grid()
#plt.savefig("Task 2g col.png", dpi = 1000)
plt.show()
#%%
""" Task 3a)a): Avalanche Size Probability vs Avalanche size s  - Unbinned """
plt.figure()

plt.plot(Val_S[6], Prob_S[6], label = "L = 256")
plt.plot(Val_S[5], Prob_S[5], label = "L = 128")
plt.plot(Val_S[4], Prob_S[4], label = "L = 64")
plt.plot(Val_S[3], Prob_S[3], label = "L = 32")
plt.plot(Val_S[2], Prob_S[2], label = "L = 16")
plt.plot(Val_S[1], Prob_S[1], label = "L = 8")
plt.plot(Val_S[0], Prob_S[0], label = "L = 4")

plt.xlabel("Avalanche size, $s$", size = "15")
plt.ylabel("Avalanche size probability, $P(s; L)$", size = "15")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()
#plt.savefig("Task 3a unbin.png", dpi = 1000)
plt.show()
#%% # PLOTTING
""" Task 3a)a): Avalanche Size Probability vs Avalanche size s - Binned using external code """

scale = 1.25
s0 = False # whether or not to include s = 0 avalanches 
 
# Need to first run logbin file - Credit: Max Falkenberg McGillivray
bin_4 = logbin(data[4,1][2],scale, s0)
bin_8 = logbin(data[8,1][2],scale, s0)
bin_16 = logbin(data[16,1][2],scale, s0)
bin_32 = logbin(data[32,1][2],scale, s0)
bin_64 = logbin(data[64,1][2],scale, s0)
bin_128 = logbin(data[128,1][2],scale, s0)
bin_256 = logbin(data[256,1][2],scale, s0)

plt.figure()

plt.plot(bin_4[0], bin_4[1], label = "L = 4")
plt.plot(bin_8[0], bin_8[1], label = "L = 8")
plt.plot(bin_16[0], bin_16[1], label = "L = 16")
plt.plot(bin_32[0], bin_32[1], label = "L = 32")
plt.plot(bin_64[0], bin_64[1], label = "L = 64")
plt.plot(bin_128[0], bin_128[1], label = "L = 128")
plt.plot(bin_256[0], bin_256[1], label = "L = 256")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Avalanche size, $s$", size = "15")
plt.ylabel("Avalanche size probability, $\~P(s; L)$", size = "15")
plt.legend()
plt.grid()
#plt.savefig("Task 3a bin.png", dpi = 1000)
plt.show()
#%%
""" Task 3a)b): Cutoff avalanche size vs System size; Parameter D """
plt.figure()

s_c = np.array([np.amax(bin_4[0]), np.amax(bin_8[0]), np.amax(bin_16[0]), np.amax(bin_32[0]), 
       np.amax(bin_64[0]), np.amax(bin_128[0]), np.amax(bin_256[0])])

def PLaw(x, m, k):
    y = m*(x**k)
    return y 

def Linear(x,m,c):
    y = m*x + c
    return y

arO = np.arange(4, 256, 0.0001) 
p0 = np.array([2.439, 2.1897]) 
p, cov = opt.curve_fit(PLaw, g[3:], s_c[3:], p0)
plt.plot(arO, PLaw(arO, p[0], p[1]), zorder=10, color = 'red')
plt.plot(g, s_c, 'x') 

plt.xscale("log")
plt.yscale("log")
plt.xlabel("System size, $L$", size = "15")
plt.ylabel("Cut-off avalanche size, $s_c(L) $", size = "15")
plt.grid()
plt.savefig("Task 3ab D.png", dpi = 1000)
plt.show()

for c in zip(p, np.sqrt(np.diag(cov))): #zips root of diag of cov matrix with related value in curve fit
    print('%.15f pm %.4g' % (c[0], c[1])) #prints value and uncertainty, f is decimal places and G is sig figs
#%%
""" Task 3a)b): Probability of avalanche size vs Avalanche size; Parameter tau """

plt.figure()

def PLaw(x, m, k):
    y = m*(x**-k)
    return y 

arO = np.arange(100, 10000, 0.01)
p0 = np.array([0.3, 1.5])
xvar = bin_256[0][16:39]
yvar = bin_256[1][16:39]
p, cov = opt.curve_fit(PLaw, xvar, yvar, p0)
plt.plot(arO, PLaw(arO, p[0], p[1]), zorder=10, color = 'red')

plt.plot(bin_256[0], bin_256[1], 'x', label = "L = 256")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Avalanche size, $s$", size = "15")
plt.ylabel("Avalanche size probability, $\~P(s; L)$", size = "15")
plt.legend()
plt.grid()
plt.savefig("Task 3ab tau.png", dpi = 1000)
plt.show()

for c in zip(p, np.sqrt(np.diag(cov))): #zips root of diag of cov matrix with related value in curve fit
    print('%.15f pm %.4g' % (c[0], c[1])) #prints value and uncertainty, f is decimal places and G is sig figs
#%%
""" Task 3a)b): Avalanche Size Probability vs Avalanche size s - Binned; Data Collapse """
plt.figure()

tau = 1.556 # Theoretical based off D
#tau = 1.539 # Experimental Value from plot bin_256[16:39]
D = 2.254 # Experimental Value from plot

y_scl = np.array([np.array(bin_4[0]), np.array(bin_8[0]), np.array(bin_16[0]),
                 np.array(bin_32[0]), np.array(bin_64[0]), np.array(bin_128[0]),
                 np.array(bin_256[0])])**tau

plt.plot(bin_4[0]/ s_c[0], y_scl[0]*bin_4[1], label = "L = 4")
plt.plot(bin_8[0]/ s_c[1], y_scl[1]*bin_8[1], label = "L = 8")
plt.plot(bin_16[0]/ s_c[2], y_scl[2]*bin_16[1], label = "L = 16")
plt.plot(bin_32[0]/ s_c[3], y_scl[3]*bin_32[1], label = "L = 32")
plt.plot(bin_64[0]/ s_c[4], y_scl[4]*bin_64[1], label = "L = 64")
plt.plot(bin_128[0]/ s_c[5], y_scl[5]*bin_128[1], label = "L = 128")
plt.plot(bin_256[0]/ s_c[6], y_scl[6]*bin_256[1], label = "L = 256")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("$s/L^D$", size = "15")
plt.ylabel(r'$ \~P(s; L) s^{\tau_s} $', size = "15")
plt.legend()
plt.grid()
#plt.savefig("Task 3ab Data Collapse.png", dpi = 1000)
plt.show()
#%% # PLOTTING
""" Task 3b: Measuring the kth moment in the steady state vs System size L """
plt.figure()

plt.plot(g, kth_moment[0], label = "k = 1")
plt.plot(g, kth_moment[1], label = "k = 2")
plt.plot(g, kth_moment[2], label = "k = 3")
plt.plot(g, kth_moment[3], label = "k = 4")
plt.plot(g, kth_moment[0],'x')
plt.plot(g, kth_moment[1],'x')
plt.plot(g, kth_moment[2],'x')
plt.plot(g, kth_moment[3],'x')

plt.xscale("log")
plt.yscale("log")
plt.xlabel("System size, $L$", size = "15")
plt.ylabel(r'$k^{th}$ moment of the avalanche size, $\langle s^k \rangle $', size = "15")
plt.grid()
plt.legend()
#plt.savefig("Task 3b1.png", dpi = 1000)
plt.show()
#%% # PLOTTING
""" Task 3b: Measuring the kth moment in the steady state vs System size L: Fitted Functions """
plt.figure()

plt.plot(g, kth_moment[0], 'x', label = "k = 1")
plt.plot(g, kth_moment[1], 'x', label = "k = 2")
plt.plot(g, kth_moment[2], 'x', label = "k = 3")
plt.plot(g, kth_moment[3], 'x', label = "k = 4")

def PLaw(x, m, k):
    y = m*(x**-k)
    return y 

arO = np.arange(4, 256, 0.001)
k_i = np.array([1,2,3,4,5,6])
alpha = D*(1+k_i-tau)

p0 = np.array([0, alpha[0]])
p1 = np.array([0, alpha[1]])
p2 = np.array([1, -alpha[2]])
p3 = np.array([1, -alpha[3]])
p4 = np.array([1, -alpha[4]])
p5 = np.array([1, -alpha[5]])

po0, cov0 = opt.curve_fit(PLaw, g, kth_moment[0], p0)
po1, cov1 = opt.curve_fit(PLaw, g, kth_moment[1], p1)
po2, cov2 = opt.curve_fit(PLaw, g, kth_moment[2], p2)
po3, cov3 = opt.curve_fit(PLaw, g, kth_moment[3], p3)

plt.plot(arO, PLaw(arO, po0[0], po0[1]), zorder=10, color = 'royalblue')
plt.plot(arO, PLaw(arO, po1[0], po1[1]), zorder=10, color = 'orange')
plt.plot(arO, PLaw(arO, po2[0], po2[1]), zorder=10, color = 'forestgreen')
plt.plot(arO, PLaw(arO, po3[0], po3[1]), zorder=10, color = 'crimson')

plt.xscale("log")
plt.yscale("log")
plt.xlabel("System size, $L$", size = "15")
plt.ylabel(r'$k^{th}$ moment of the avalanche size, $\langle s^k \rangle $', size = "15")
plt.grid()
plt.legend()
#plt.savefig("Task 3b Fitted.png", dpi = 1000)
plt.show()
#%% 
""" Task 3b: kth moment; Revealing corrections to scaling """
plt.figure()

def err_alpha(k):
    A = (np.square(1+k-1.53929)+np.square(2.2535315))*0.01
    B = np.sqrt(A)
    return B

def err_scl(alpha, A, L, err_alph): # A is the kth moment of s 
    A = alpha*A*(L**(-alpha-1))*err_alph
    return A

scale_k0 = np.array(g**alpha[0])
scale_k1 = np.array(g**alpha[1])
scale_k2 = np.array(g**alpha[2])
scale_k3 = np.array(g**alpha[3])
scale_k4 = np.array(g**alpha[4])

plt.plot(g, np.array(kth_moment[0])/scale_k0, label = "k = 1")
plt.plot(g, np.array(kth_moment[1])/scale_k1, label = "k = 2")
plt.plot(g, np.array(kth_moment[2])/scale_k2, label = "k = 3")
plt.plot(g, np.array(kth_moment[3])/scale_k3, label = "k = 4")
plt.plot(g, np.array(kth_moment[0])/scale_k0,'x')
plt.plot(g, np.array(kth_moment[1])/scale_k1, 'x')
plt.plot(g, np.array(kth_moment[2])/scale_k2, 'x')
plt.plot(g, np.array(kth_moment[3])/scale_k3, 'x')


y_err0 = err_scl(alpha[0], np.array(kth_moment[0])/scale_k0, g, err_alpha(1))
y_err1 = err_scl(alpha[1], np.array(kth_moment[1])/scale_k1, g, err_alpha(2))
y_err2 = err_scl(alpha[2], np.array(kth_moment[2])/scale_k2, g, err_alpha(3))
y_err3 = err_scl(alpha[3], np.array(kth_moment[3])/scale_k3, g, err_alpha(4))

plt.xlabel("System size, $L$", size = "15")
plt.ylabel(r'$\langle s^k \rangle /L^{D(1+k-\tau_s)} $', size = "15")
plt.grid()
plt.legend()
#plt.savefig("Task 3b Scaling.png", dpi = 1000)
plt.show()

#%%
""" Task 3b: Moment Analysis to determine D and tau from kth moment """
plt.figure()

alph = np.array([-po0[1], -po1[1], -po2[1], -po3[1]])
k_i = np.array([1,2,3,4])

def err(k):
    A = np.square(1+k-1.561)
    B = np.square(2.254)
    C = (A+B)/400
    D = np.sqrt(C)
    return D
    
def Linear(x, m, c):
    y = m*x + c
    return y 

err_a = err(k_i)
arO = np.arange(1, 4, 0.0001)
p0 = np.array([2.25, -1.3])
p, cov = opt.curve_fit(Linear, k_i, alph, p0)
plt.plot(arO, Linear(arO, p[0], p[1]), zorder=10, color = 'red')
plt.errorbar(k_i, alph, yerr = err_a, color = "royalblue", fmt='o', mew=1, ms=0.2, capsize=6)

plt.plot(k_i, alph, 'x')

plt.xlabel("Moment, $k$", size = "15")
plt.ylabel(r'$ D( 1 + k - \tau_s)$', size = "15")
plt.grid()
plt.savefig("Task 3b Moment Analysis.png", dpi = 1000)
plt.show()


print("D estimate:")
print(p[0])
D_est = p[0]

print("tau estimate:")
print(-p[1]/p[0] + 1)
tau_est = -p[1]/p[0] + 1

print("x-intercept:")
print(-p[1]/p[0])
