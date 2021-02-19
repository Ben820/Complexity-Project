# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 12:12:17 2021

@author: 44743
"""
"""
Complexity and Networks Lab """

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
    
    s_list.append(sum(s))
    height_1.append(heights[0])
    num_grains_pile.append(sum(heights))
    if tc[0] == True:
        tc_list.append(sum(heights))


def Algrithm(Iterations):
    for b in range(Iterations):
        Drive()
        Relaxation()

""" Initialisation """

# System size L
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

for x in range(0,L):
    z_thres[x] = np.random.choice([1,2], p = [prob,1-prob])

data = {}
R = 11 # Number of Realisations is R-1
I = 100000
g = np.array([4,8,16,32,64,128,256])
for j in range(len(g)):
    for h in range(1,R):
        Algrithm(I)
        data[g[j],h] = [height_1, num_grains_pile, s_list, tc_list]
        
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
        
        # Initialisation
        for x in range(0,L):
            #z_thres[x] = np.random.randint(1,3)
            z_thres[x] = np.random.choice([1,2], p = [prob,1-prob])
#%%
""" Task 2b """

R = 2 # Number of Realisations is R-1
I = 1500000
g = np.array([4,8,16,32,64,128,256])
t_c_av = [[],[],[],[],[],[],[]]

for e in range(len(g)):
    for q in range(1,R,1):
        t_c_av[e].append(data[g[e],q][3][0])
##%%
#t_c_av = []
#t_c_list = []
#t_cross = []
##data = data000 
#
#for f in range(len(g)):
#    for q in np.arange(1,R,1):
#        for i in range(len(data[4,1][1])):
#            if data[g[f],q][1][i] <= data[g[f],q][1][i-1]:
#                t_cross.append(data[g[f],q][1][i-1])
#                t_c = min(t_cross)
#                #print(np.amin(t_cross))
#        t_cross = []
#        t_c_list.append(t_c)
#        t_c_mean = np.mean(t_c_list)
#        t_c_list = []
#    t_c_av.append(t_c_mean)

""" Cut off Time """

dataTC_Complete = {}
List_Hei = []
List_Num = []
List_S = []

for e in range(len(g)):#7):
    for i in range(len(data[4,1][0])):
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

""" 1. Average Height """

avgH = []
for e in range(len(g)):
    sum_H = sum(dataTC_Complete[g[e],0])
    av_H = sum_H/(I-t_c_av[e][0])
    avgH.append(av_H)

#avg_height = []
#h_list = []
#
#for e in range(len(g)):#7):
#    for v in range(I):#len(data[4,1][0])):
#        if v > t_c_av[e]:
#            h_list.append(data[g[e],1][0][v])
#            sum_height = sum(h_list)
#    h_list = []
#    av_height = sum_height/(I-t_c_av[e])
#    print(t_c_av[e])
#    avg_height.append(av_height)


""" 2. Standard Deviation """

avgHsq = []
for e in range(len(g)):
    sum_Hsq = sum(np.array(dataTC_Complete[g[e],0])**2)
    av_Hsq = sum_Hsq/(I-t_c_av[e][0])
    avgHsq.append(av_Hsq)

#avg_height_sq = []
#h_sq_list = []
#
#for e in range(len(g)):#7):
#    for v in range(I):#len(data[4,1][0])):
#        if v > t_c_av[e]:
#            SQ = np.square(data[g[e],1][0][v])
#            h_sq_list.append(SQ)
#            sum_height = sum(h_sq_list)
#    h_sq_list = []
#    av_height_sq = sum_height/(I-t_c_av[e])
#    print(t_c_av[e])
#    avg_height_sq.append(av_height_sq)

""" 3. Probability P(h;L) """

dataTC = {}
List_Val = []
for e in range(len(g)):#7):
    for i in range(len(data[4,1][0])):
        if i > t_c_av[e][0]:
            List_Val.append(data[g[e],1][0][i])
            dataTC[g[e]] = List_Val
    List_Val = []

Y = np.array([np.size(dataTC[4]), np.size(dataTC[8]), np.size(dataTC[16]), 
              np.size(dataTC[32]), np.size(dataTC[64]), np.size(dataTC[128]),
              np.size(dataTC[256])])

Count = {}

from collections import Counter

for e in range(len(g)):#7):
    Count[g[e],0] = dict(Counter(dataTC[g[e]]))

#Count = Count000
Prb = Count.copy()
for e in range(len(g)):#7):
    for c in sorted(Count[g[e],0].keys()):
        Prb[g[e],0][c] = Count[g[e],0][c]/Y[e]

Prob = [[],[],[],[],[],[],[]]
Val = [[],[],[],[],[],[],[]]

#for e in range(len(g)):#7):
#    for r in range(1, len(Prb[2**(e+2),0]) + 1):
#        Val[e].append(r)
#        Prob[e].append(Prb[g[e],0][r])
for e in range(len(g)):#7):
    for r in sorted(Prb[g[e],0].keys()):
        Val[e].append(r)
        Prob[e].append(Prb[g[e],0][r])

avg_H = np.array(avgH)
avg_H_sq = np.array(avgHsq)
sigma = np.sqrt(avg_H_sq - np.square(avg_H))

scalex = [[],[],[],[],[],[],[],[]]
for z in range(7):
    scalex[z].append((Val[z]-avg_H[z])/sigma[z])

scaley = [[],[],[],[],[],[],[],[]]
for k in range(7):
    scaley[k].append(np.array(Prob[k])*sigma[k])

""" Task 3a """

dataTCP = {}
S_list = []
for e in range(len(g)):#7):
    for i in range(len(data[4,1][0])):
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

Raw_S = [[],[],[],[],[],[],[]]
Height_S = [[],[],[],[],[],[],[]]

for e in range(len(g)):#7):
    for r in sorted(Count_S[g[e]].keys()):
        Height_S[e].append(r)
        Raw_S[e].append(Count_S[g[e]][r])

Prb_S = Count_S.copy()
for e in range(len(g)):#7):
    for c in sorted(Count_S[g[e]].keys()):
        Prb_S[g[e]][c] = Count_S[g[e]][c]/Y_S[e]

Prob_S = [[],[],[],[],[],[],[]]
Val_S = [[],[],[],[],[],[],[]]

for e in range(len(g)):#7):
    for r in sorted(Prb_S[g[e]].keys()):
        Val_S[e].append(r)
        Prob_S[e].append(Prb_S[g[e]][r])

""" Task 3b """
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
#%%
print("Mean number grains in the pile")
print(np.mean(num_grains_pile))
print("Mean height of site 1 = 1")
print(np.mean(height_1[845:]))
##%%
plt.figure()
plt.bar(np.arange(0,len(heights)), heights)
plt.show()
#%% # PLOTTING 
""" TASK 2a: Height vs Time; Uncollapsed """
plt.figure()
#data = data000 

time = np.arange(1, len(data[4,1][0])+1)

plt.plot(time, data[4,1][0], 'x', label = "L = 4")
plt.plot(time, data[8,1][0], 'x', label = "L = 8")
plt.plot(time, data[16,1][0], 'x', label = "L = 16")
plt.plot(time, data[32,1][0], 'x', label = "L = 32")
plt.plot(time, data[64,1][0], 'x', label = "L = 64")
plt.plot(time, data[128,1][0], 'x', label = "L = 128")
plt.plot(time, data[256,1][0], 'x', label = "L = 256")


#plt.xlabel("Time (Number grains added)", size = "13")
plt.xlabel("$t$", size = "14")
plt.ylabel("$h(t; L)$", size = "14")

plt.legend()
plt.grid()
plt.show()
#%% # PLOTTING
""" Task 2b: Average cross-over-time vs System size; Fitted """
plt.figure()

#av_t_L = np.array([208.4,847,1913.2,3476.4,5426.2, 13838, 56209])
#av_t_L = np.array([10.0, 49.0, 211.0, 817.0, 3393.0, 13673.0, 56166.0])
#""" Observed 60000 iterations, 5 realisations, 256 upper size 
#[10.0, 49.0, 211.0, 817.0, 3393.0, 13673.0, 56166.0] """
#L_list = np.array([16, 32, 48, 64, 80, 128, 256])

t_c_AV = [np.mean(np.array(t_c_av[0])), np.mean(np.array(t_c_av[1])), 
          np.mean(np.array(t_c_av[2])), np.mean(np.array(t_c_av[3])),
          np.mean(np.array(t_c_av[4])), np.mean(np.array(t_c_av[5])),
          np.mean(np.array(t_c_av[6]))]


def Square(x, m, c):
    y = m*(x**2) + c 
    return y 

arO = np.arange(4, 256, 0.01)
p0 = np.array([700000, -2.22])
p, cov = opt.curve_fit(Square, g, t_c_AV, p0)
plt.plot(arO, Square(arO, p[0], p[1]), zorder=10,color = 'red')
#pl.errorbar(m_array_mod, N, xerr = err_TO, yerr = err_lnO, color = "royalblue", fmt='o', mew=1, ms=0.2, capsize=6)

#plt.plot(L_list, av_t_L, "x")
plt.plot(g, t_c_AV, "x")

plt.xlabel("$L$", size = "14")
plt.ylabel(r'$\langle t_c \rangle$', size = "14")
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.show()

for c in zip(p, np.sqrt(np.diag(cov))):#zips root of diag of cov matrix with related value in curve fit
    print('%.15f pm %.4g' % (c[0], c[1]))#prints value and uncertainty, f is decimal places and G is sig figs
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

plt.xlabel("$ t $", size = "14")
plt.ylabel(" $ \~h(t; L) $", size = "14")
plt.legend()
plt.grid()
plt.show()
#%% # PLOTTING
""" Task 2d: Processed Height vs Time; Collapsed """
plt.figure()

height_proc_4 = (1/(R-1))*(np.array(dataA[4,1][0]) + np.array(dataA[4,2][0]) + np.array(dataA[4,3][0]) + np.array(dataA[4,4][0]) + np.array(dataA[4,5][0]))
height_proc_8 = (1/(R-1))*(np.array(dataA[8,1][0]) + np.array(dataA[8,2][0]) + np.array(dataA[8,3][0]) + np.array(dataA[8,4][0]) + np.array(dataA[8,5][0]))
height_proc_16 = (1/(R-1))*(np.array(dataA[16,1][0]) + np.array(dataA[16,2][0]) + np.array(dataA[16,3][0]) + np.array(dataA[16,4][0]) + np.array(dataA[16,5][0]))
height_proc_32 = (1/(R-1))*(np.array(dataA[32,1][0]) + np.array(dataA[32,2][0]) + np.array(dataA[32,3][0]) + np.array(dataA[32,4][0]) + np.array(dataA[32,5][0]))
height_proc_64 = (1/(R-1))*(np.array(dataA[64,1][0]) + np.array(dataA[64,2][0]) + np.array(dataA[64,3][0]) + np.array(dataA[64,4][0]) + np.array(dataA[64,5][0]))
height_proc_128 = (1/(R-1))*(np.array(dataA[128,1][0]) + np.array(dataA[128,2][0]) + np.array(dataA[128,3][0]) + np.array(dataA[128,4][0]) + np.array(dataA[128,5][0]))
height_proc_256 = (1/(R-1))*(np.array(dataA[256,1][0]) + np.array(dataA[256,2][0]) + np.array(dataA[256,3][0]) + np.array(dataA[256,4][0]) + np.array(dataA[256,5][0]))

time = np.arange(1, len(height_proc_4)+1)

#plt.plot(time/(4**2), height_proc_4/4, 'x', label = "L = 4")
#plt.plot(time/(8**2), height_proc_8/8, 'x', label = "L = 8")
#plt.plot(time/(16**2), height_proc_16/16, 'x', label = "L = 16")
#plt.plot(time/(32**2), height_proc_32/32, 'x', label = "L = 32")
#plt.plot(time/(64**2), height_proc_64/64, 'x', label = "L = 64")
#plt.plot(time/(128**2), height_proc_128/128, 'x', label = "L = 128")
#plt.plot(time/(256**2), height_proc_256/256, 'x', label = "L = 256")

plt.plot(time/(4**2), height_proc_4/4, label = "L = 4")
plt.plot(time/(8**2), height_proc_8/8, label = "L = 8")
plt.plot(time/(16**2), height_proc_16/16, label = "L = 16")
plt.plot(time/(32**2), height_proc_32/32, label = "L = 32")
plt.plot(time/(64**2), height_proc_64/64, label = "L = 64")
plt.plot(time/(128**2), height_proc_128/128, label = "L = 128")
plt.plot(time/(256**2), height_proc_256/256, label = "L = 256")

plt.xscale("log")
plt.yscale("log")

plt.xlabel("$ t / L^2 $", size = "14")
plt.ylabel(" $ \~h(t; L) / L $", size = "14")
plt.legend()
plt.grid()
plt.show()
#%% # PLOTTING
""" Task 2e: Time averaged height vs System size """
## 15000 iterations , 64
# avg_height = [6.3244162775183455, 12.965487258377365, 26.498140509838393, 
# 53.90002115208348, 108.92943913155854]
## 60000 iterations , 256
#avg_height = [6.308634772462077, 12.968857900618838, 26.51313786816973, 
# 53.890627376104625, 108.93797586870882, 219.25980529712695, 439.1225873761085]
## 150000 iterations, 256 I = 60000
#avg_height = [6.308634772462077, 12.968857900618838, 26.51313786816973, 
# 53.890627376104625, 108.93797586870882, 219.25980529712695, 439.1225873761085]

#avg_H = np.array(avg_height)
avg_H = np.array(avgH)

plt.figure()

def Corr(x, a0, a1, w1):
    y = a0*x*(1-a1*(x**-w1))
    return y 

arO = np.arange(0, 256, 0.0001)
p0 = np.array([1.7, 1, 0.64])
p, cov = opt.curve_fit(Corr, g, avg_H, p0)
plt.plot(arO, Corr(arO, p[0], p[1], p[2]), zorder=10,color = 'red')
#pl.errorbar(m_array_mod, N, xerr = err_TO, yerr = err_lnO, color = "royalblue", fmt='o', mew=1, ms=0.2, capsize=6)

#plt.plot(g, Linear(g,1.7,0.0001), "red")
plt.plot(g, avg_H, 'x')

plt.xlabel("$L$", size = "14")
plt.ylabel(r'$\langle h(t; L) \rangle _t $', size = "14")
plt.grid()
plt.show()

for c in zip(p, np.sqrt(np.diag(cov))):#zips root of diag of cov matrix with related value in curve fit
    print('%.15f pm %.4g' % (c[0], c[1]))#prints value and uncertainty, f is decimal places and G is sig figs
#%% # PLOTTING
""" Task 2f: Standard deviation of the Height vs System Size """
plt.figure()

#avg_height_sq 60000 , 256 
#avg_height_sq = [40.44915819303217, 169.10940601491217, 704.2378363913094, 
# 2906.0064038659752, 11870.27077923225, 48079.52112159216, 192882.36932707354]
#
#avg_height = [6.308634772462077, 12.968857900618838, 26.51313786816973, 
# 53.890627376104625, 108.93797586870882, 219.25980529712695, 439.1225873761085]

#avg_H = np.array(avg_height)
#avg_H_sq = np.array(avg_height_sq)

avg_H = np.array(avgH)
avg_H_sq = np.array(avgHsq)
sigma = np.sqrt(avg_H_sq - np.square(avg_H))

def PLaw(x, m, k):
    y = m*(x**-k)
    return y 

arO = np.arange(0, 256, 0.0001)
p0 = np.array([0.2, 0.58])
p, cov = opt.curve_fit(PLaw, g, sigma, p0)
plt.plot(arO, PLaw(arO, p[0], p[1]), zorder=10, color = 'red')
#pl.errorbar(m_array_mod, N, xerr = err_TO, yerr = err_lnO, color = "royalblue", fmt='o', mew=1, ms=0.2, capsize=6)


plt.plot(g, sigma, 'x')

plt.xlabel("$L$", size = "14")
plt.ylabel(r'$\sigma_h(L) $', size = "14")
plt.xscale("log")
plt.yscale("log")
#plt.grid()
plt.show()

for c in zip(p, np.sqrt(np.diag(cov))):#zips root of diag of cov matrix with related value in curve fit
    print('%.15f pm %.4g' % (c[0], c[1]))#prints value and uncertainty, f is decimal places and G is sig figs
#%% # PLOTTING
print(sum(Prb[4,0].values()))
print(sum(Prb[8,0].values()))
print(sum(Prb[16,0].values()))
print(sum(Prb[32,0].values()))
print(sum(Prb[64,0].values()))
print(sum(Prb[128,0].values()))
print(sum(Prb[256,0].values()))         
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

plt.xlabel("$h$", size = "14")
plt.ylabel("$P(h; L)$", size = "14")
plt.legend()
plt.grid()
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

plt.xlabel(r'$ (h - \langle h(t; L) \rangle _t )/\sigma_h(L) $', size = "14")
plt.ylabel("$ P(h; L)\sigma_h(L) $", size = "14")
plt.legend()
plt.grid()
plt.show()
#%%
""" Task 3a)a): Avalanche Size Probability vs Avalanche size s  - Unbinned; 
Done from your code not external binning code: do not use in end report """
plt.figure()

plt.plot(Val_S[0], Prob_S[0], label = "L = 4")
plt.plot(Val_S[1], Prob_S[1], label = "L = 8")
plt.plot(Val_S[2], Prob_S[2], label = "L = 16")
plt.plot(Val_S[3], Prob_S[3], label = "L = 32")
plt.plot(Val_S[4], Prob_S[4], label = "L = 64")
plt.plot(Val_S[5], Prob_S[5], label = "L = 128")
plt.plot(Val_S[6], Prob_S[6], label = "L = 256")

plt.xlabel("Height of System", size = "13")
plt.ylabel("Probability", size = "13")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()
plt.show()
#%% # PLOTTING
""" Task 3a)a): Avalanche Size Probability vs Avalanche size s - Binned using external code """

scale = 1.25
s0 = False # whether or not to include s = 0 avalanches 

bin_4 = logbin(data[4,1][2],scale, s0)
bin_8 = logbin(data[8,1][2],scale, s0)
bin_16 = logbin(data[16,1][2],scale, s0)
bin_32 = logbin(data[32,1][2],scale, s0)
bin_64 = logbin(data[64,1][2],scale, s0)
bin_128 = logbin(data[128,1][2],scale, s0)
bin_256 = logbin(data[256,1][2],scale, s0)

plt.figure()
#plt.plot(bin_4[0], bin_4[1], 'x', label = "L = 4")
#plt.plot(bin_8[0], bin_8[1], 'x', label = "L = 8")
#plt.plot(bin_16[0], bin_16[1], 'x', label = "L = 16")
#plt.plot(bin_32[0], bin_32[1], 'x', label = "L = 32")
#plt.plot(bin_64[0], bin_64[1], 'x', label = "L = 64")
#plt.plot(bin_128[0], bin_128[1], 'x', label = "L = 128")
#plt.plot(bin_256[0], bin_256[1], 'x', label = "L = 256")

plt.plot(bin_4[0], bin_4[1], label = "L = 4")
plt.plot(bin_8[0], bin_8[1], label = "L = 8")
plt.plot(bin_16[0], bin_16[1], label = "L = 16")
plt.plot(bin_32[0], bin_32[1], label = "L = 32")
plt.plot(bin_64[0], bin_64[1], label = "L = 64")
plt.plot(bin_128[0], bin_128[1], label = "L = 128")
plt.plot(bin_256[0], bin_256[1], label = "L = 256")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("$s$", size = "14")
plt.ylabel("$\~P(s; L)$", size = "14")
plt.legend()
plt.grid()
plt.show()
#%%
""" Task 3a)b): Cutoff avalanche size vs System size; Parameter D """
plt.figure()

s_c = [np.amax(bin_4[0]), np.amax(bin_8[0]), np.amax(bin_16[0]), np.amax(bin_32[0]), 
       np.amax(bin_64[0]), np.amax(bin_128[0]), np.amax(bin_256[0])]

def PLaw(x, m, k):
    y = m*(x**-k)
    return y 

def Corr(x, b0, D, b1, w1):
    y = b0*(x**D)*(1 + b1*(x**-w1))
    return y 

arO = np.arange(4, 256, 0.0001)
p0 = np.array([1, 2.2])
p, cov = opt.curve_fit(PLaw, g[3:], s_c[3:], p0)
plt.plot(arO, PLaw(arO, p[0], p[1]), zorder=10, color = 'red')

#p0 = np.array([1, 2.2, 0.5, 0.5])
#p, cov = opt.curve_fit(Corr, g, s_c, p0)
#plt.plot(arO, Corr(arO, p[0], p[1], p[2], p[3]), zorder=10, color = 'red')

#pl.errorbar(m_array_mod, N, xerr = err_TO, yerr = err_lnO, color = "royalblue", fmt='o', mew=1, ms=0.2, capsize=6)

plt.plot(g, s_c, 'x')

plt.xscale("log")
plt.yscale("log")
plt.xlabel("$L$", size = "14")
plt.ylabel("$ s_c(L) $", size = "14")
plt.grid()
plt.show()

for c in zip(p, np.sqrt(np.diag(cov))):#zips root of diag of cov matrix with related value in curve fit
    print('%.15f pm %.4g' % (c[0], c[1]))#prints value and uncertainty, f is decimal places and G is sig figs
#%%
""" Task 3a)b): Probability of avalanche size vs Avalanche size; Parameter tau """
#scale = 1.25
#s0 = False # whether or not to include s = 0 avalanches 

#bin_256 = logbin(data[256,1][2],scale, s0)

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
#pl.errorbar(m_array_mod, N, xerr = err_TO, yerr = err_lnO, color = "royalblue", fmt='o', mew=1, ms=0.2, capsize=6)

plt.plot(bin_256[0], bin_256[1], 'x', label = "L = 256")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("$s$", size = "14")
plt.ylabel("$\~P(s; L)$", size = "14")
plt.legend()
plt.grid()
plt.show()

for c in zip(p, np.sqrt(np.diag(cov))):#zips root of diag of cov matrix with related value in curve fit
    print('%.15f pm %.4g' % (c[0], c[1]))#prints value and uncertainty, f is decimal places and G is sig figs
#%%
""" Task 3a)a): Avalanche Size Probability vs Avalanche size s - Binned; Data Collapse """
plt.figure()

# 500k; 2
#tau = 1.479
#D = 1.968

# 1.5M; 2
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
plt.xlabel("$s/L^D$", size = "14")
plt.ylabel(r'$ \~P(s; L) s^{\tau_s} $', size = "14")
plt.legend()
plt.grid()
plt.show()
#%% # PLOTTING
""" Task 3b: Measuring the kth moment in the steady state vs System size L """
plt.figure()

plt.plot(g, kth_moment[0], label = "k = 1")
plt.plot(g, kth_moment[1], label = "k = 2")
plt.plot(g, kth_moment[2], label = "k = 3")
plt.plot(g, kth_moment[3], label = "k = 4")
plt.plot(g, kth_moment[4], label = "k = 5")
plt.plot(g, kth_moment[5], label = "k = 6")
plt.plot(g, kth_moment[6], label = "k = 7")
plt.plot(g, kth_moment[7], label = "k = 8")
plt.plot(g, kth_moment[8], label = "k = 9")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("$L$", size = "14")
plt.ylabel(r'$\langle s^k \rangle $', size = "14")
plt.grid()
plt.legend()
plt.show()
#%% # PLOTTING
""" Task 3b: Measuring the kth moment in the steady state vs System size L: Fitted Functions """
plt.figure()

plt.plot(g, kth_moment[0], 'x', label = "k = 1")
plt.plot(g, kth_moment[1], 'x', label = "k = 2")
plt.plot(g, kth_moment[2], 'x', label = "k = 3")
plt.plot(g, kth_moment[3], 'x', label = "k = 4")
plt.plot(g, kth_moment[4], 'x', label = "k = 5")
plt.plot(g, kth_moment[5], 'x', label = "k = 6")
#plt.plot(g, kth_moment[6], 'x', label = "k = 7")
#plt.plot(g, kth_moment[7], 'x', label = "k = 8")
#plt.plot(g, kth_moment[8], 'x', label = "k = 9")

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
po4, cov4 = opt.curve_fit(PLaw, g, kth_moment[4], p4)
po5, cov5 = opt.curve_fit(PLaw, g, kth_moment[5], p5)

plt.plot(arO, PLaw(arO, po0[0], po0[1]), zorder=10, color = 'royalblue', label = "Fitted Function")
plt.plot(arO, PLaw(arO, po1[0], po1[1]), zorder=10, color = 'royalblue')
plt.plot(arO, PLaw(arO, po2[0], po2[1]), zorder=10, color = 'royalblue')
plt.plot(arO, PLaw(arO, po3[0], po3[1]), zorder=10, color = 'royalblue')
plt.plot(arO, PLaw(arO, po4[0], po4[1]), zorder=10, color = 'royalblue')
plt.plot(arO, PLaw(arO, po5[0], po5[1]), zorder=10, color = 'royalblue')

plt.xscale("log")
plt.yscale("log")
plt.xlabel("$L$", size = "14")
plt.ylabel(r'$\langle s^k \rangle $', size = "14")
plt.grid()
plt.legend()
plt.show()

#for c in zip(po0, np.sqrt(np.diag(cov0))):#zips root of diag of cov matrix with related value in curve fit
#    print('%.15f pm %.4g' % (c[0], c[1]))#prints value and uncertainty, f is decimal places and G is sig figs
#
#for c in zip(po1, np.sqrt(np.diag(cov1))):#zips root of diag of cov matrix with related value in curve fit
#    print('%.15f pm %.4g' % (c[0], c[1]))#prints value and uncertainty, f is decimal places and G is sig figs
#
#for c in zip(po2, np.sqrt(np.diag(cov2))):#zips root of diag of cov matrix with related value in curve fit
#    print('%.15f pm %.4g' % (c[0], c[1]))#prints value and uncertainty, f is decimal places and G is sig figs
#
#for c in zip(po3, np.sqrt(np.diag(cov3))):#zips root of diag of cov matrix with related value in curve fit
#    print('%.15f pm %.4g' % (c[0], c[1]))#prints value and uncertainty, f is decimal places and G is sig figs
#
#for c in zip(po4, np.sqrt(np.diag(cov4))):#zips root of diag of cov matrix with related value in curve fit
#    print('%.15f pm %.4g' % (c[0], c[1]))#prints value and uncertainty, f is decimal places and G is sig figs
#
#for c in zip(po5, np.sqrt(np.diag(cov5))):#zips root of diag of cov matrix with related value in curve fit
#    print('%.15f pm %.4g' % (c[0], c[1]))#prints value and uncertainty, f is decimal places and G is sig figs
#%%
""" Task 3b: Moment Analysis to determine D and tau from kth moment """
plt.figure()

alph = np.array([-po0[1], -po1[1], -po2[1], -po3[1], -po4[1], -po5[1]])
k_i = np.array([1,2,3,4,5,6])

def Linear(x, m, c):
    y = m*x + c
    return y 

arO = np.arange(1, 6, 0.0001)
p0 = np.array([2.25, -1.3])
p, cov = opt.curve_fit(Linear, k_i, alph, p0)
plt.plot(arO, Linear(arO, p[0], p[1]), zorder=10, color = 'red')
#pl.errorbar(m_array_mod, N, xerr = err_TO, yerr = err_lnO, color = "royalblue", fmt='o', mew=1, ms=0.2, capsize=6)

plt.plot(k_i, alph, 'x')

plt.xlabel("$k$", size = "14")
plt.ylabel(r'$ D( 1 + k - \tau_s)$', size = "14")
plt.grid()
#plt.legend()
plt.show()

#for c in zip(p, np.sqrt(np.diag(cov))):#zips root of diag of cov matrix with related value in curve fit
#    print('%.15f pm %.4g' % (c[0], c[1]))#prints value and uncertainty, f is decimal places and G is sig figs

print("D estimate:")
print(p[0])
D_est = p[0]

print("tau estimate:")
print(-p[1]/p[0] + 1)
tau_est = -p[1]/p[0] + 1

print("x-intercept:")
print(-p[1]/p[0])