# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 10:45:50 2021

@author: 44743
"""
"""
Complexity and Networks Lab """
""" TASKS
1. Compute avalanche size s 
2. Create seperate test scripts 
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pickle
#%% WRITE
import pickle

with open(r'C:\Users\44743\Documents\Imperial Year 3\Complexity & Networks\AAAAAAAA', 'wb') as dummy:
    pickle.dump(data, dummy, protocol=pickle.HIGHEST_PROTOCOL)
    
#%% READ
import pickle
# Data100k;6
# Data100k;10
# Data1.5M;2R
with open(r'C:\Users\44743\Documents\Imperial Year 3\Complexity & Networks\Data100k;10', 'rb') as dummy:
    dataA = pickle.load(dummy)




#%% READ TESTING
with open(r'C:\Users\44743\Documents\Imperial Year 3\Complexity & Networks\Testingp0;128', 'rb') as dummy:
    dataA = pickle.load(dummy)

with open(r'C:\Users\44743\Documents\Imperial Year 3\Complexity & Networks\Testingp1;128', 'rb') as dummy:
    dataB = pickle.load(dummy)

with open(r'C:\Users\44743\Documents\Imperial Year 3\Complexity & Networks\Testingp01632', 'rb') as dummy:
    dataC = pickle.load(dummy)

with open(r'C:\Users\44743\Documents\Imperial Year 3\Complexity & Networks\Testingp1;1632', 'rb') as dummy:
    dataD = pickle.load(dummy)

with open(r'C:\Users\44743\Documents\Imperial Year 3\Complexity & Networks\Testingp005;', 'rb') as dummy:
    dataE = pickle.load(dummy)

with open(r'C:\Users\44743\Documents\Imperial Year 3\Complexity & Networks\Testingp095;', 'rb') as dummy:
    dataF = pickle.load(dummy)














#%%
#""" Initialisation """
#
## System size L
#L = 4
# s =0 
###%%
## Probability of z threshold z_th = 1
#prob = 0.5
#
#z_thres = np.zeros(L)
#heights = np.zeros(L)
#z = np.zeros(L)
#x_pos = np.arange(1,L+1)
#
#num_grains_added = np.array([0])
#height_1 = []
#num_grains_pile = []
#
## Initialisation
#for x in range(0,L):
#    #z_thres[x] = np.random.randint(1,3)
#    z_thres[x] = np.random.choice([1,2], p = [prob,1-prob])
#%% 
""" TASK 1 """

def Drive():
    num_grains_added[0] = num_grains_added[0] + 1
    heights[0] = heights[0] + 1
    z[0] = z[0] + 1
    s[0] = 0
    s[1] = 0
    s[2] = 0

def Relaxation():
    # Of no consequence - i has no meaning in the rest of the for loop
#    for i in range(0,L):
#        while z[i] > z_thres[i]:
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
            s[2] += 1
            z[-1] -= 1
            z[-2] += 1
            heights[-1] -= 1
            z_thres[-1] = np.random.choice([1,2], p = [prob,1-prob])
    
    s_list.append(sum(s))
    height_1.append(heights[0])
    num_grains_pile.append(sum(heights))
#    else:
#        print("end")

def Algrithm(Iterations):
    for b in range(Iterations):
        Drive()
        Relaxation()
##%% 
#for l in range(0,10):
#
#def Realisations(Itr, Number):
#    for j in range(Number):
#        Algrithm(Itr)
#        

#Realisations(100, 1)
#%%
""" Initialisation """

# System size L
L = 4
prob = 0.5

s = np.array([0,0,0])
z_thres = np.zeros(L)
heights = np.zeros(L)
z = np.zeros(L)
x_pos = np.arange(1,L+1)

num_grains_added = np.array([0])
height_1 = []
num_grains_pile = []
s_list = []

for x in range(0,L):
    z_thres[x] = np.random.choice([1,2], p = [prob,1-prob])

dataB = {}
R = 2 # Number of Realisations
I = 60000
g = np.array([4,8,16,32,64,128,256])
for j in range(len(g)):
    for h in range(1,R):
        Algrithm(I)
        dataB[g[j],h] = [height_1, num_grains_pile, s_list]
        
        if h < R-1:
            L = g[j]
        if h == R-1:
            if j >= len(g)-1:
                L = g[j]
            else:
                L = g[j+1]
        s = np.array([0,0,0])
        z_thres = np.zeros(L)
        heights = np.zeros(L)
        z = np.zeros(L)
        x_pos = np.arange(1,L+1)
        
        num_grains_added = np.array([0])
        height_1 = []
        num_grains_pile = []
        s_list = []
        
        # Initialisation
        for x in range(0,L):
            #z_thres[x] = np.random.randint(1,3)
            z_thres[x] = np.random.choice([1,2], p = [prob,1-prob])
#%%
#L = []
#for i in range(2,9):
##    L.append(2**i)
##%%
#data = {}
#
#data['a'] = height_1
#data['a'] = [height_1,num_grains_pile]
#

##%%
print("Mean number grains in the pile")
print(np.mean(num_grains_pile))
print("Mean height of site 1 = 1")
print(np.mean(height_1[845:]))
##%%
plt.figure()
plt.bar(np.arange(0,len(heights)), heights)
plt.show()
#%%
""" TASK 2a """

#data = data000 

time = np.arange(1, len(data[4,1][0])+1)

plt.plot(time, data[4,1][0], 'x', label = "L = 4")
plt.plot(time, data[8,1][0], 'x', label = "L = 8")
plt.plot(time, data[16,1][0], 'x', label = "L = 16")
plt.plot(time, data[32,1][0], 'x', label = "L = 32")
plt.plot(time, data[64,1][0], 'x', label = "L = 64")
plt.plot(time, data[128,1][0], 'x', label = "L = 128")
plt.plot(time, data[256,1][0], 'x', label = "L = 256")


plt.xlabel("Time (Number grains added)", size = "13")
plt.ylabel("Height of the pile", size = "13")
plt.legend()
plt.grid()
plt.show()
#%% 
""" Task 2b """

#DO NOT RUN AGAIN
t_c_av = []
t_c_list = []

##%% 

t_cross = []
#data = data000 

#Temp = {}

#for f in range(len(s)):
#    for q in range(1,R):
#        Temp[s[f],q] = [height_1, num_grains_pile]

for f in range(len(g)):
    for q in np.arange(1,R,1):
        for i in range(len(data[4,1][1])):
            if data[g[f],q][1][i] <= data[g[f],q][1][i-1]:
                t_cross.append(data[g[f],q][1][i-1])
                t_c = min(t_cross)
                #print(np.amin(t_cross))
        t_cross = []
        t_c_list.append(t_c)
        t_c_mean = np.mean(t_c_list)
        t_c_list = []
    t_c_av.append(t_c_mean)

##%%
#t_c_av.append(t_c)
#%%
# Fitting Square Fit onto t_c_av
plt.figure()

#av_t_L = np.array([208.4,847,1913.2,3476.4,5426.2, 13838, 56209])
av_t_L = np.array([10.0, 49.0, 211.0, 817.0, 3393.0, 13673.0, 56166.0])

""" Observed 60000 iterations, 5 realisations, 256 upper size 
[10.0, 49.0, 211.0, 817.0, 3393.0, 13673.0, 56166.0] """
L_list = np.array([16, 32, 48, 64, 80, 128, 256])

def Square(x, m, c):
    y = m*(x**2) + c 
    return y 

arO = np.arange(0, 256, 0.01)
p0 = np.array([700000, -2.22])
p, cov = opt.curve_fit(Square, g, t_c_av, p0)
plt.plot(arO, Square(arO, p[0], p[1]), zorder=10,color = 'red')
#pl.errorbar(m_array_mod, N, xerr = err_TO, yerr = err_lnO, color = "royalblue", fmt='o', mew=1, ms=0.2, capsize=6)

#plt.plot(L_list, av_t_L, "x")
plt.plot(g, t_c_av, "x")

plt.xlabel("System Size [$L$]", size = "13")
plt.ylabel("Average Cross-over time", size = "13")
plt.grid()
plt.show()

for c in zip(p, np.sqrt(np.diag(cov))):#zips root of diag of cov matrix with related value in curve fit
    print('%.15f pm %.4g' % (c[0], c[1]))#prints value and uncertainty, f is decimal places and G is sig figs

#%%
""" Task 2d """
# Processed Height
plt.figure()

#data = data000
height_proc_4 = (1/(R-1))*(np.array(data[4,1][0]) + np.array(data[4,2][0]) + np.array(data[4,3][0]) + np.array(data[4,4][0]) + np.array(data[4,5][0]))
height_proc_8 = (1/(R-1))*(np.array(data[8,1][0]) + np.array(data[8,2][0]) + np.array(data[8,3][0]) + np.array(data[8,4][0]) + np.array(data[8,5][0]))
height_proc_16 = (1/(R-1))*(np.array(data[16,1][0]) + np.array(data[16,2][0]) + np.array(data[16,3][0]) + np.array(data[16,4][0]) + np.array(data[16,5][0]))
height_proc_32 = (1/(R-1))*(np.array(data[32,1][0]) + np.array(data[32,2][0]) + np.array(data[32,3][0]) + np.array(data[32,4][0]) + np.array(data[32,5][0]))
height_proc_64 = (1/(R-1))*(np.array(data[64,1][0]) + np.array(data[64,2][0]) + np.array(data[64,3][0]) + np.array(data[64,4][0]) + np.array(data[64,5][0]))
height_proc_128 = (1/(R-1))*(np.array(data[128,1][0]) + np.array(data[128,2][0]) + np.array(data[128,3][0]) + np.array(data[128,4][0]) + np.array(data[128,5][0]))
height_proc_256 = (1/(R-1))*(np.array(data[256,1][0]) + np.array(data[256,2][0]) + np.array(data[256,3][0]) + np.array(data[256,4][0]) + np.array(data[256,5][0]))

time = np.arange(1, len(height_proc_4)+1)

#plt.plot(time, height_proc_4, 'x', label = "L = 4")
#plt.plot(time, height_proc_8, 'x', label = "L = 8")
#plt.plot(time, height_proc_16, 'x', label = "L = 16")
#plt.plot(time, height_proc_32, 'x', label = "L = 32")
#plt.plot(time, height_proc_64, 'x', label = "L = 64")
#plt.plot(time, height_proc_128, 'x', label = "L = 128")
#plt.plot(time, height_proc_256, 'x', label = "L = 256")

plt.plot(time/(4**2), height_proc_4/4, 'x', label = "L = 4")
plt.plot(time/(8**2), height_proc_8/8, 'x', label = "L = 8")
plt.plot(time/(16**2), height_proc_16/16, 'x', label = "L = 16")
plt.plot(time/(32**2), height_proc_32/32, 'x', label = "L = 32")
plt.plot(time/(64**2), height_proc_64/64, 'x', label = "L = 64")
plt.plot(time/(128**2), height_proc_128/128, 'x', label = "L = 128")
plt.plot(time/(256**2), height_proc_256/256, 'x', label = "L = 256")

#plt.xscale("log")
#plt.yscale("log")

plt.xlabel("Time / L^2", size = "13")
plt.ylabel("Processed Height / L", size = "13")
plt.legend()
plt.grid()
plt.show()

#%%
""" Task 2e """

""" 1. Average Height """
#data = data000

#t_c_av = [10.0, 49.0, 211.0, 817.0, 3393.0, 13673.0, 56166.0]

I = 150000
avg_height = []
h_list = []

for e in range(len(g)):#7):
    for v in range(I):#len(data[4,1][0])):
        if v > t_c_av[e]:
            h_list.append(data[g[e],1][0][v])
            sum_height = sum(h_list)
    h_list = []
    av_height = sum_height/(I-t_c_av[e])
    print(t_c_av[e])
    avg_height.append(av_height)
        #h_list = []
#%%
avgH = []
for e in range(len(g)):
    sum_H = sum(np.array(dataTC_Complete[g[e],0])**2)
    av_H = sum_H/(I-t_c_av[e])
    avgH.append(av_H)
    
#%% FAST VERSION
#data = datat000
av_height_4 = (sum(np.array(data[4,1][0]) > t_c_av[0]))/(I-t_c_av[0])
av_height_8 = (sum(np.array(data[8,1][0])))/(I-t_c_av[1])
av_height_16 = (sum(np.array(data[16,1][0])))/(I-t_c_av[2])
av_height_32 = (sum(np.array(data[32,1][0])))/(I-t_c_av[3])
av_height_64 = (sum(np.array(data[64,1][0])))/(I-t_c_av[4])
av_height_128 = (sum(np.array(data[128,1][0])))/(I-t_c_av[5])
av_height_256 = (sum(np.array(data[256,1][0])))/(I-t_c_av[6])

av_height_L = np.array([av_height_4, av_height_8, av_height_16, 
                        av_height_32, av_height_64, av_height_128, av_height_256])

#%%
# 15000 iterations , 64
# avg_height = [6.3244162775183455, 12.965487258377365, 26.498140509838393, 
# 53.90002115208348, 108.92943913155854]

# 60000 iterations , 256
avg_height = [6.308634772462077, 12.968857900618838, 26.51313786816973, 
 53.890627376104625, 108.93797586870882, 219.25980529712695, 439.1225873761085]

# 150000 iterations, 256 I = 150000
#avg_height = [6.308509290681316, 12.978473685614446, 26.52951818891908, 
# 53.87442253407803, 108.93310310938823, 219.35373059736594, 440.4234454114347]

avg_H = np.array(avg_height)

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

plt.xlabel("System size L", size = "13")
plt.ylabel("Average height", size = "13")
plt.grid()
plt.show()

for c in zip(p, np.sqrt(np.diag(cov))):#zips root of diag of cov matrix with related value in curve fit
    print('%.15f pm %.4g' % (c[0], c[1]))#prints value and uncertainty, f is decimal places and G is sig figs
#%%
""" 2. Standard Deviation """

#data = data000

I = 150000
avg_height_sq = []
h_sq_list = []

for e in range(len(g)):#7):
    for v in range(I):#len(data[4,1][0])):
        if v > t_c_av[e]:
            SQ = np.square(data[g[e],1][0][v])
            h_sq_list.append(SQ)
            sum_height = sum(h_sq_list)
    h_sq_list = []
    av_height_sq = sum_height/(I-t_c_av[e])
    print(t_c_av[e])
    avg_height_sq.append(av_height_sq)

#%%

#avg_height_sq 60000 , 256 
#avg_height_sq = [40.44915819303217, 169.10940601491217, 704.2378363913094, 
# 2906.0064038659752, 11870.27077923225, 48079.52112159216, 192882.36932707354]
#
## avg_height_sq 150000 256
##[40.444519264746084, 169.35126303715757, 705.1064764435306, 2904.276077321765,
## 11868.954737790396, 48119.93691204139, 193979.7144524016]
#
#avg_height = [6.308634772462077, 12.968857900618838, 26.51313786816973, 
# 53.890627376104625, 108.93797586870882, 219.25980529712695, 439.1225873761085]

avg_H = np.array(avg_height)
avg_H_sq = np.array(avg_height_sq)

sigma = np.sqrt(avg_H_sq - np.square(avg_H))

plt.figure()

plt.plot(g, sigma, 'x')

plt.xlabel("System size L", size = "13")
plt.ylabel("Standard Deviation of system height", size = "13")
plt.grid()
plt.show()
#%%
""" 3. Probability P(h;L) """

#data = data000
dataTC = {}
List_Val = []
for e in range(len(g)):#7):
    for i in range(len(data[4,1][0])):
        if i > t_c_av[e]:
            List_Val.append(data[g[e],1][0][i])
            dataTC[g[e]] = List_Val
    List_Val = []

##%%
Y = np.array([np.size(dataTC[4]), np.size(dataTC[8]), np.size(dataTC[16]), 
              np.size(dataTC[32]), np.size(dataTC[64]), np.size(dataTC[128]),
              np.size(dataTC[256])])
#%%
""" Cut off Time Dictionary """
dataTC_Complete = {}
List_Hei = []
List_Num = []
List_S = []

for e in range(len(g)):#7):
    for i in range(len(dataB[4,1][0])):
        if i > t_c_av[e]:
            List_Hei.append(dataB[g[e],1][0][i])
            List_Num.append(dataB[g[e],1][1][i])
            List_S.append(dataB[g[e],1][2][i])
            dataTC_Complete[g[e],0] = List_Hei
            dataTC_Complete[g[e],1] = List_Num
            dataTC_Complete[g[e],2] = List_S
    List_Hei = []
    List_Num = []
    List_S = []
#%%
Count = {}

from collections import Counter

for e in range(len(g)):#7):
    Count[g[e],0] = dict(Counter(dataTC[g[e]]))
#%%
#Count = Count000
Prb = Count.copy()
for e in range(len(g)):#7):
    for c in sorted(Count[g[e],0].keys()):
        Prb[g[e],0][c] = Count[g[e],0][c]/Y[e]
        #Prb[g[e],0] = Prob 
#Count = {}
#for e in range(len(g)):#7):
#    Count[g[e],0] = {c:dataTC[g[e]].count(c) for c in dataTC[g[e]]}
#%%
print(sum(Prb[4,0].values()))
print(sum(Prb[8,0].values()))
print(sum(Prb[16,0].values()))
print(sum(Prb[32,0].values()))
print(sum(Prb[64,0].values()))
print(sum(Prb[128,0].values()))
print(sum(Prb[256,0].values()))         
#%%
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
#%%
plt.figure()

plt.plot(Val[0], Prob[0], label = "4")
plt.plot(Val[1], Prob[1], label = "8")
plt.plot(Val[2], Prob[2], label = "16")
plt.plot(Val[3], Prob[3], label = "32")
plt.plot(Val[4], Prob[4], label = "64")
plt.plot(Val[5], Prob[5], label = "128")
plt.plot(Val[6], Prob[6], label = "256")

plt.xlabel("Height of System", size = "13")
plt.ylabel("Probability", size = "13")
plt.legend()
plt.grid()
plt.show()
#%% Data Collapse

avg_H = np.array(avgH)
avg_H_sq = np.array(avgHsq)
sigma = np.sqrt(avg_H_sq - np.square(avg_H))

scalex = [[],[],[],[],[],[],[],[]]
for z in range(7):
    scalex[z].append((Val[z]-avg_H[z])/sigma[z])

scaley = [[],[],[],[],[],[],[],[]]
for k in range(7):
    scaley[k].append(np.array(Prob[k])*sigma[k])
#%%
plt.figure()
plt.plot(scalex[0][0], scaley[0][0], 'x', label = "4")
plt.plot(scalex[1][0], scaley[1][0], 'x',  label = "8")
plt.plot(scalex[2][0], scaley[2][0], 'x',  label = "16")
plt.plot(scalex[3][0], scaley[3][0], 'x',  label = "32")
plt.plot(scalex[4][0], scaley[4][0], 'x',  label = "64")
plt.plot(scalex[5][0], scaley[5][0], 'x',  label = "128")
plt.plot(scalex[6][0], scaley[6][0], 'x',  label = "256")

plt.xlabel("(H-av_H)/sigma", size = "13")
plt.ylabel("P*sigma", size = "13")
plt.legend()
plt.grid()
plt.show()
#%% 
""" Task 3a """

dataTCP = {}
S_list = []
for e in range(len(g)):#7):
    for i in range(len(data[4,1][0])):
        if i > t_c_av[e]:
            S_list.append(data[g[e],1][2][i])
            dataTCP[g[e]] = S_list
    S_list = []
#%%
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
#%%
Prb_S = Count_S.copy()
for e in range(len(g)):#7):
    for c in sorted(Count_S[g[e]].keys()):
        Prb_S[g[e]][c] = Count_S[g[e]][c]/Y_S[e]
        #%%
print(sum(Prb_S[4].values()))
print(sum(Prb_S[8].values()))
print(sum(Prb_S[16].values()))
print(sum(Prb_S[32].values()))
print(sum(Prb_S[64].values()))
print(sum(Prb_S[128].values()))
print(sum(Prb_S[256].values()))
#%%
Prob_S = [[],[],[],[],[],[],[]]
Val_S = [[],[],[],[],[],[],[]]

#for e in range(len(g)):#7):
#    for r in range(1, len(Prb[2**(e+2),0]) + 1):
#        Val[e].append(r)
#        Prob[e].append(Prb[g[e],0][r])
for e in range(len(g)):#7):
    for r in sorted(Prb_S[g[e]].keys()):
        Val_S[e].append(r)
        Prob_S[e].append(Prb_S[g[e]][r])
#%%
# Val_S is the avala
plt.figure()

plt.plot(Val_S[0], Prob_S[0], label = "4")
plt.plot(Val_S[1], Prob_S[1], label = "8")
plt.plot(Val_S[2], Prob_S[2], label = "16")
plt.plot(Val_S[3], Prob_S[3], label = "32")
plt.plot(Val_S[4], Prob_S[4], label = "64")
plt.plot(Val_S[5], Prob_S[5], label = "128")
plt.plot(Val_S[6], Prob_S[6], label = "256")

plt.xlabel("Height of System", size = "13")
plt.ylabel("Probability", size = "13")
#plt.xscale("log")
#plt.yscale("log")
plt.legend()
plt.grid()
plt.show()
#%%

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
#plt.plot(bin_4[0], bin_4[1], 'x')
#plt.plot(bin_8[0], bin_8[1], 'x')
#plt.plot(bin_16[0], bin_16[1], 'x')
#plt.plot(bin_32[0], bin_32[1], 'x')
#plt.plot(bin_64[0], bin_64[1], 'x')
#plt.plot(bin_128[0], bin_128[1], 'x')
#plt.plot(bin_256[0], bin_256[1], 'x')

plt.plot(bin_4[0], bin_4[1])
plt.plot(bin_8[0], bin_8[1])
plt.plot(bin_16[0], bin_16[1])
plt.plot(bin_32[0], bin_32[1])
plt.plot(bin_64[0], bin_64[1])
plt.plot(bin_128[0], bin_128[1])
plt.plot(bin_256[0], bin_256[1])

plt.xscale("log")
plt.yscale("log")
plt.xlabel("$s$", size = "14")
plt.ylabel("$\~P(s; L)$", size = "14")
plt.legend()
plt.grid()
plt.show()
#%% 
""" Cutoff avalanche size vs System size; Attempt D """
plt.figure()

s_c = [np.amax(bin_4[0]), np.amax(bin_8[0]), np.amax(bin_16[0]), np.amax(bin_32[0]), 
       np.amax(bin_64[0]), np.amax(bin_128[0]), np.amax(bin_256[0])]

def PLaw(x, m, k):
    y = m*(x**-k)
    return y 

arO = np.arange(0, 256, 0.0001)
p0 = np.array([1, 2.2])
p, cov = opt.curve_fit(PLaw, g, s_c, p0)
plt.plot(arO, PLaw(arO, p[0], p[1]), zorder=10, color = 'red')
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
""" Probability of avalanche size vs Avalanche size; Attempt tau """
plt.figure()

s_c = [np.amax(bin_4[0]), np.amax(bin_8[0]), np.amax(bin_16[0]), np.amax(bin_32[0]), 
       np.amax(bin_64[0]), np.amax(bin_128[0]), np.amax(bin_256[0])]

def PLaw(x, m, k):
    y = m*(x**-k)
    return y 

arO = np.arange(4, 256, 0.0001)
p0 = np.array([1, 2.2])
p, cov = opt.curve_fit(PLaw, g, s_c, p0)
plt.plot(arO, PLaw(arO, p[0], p[1]), zorder=10, color = 'red')
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
""" Task 3b """

#I = 150000
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
#%%
import math
K = 10

kth_mom = [[] for x in range(K)]
kth_moment = [[] for x in range(K)]
kth = {}

for e in range(len(g)):
    for k in range(1,K+1):
        s_k = []
        for b in range(len(dataTC_Complete[g[e],2])):
            #s_k.append((dataTC_Complete[g[e],2][b]**k)
            s_k.append(math.pow(dataTC_Complete[g[e],2][b], k))
    
        #kth_mom[k-1].append(s_k)
        #Var = np.sum(s_k)/len(dataTC_Complete[g[e],0])
        kth_moment[k-1].append((np.sum(s_k))/len(dataTC_Complete[g[e],0]))
            #kth_mom = []
#            kth[g[e],k] = (sum(kth_moment))/np.size(dataTC_Complete[g[e],0])
#        kth_moment = []
#%%            
#K = 6
#kth_ = [[] for x in range(K)]
#for e in range(len(g)):
#    for k in range(1,K+1):
#        sum_k = sum(np.array(dataTC_Complete[g[e],2])**k)
#        kt_ = sum_k/(I-t_c_av[e])
#        kth_.append(kt_)  

#for e in range(len(g)):
#    for k in [1,2,3,4,5,6,7]:
#        s_k = []
#        for b in range(len(dataTC_Complete[g[e],2])):
#            s_k.append((dataTC_Complete[g[e],2][b])**k)
#        #kth_mom[k-1].append(s_k)
#        kth_moment[k-1].append((sum(kth_mom[k-1]))/np.size(dataTC_Complete[g[e],0]))
#            #kth_mom = []
##            kth[g[e],k] = (sum(kth_moment))/np.size(dataTC_Complete[g[e],0])
##        kth_moment = []


#%% # PLOTTING
plt.figure()

#for k in range(1,K+1):
#    plt.plot(g, kth_moment[0], label = "g[e]")

plt.plot(g, kth_moment[0], label = "k = 1")
plt.plot(g, kth_moment[1], label = "k = 2")
plt.plot(g, kth_moment[2], label = "k = 3")
plt.plot(g, kth_moment[3], label = "k = 4")
plt.plot(g, kth_moment[4], label = "k = 5")
plt.plot(g, kth_moment[5], label = "k = 6")
plt.plot(g, kth_moment[6], label = "k = 7")
plt.plot(g, kth_moment[7], label = "k = 8")
plt.plot(g, kth_moment[8], label = "k = 9")
#plt.plot(g, kth_moment[0])

plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.legend()
plt.show()






















