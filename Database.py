# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 20:26:39 2021

@author: 44743
"""
"""
Database """

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pickle
#%%
"""
Experimental

D = 2.253531542387533 pm 0.1
tau = 1.539292270260908 pm 0.1



"""
#%%
"""
Testing

p = 0.5
Time Averaged Height 
L = 16      h_av = 26.53 pm np.sqrt(1.29)
L = 32      h_av = 53.89 pm np.sqrt(1.79)


p = 0 A & C
Time Averaged Height 
L = 16      h_av = 256 # No error since one config in steady state 
L = 32
L = 128     h_av = 256 

p = 1 B & D

p = 0.05 E

p = 0.95 F
"""
#plt.figure()
#Sys = np.array([16,32,128])
#Np0 = np.array([272,1056,16512])
#Np1 = np.array([136,528,8256])
#
#plt.plot(Sys, Np0, 'x', label = "p = 0")
#plt.plot(Sys, Np1, 'x', label = "p = 1")
#
#plt.grid()
#plt.legend()
#plt.show()




#%%

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pickle

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

np.where(dataB[128,1][3] != 8256)
Out[3]: (array([0], dtype=int64),)

len(np.where(dataB[128,1][3] != 8256))
Out[4]: 1

len(np.where(dataB[128,1][3] != 8257))
Out[5]: 1

A = np.array([4,4,4,4,4,4,4,4,4,4,4])

np.var(A)
Out[7]: 0.0

"""
Testing

p = 0.5
Time Averaged Height 
L = 16      h_av = 26.53 pm 1.29
L = 32      h_av = 53.89 pm 1.79


p = 0 A & C
Time Averaged Height 
L = 16      h_av = 256 # No error since one config in steady state 
L = 32
L = 128     h_av = 256 

p = 1 B & D
"""
plt.figure()
Sys = np.array([16,32,128])
Np0 = np.array([272,1056,16512])
Np1 = np.array([136,528,8256])

plt.plot(Sys, Np0, 'x', label = "p = 0")
plt.plot(Sys, Np1, 'x', label = "p = 1")

plt.grid()
plt.legend()
plt.show()
#%%
#50000-49757
#Out[9]: 243
#
#np.mean(dataE[16,1][0][243:])
#Out[10]: 31.410957252245915
#
#np.sqrt(np.var(dataE[16,1][0][243:]))
#Out[11]: 0.5487058975341672
#
#np.sqrt(np.var(dataE[16,1][2][243:]))
#Out[12]: 21.314978940400458
#
#np.mean(dataE[16,1][2][243:])
#Out[13]: 15.994654018530056
#
#np.mean(dataE[16,1][1][243:])
#Out[14]: 270.8416906163957
#
#np.sqrt(np.var(dataE[16,1][1][243:]))
#Out[15]: 1.493465726929039
#
#np.sqrt(np.var(dataE[16,1][3]))
#Out[16]: 1.493465726929039
#
#np.mean(dataE[16,1][3])
#Out[17]: 270.8416906163957
#
#np.mean(dataE[16,1][0])
#Out[18]: 31.35164
#
#np.mean(dataE[16,1][0])
#Out[19]: 31.35164
#
#np.mean(dataE[32,1][0])
#Out[20]: 62.62394
#
#np.sqrt(np.var(dataE[32,1][0][1024:]))
#Out[21]: 0.5518841040786128
#
#np.mean(dataE[32,1][0][1024:])
#Out[22]: 63.07956958510291
#
#np.mean(dataE[32,1][1][1024:])
#Out[23]: 1052.5996814766415
#
#np.sqrt(np.var(dataE[32,1][1][1024:]))
#Out[24]: 3.303409665231896
#
#np.sqrt(np.var(dataE[128,1][0][16267:]))
#Out[25]: 0.6698082398796831
#
#np.mean(dataE[128,1][0][16267:])
#Out[26]: 253.7240387750867
#
#np.mean(dataE[128,1][1][16267:])
#Out[27]: 16444.706548483682
#
#np.sqrt(np.var(dataE[128,1][1][16267:]))
#Out[28]: 28.32265161134407
#
#np.mean(dataF[16,1][0][142:])
#Out[29]: 17.496289462072284
#
#np.sqrt(np.var(dataF[16,1][0][142:]))
#Out[30]: 0.8620828973293532
#
#np.mean(dataF[16,1][1][142:])
#Out[31]: 156.35795659673474
#
#np.sqrt(np.var(dataF[16,1][1][142:]))
#Out[32]: 7.766546053888167
#
#np.sqrt(np.var(dataF[16,1][3]))
#Out[33]: 7.766546053888167
#
#np.sqrt(np.var(dataF[32,1][0]))
#Out[34]: 1.968178102103567
#
#np.mean(dataF[23,1][0])
#Traceback (most recent call last):
#
#  File "<ipython-input-35-16f529bc138a>", line 1, in <module>
#    np.mean(dataF[23,1][0])
#
#KeyError: (23, 1)
#
#np.mean(dataF[32,1][0])
#Out[36]: 35.44828
#
#np.mean(dataF[32,1][3])
#Out[37]: 609.1319437417789
#
#np.sqrt(np.var(dataF[32,1][3]))
#Out[38]: 19.013233821264446
#
#np.mean(dataF[128,1][0][9306:])
#Out[39]: 145.3860274241903
#
#np.sqrt(np.var(dataF[128,1][0][9306:]))
#Out[40]: 1.6125315206779656
#
#np.sqrt(np.var(dataF[128,1][3][9306:]))
#Out[41]: 115.4313233481922
#
#np.sqrt(np.var(dataF[128,1][3]))
#Out[42]: 113.34697499476155
#
#np.mean(dataF[128,1][3])
#Out[43]: 9536.976286430432


