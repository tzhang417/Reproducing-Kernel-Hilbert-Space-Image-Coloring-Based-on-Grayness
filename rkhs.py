# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 19:54:45 2021

@author: hanso
"""

import cv2
import numpy as np
import math

'''open images'''

orimg = cv2.imread('part_candy.png')
orimg = orimg.astype(np.float32)
img = cv2.imread('dmg_candy.png')
img = img.astype(np.float32)
gray = cv2.imread('gray_candy.png', cv2.IMREAD_GRAYSCALE)
gray = gray.astype(np.float32)
b,g,r = cv2.split(img)
print (gray)

'''parameters'''
t=10
gemma =0.1
p=2
Im = np.zeros((100,100))
k=0
for i in range (0,100):
    for j in range (0,100):
        if (i==j):
            Im[i,j] = 1

givenb =  np.zeros((100,1))
giveng =  np.zeros((100,1))
givenr =  np.zeros((100,1))
for j in range(0,50):
    givenb[j,0] = b[24,j]
    giveng[j,0] = g[24,j]
    givenr[j,0] = r[24,j]
    givenb[j+50,0] = b[25,j]
    giveng[j+50,0] = g[25,j]
    givenr[j+50,0] = r[25,j]
'''kernel'''
kD = np.zeros((100,100))
for j in range (0,50):
    for y in range (0,50):
            kD[j,y] = math.exp(-((gray[24,j] - gray[24,y])**p)/4/t)
            kD[j,y+50] = math.exp(-((gray[24,j] - gray[25,y])**p)/4/t)
            kD[j+50,y+50] = math.exp(-((gray[25,j] - gray[25,y])**p)/4/t)
            kD[j+50,y] = math.exp(-((gray[25,j] - gray[24,y])**p)/4/t)
            
kCD = np.zeros((2500,100))
for x in range (0,50):
    for y in range (0,50):
            for j in range (0,50):
                kCD[y+x*50,j] = math.exp(-((gray[x,y]-gray[24,j])**p)/4/t)
                kCD[y+x*50,j+50] = math.exp(-((gray[x,y]-gray[25,j])**p)/4/t)

print(kD)
'''linear sovler'''
Ab = np.linalg.solve(kD+gemma*100*Im, givenb)
Ag = np.linalg.solve(kD+gemma*100*Im, giveng)
Ar = np.linalg.solve(kD+gemma*100*Im, givenr)
Fb = np.matmul(kCD,Ab)
Fg = np.matmul(kCD,Ag)
Fr = np.matmul(kCD,Ar)
print(Ab)
print(Fb)


newb=np.zeros((50,50))
newg=np.zeros((50,50))
newr=np.zeros((50,50))
for i in range (0,50):
    for j in range(0,50):
        newb[i,j] = Fb[k,0]
        newg[i,j] = Fg[k,0]
        newr[i,j] = Fr[k,0]
        k+=1

print(k)
print(b[25,:])
print(newb[25,:])
'''show results'''
colored = cv2.merge([newb,newg,newr])
img   = np.clip(img, 0, 255).astype(np.uint8)
orimg   = np.clip(orimg, 0, 255).astype(np.uint8)
colored   = np.clip(colored, 0, 255).astype(np.uint8)
gray   = np.clip(gray, 0, 255).astype(np.uint8)
cv2.imshow('original'  , orimg)
cv2.imshow('given'  , img)
cv2.imshow('result'  , colored)
cv2.imshow('gray'  , gray)
cv2.waitKey(0)
cv2.destroyAllWindows()