import numpy as np
import math
def countc1(m,n): #采用的是第一种放缩方式
    m1 = np.delete(m,1,axis=1).reshape(1,3)
    temp = m1*n #得到[[w11*w1 w12*w2 w13*w3]]的值
    c = 0
    for i in temp[0]:
        c += (abs(1/16*i)+abs(1/8*i)) #modify the coefficients
    return c

def countc2(m,n):

    m1 = np.delete(m,2,axis=1)  #删掉不需要的那一列

    for i in range(3):
        for j in range(2):
            m1[i][j] = n[0][i]* m1[i][j] #得到[[w1*w11 w1*w21],[w2*w12 w2*w22],[w3*w13 w3*w23]]
    m2= m1.reshape(1,6) #get[[w1*w11,w1*w21,w2*w12, w2*w22,w3*w13, w3*w23]]
    parameter1 = [2,0,2,0,2,0] #f11,f21,f11,f21,f11,f21
    parameter2 = [0,1/math.exp(1),0,1/math.exp(1),0,1/math.exp(1)] #f12,f22,f12,f22,f12,f22
    c=0
    for i in m2[0]: # compute teh value of |u'|
        c+=(1/16)*abs(i)
    m3 = m2*parameter1
    for i in m3[0]:
        c+=(1/16)*abs(i) # get the value of |u'/u1|
    m4 = m2*parameter2
    for i in m4[0]:
        c += (1 / 16) * abs(i)# get the value of |u'/u2|
    return c

def countc3(m,n):
    m1 = np.delete(m,3,axis=1)
    for i in range(3):
        for j in range(3):
            m1[i][j] = n[0][i]* m1[i][j] # 得到[[w1*w11 w1*w21 w1*w31],
                                            # [w2*w12 w2*w22 w2*w32],
                                            #[w3*w13 w3*w23 w3*w33]]
    m2 = m1.reshape(1,9)
    parameter1 = [1,0,0,1,0,0,1,0,0] #f11,f21,f31,f11,f21,f31,f11,f21,f31
    parameter2 = [0,1.28,0,0,1.28,0,0,1.28,0] #f12,f22,f32,f12,f22,f32,f12,f22,f32
    parameter3 = [0,0,1,0,0,1,0,0,1] #f13,f23,f33,f13,f23,f33,f13,f23,f33
    c=0
    for i in m2[0]:  # compute teh value of |u'|
        c += (1 / 16) * abs(i)
    m3 = m2 * parameter1
    for i in m3[0]:
        c += (1 / 16) * abs(i)  # get the value of |u'/u1|
    m4 = m2 * parameter2
    for i in m4[0]:
        c += (1 / 16) * abs(i)  # get the value of |u'/u2|
    m5 = m2*parameter3
    for i in m5[0]:
        c+= (1/16)*abs(i) # get the value of |u'/u3|
    return c

#
# a=[[1,1,1],[2,2,2],[3,3,3]]
# b=[[2, 2, 2 ]]
# countc2(a,b)