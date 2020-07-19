import numpy as np
from sympy import *
import math
import time

#二维
def func2(i,j,n):
    L = {
        # 袁月循环
        1: i + j - 1 ,  # 自建循环
        2: i ** 2 + j ** 2 - 1,  # 循环1
        3: {1:j**2 + 10 - (i + 6*j) ,2: i**2 + 6 - (4*i + j)}, #循环2
        4: {1: i - j - 1 ,2: i + j - 1 ,3: i - 2},  # 循环3
        5: {1:i + j - 1 ,2: i - 3 ,3: j ** 2 - 1},  # 循环4
        6: {1:i - 0 ,2: j - 2 * i - 1,3:j - 3},  # 循环5
        7: i - j,  # 循环7
        8: i ** 2 - i + 1 - j ** 2,  # 循环8
        9: i - 0,  # 循环9
        10: i -(j ** 2 - j),  # 循环10
        11: {1: j**4 + 1 - i * (j ** 2) ,2: j + 1},  # 循环11
        12: {1: i - 1 ,2: j ** 2 + 2 * i - 3 * j},  # 循环12
        13: {1: i - 0 ,2: 1 + 4 * (j ** 2) + 8 * (i**3) - (j**3 + j)},  # 循环13
        #自建
        32:{1:i ** 2 + j ** 2 - 1,2:cos(i)-0},
        33:i ** 2 + j ** 2 - 1,
        34:{1:i ** 2 + j ** 2 - 1,2:cos(i) - sin(j)}
    }
    return L[n]
def LOOP2(i,j,n):
    L = {  # 袁月循环
        1: i + j >= 1,  # 自建循环
        2: i ** 2 + j ** 2 <= 1,  # 循环1
        3: {1:j ** 2 + 10 <= i + 6 * j ,2: i ** 2 + 6 <= 4 * i + j},  # 循环2
        4: {1:i - j >= 1 ,2: i + j >= 1 ,3: i <= 2},  # 循环3
        5: {1:i + j >= 1 ,2: i <= 3 ,3: j ** 2 <= 1},  # 循环4
        6: {1:i >= 0 ,2: j - 2 * i >= 1,3:j <=3},  # 循环5
        7: i >= j,  # 循环7
        8: i ** 2 - i + 1 <= j ** 2,  # 循环8
        9: i >= 0,  # 循环9
        10: i >= j ** 2 - j,  # 循环10
        11: {1:j**4 + 1 <= i * (j ** 2) ,2: j <= -1},  # 循环11
        12: {1:i >= 1 ,2: j ** 2 + 2 * i <= 3 * j},  # 循环12
        13: {1:i >= 0 ,2: 1 + 4 * (j ** 2) + 8 * (i**3) <= j**3 + j},  # 循环13

        32:{1:i ** 2 + j ** 2 <= 1,2:cos(i)>=0},
        33:i ** 2 + j ** 2 <= 1,
        34:{1:i ** 2 + j ** 2 <= 1,2:cos(i) - sin(j) >= 0}
    }

    return L[n]

#三维
def func3(i,j,k,n):
    L = {
        1: {1:k**4 + 1 -(i + j + k ** 2) ,2:k - i ** 2},  # 循环6
        2: {1:j - 1 ,2: j - i ,3: k - 0},  # 循环17
        3: {1:i - j,2:j,3:1-i},  # 循环18改动
        4: {1:5 * i ** 2 + 4 * k ** 2 - 40 * j ,2: k + j + 1},  # 循环15
        5: {1:i - 0 ,2: i + j - 0},  # 循环14

        14:{1:i**2 + j**2 + k**2 - 1 ,2: cos(i) - sin(j) - 0}
    }
    return L[n]

def LOOP3(i,j,k,n):
    L = {
        1: {1:k**4 + 1 <= i + j + k ** 2 ,2: k <= i ** 2},  # 循环6
        2: {1:j >= 1 ,2: j <= i ,3: k >= 0},  # 循环17
        3: {1:i >= j,2:j>=0,3:1>=i},  # 循环18改动
        4: {1:5 * i ** 2 + 4 * k ** 2 <= 40 * j ,2: k + j <= -1},  # 循环15
        5: {1:i >= 0 ,2: i + j >= 0},  # 循环14

        14:{1:i**2 + j**2 + k**2 <= 1 ,2: cos(i) - sin(j) >= 0}
    }
    return L[n]


t1 = time.time()
#二维
# m = np.linspace(0.002,0.998,499)
#三维
m = np.linspace(0.01 * 2 / 3, 1 - 0.01 * 2 / 3, 149)
n = np.linspace(0.01 * 2 / 3, 1 - 0.01 * 2 / 3, 149)

# m = np.linspace(0.01*4/3, 1-0.01*4/3, 74)
# n = np.linspace(0.01*4/3, 1-0.01*4/3, 74)
x=Symbol('x',real = True)
y=Symbol('y',real = True)
z=Symbol('z',real = true)
datax = []
datay = []
dataz = []

#二维:循环不等式条件只有一个约束
def getlinedatax_2d(k):
    for i in m:
        ans = solve(func2(log(i/(1-i)),y,k), y)
        for j in ans:
            if j.is_real:
                datax.append(i)
                datay.append(min(np.float64(1 / (1 + exp(-j))),0.998))

def getlinedatay_2d(k):
    for i in m:
        ans = solve(func2(x, (log(i / (1 - i))), k),x)
        for j in ans:
            if j.is_real:
                datax.append(min(np.float64(1 / (1 + exp(-j))),0.998))
                datay.append(i)

#二维：循环不等式有两个约束
def getlinedatax2_2d(k1,k2,k3):
    for i in m:
        ans = solve(func2(math.log(i / (1 - i)), y, k1)[k2], y)
        for j in ans:
            if j.is_real:
                if LOOP2(math.log(i / (1 - i)), j, k1)[k3]:
                    datax.append(i)
                    datay.append(np.float64(1 / (1 + exp(-j))))
def getlinedatay2_2d(k1,k2,k3):
    for i in m:
        ans = solve(func2(x, (math.log(i / (1 - i))), k1)[k2], x)
        for j in ans:
            if j.is_real:
                if LOOP2(j, math.log(i / (1 - i)), k1)[k3]:
                    datax.append(np.float64(1 / (1 + exp(-j))))
                    datay.append(i)

#二维：循环不等式有三个约束
def getlinedatax3_2d(k1,k2,k3,k4):
    for i in m:
        ans = solve(func2(math.log(i / (1 - i)), y, k1)[k2], y)
        for j in ans:
            if LOOP2(math.log(i / (1 - i)), j, k1)[k3] and LOOP2(math.log(i / (1 - i)), j, k1)[k4]:
                datax.append(i)
                datay.append(np.float64(1 / (1 + exp(-j))))
def getlinedatay3_2d(k1,k2,k3,k4):
    for i in m:
        ans = solve(func2(math.log(i / (1 - i)), y, k1)[k2], y)
        for j in ans:
            if LOOP2(math.log(i / (1 - i)), j, k1)[k3] and LOOP2(math.log(i / (1 - i)), j, k1)[k4]:
                datax.append(np.float64(1 / (1 + exp(-j))))
                datay.append(i)

#三维：循环不等式一个约束
def getlinedataxy_3d(k):
    for i in m:
        for j in n:
            ans = solve(func3(log(i/(1-i)),log(j/(1-j)),z,k), z)
            for k in ans:
                if k.is_real:
                    datax.append(i)
                    datay.append(j)
                    dataz.append(np.float64(1 / (1 + exp(-k))))

def getlinedatayz_3d(k):
    for i in m:
        for j in n:
            ans = solve(func3(x, (log(i / (1 - i))), (log(j / (1 - j))),k),x)
            for k in ans:
                if k.is_real:
                    datax.append(np.float64(1 / (1 +  exp(-k))))
                    datay.append(i)
                    dataz.append(j)

def getlinedatazx_3d(k):
    for i in m:
        for j in n:
            ans = solve(func3((log(i / (1 - i))), y,(log(j / (1 - j))),k),y)
            for k in ans:
                if k.is_real:
                    datax.append(i)
                    datay.append(np.float64(1 / (1 + exp(-k))))
                    dataz.append(j)

#三维：两个约束
def getlinedataxy2_3d(k1,k2,k3):
    for i in m:
        for j in n:
            ans = solve(func3(log(i/(1-i)),log(j/(1-j)),z,k1)[k2], z)
            for k in ans:
                if k.is_real:
                    if LOOP3(log(i/(1-i)),log(j/(1-j)),k, k1)[k3]:
                        datax.append(i)
                        datay.append(j)
                        dataz.append(np.float64(1 / (1 + exp(-k))))

def getlinedatayz2_3d(k1,k2,k3):
    for i in m:
        for j in n:
            ans = solve(func3(x, (log(i / (1 - i))), (log(j / (1 - j))),k1)[k2],x)
            for k in ans:
                if k.is_real:
                    if LOOP3(k,log(i / (1 - i)), log(j / (1 - j)), k1)[k3]:
                        datax.append(np.float64(1 / (1 + exp(-k))))
                        datay.append(i)
                        dataz.append(j)

def getlinedatazx2_3d(k1,k2,k3):
    for i in m:
        for j in n:
            ans = solve(func3((log(i / (1 - i))), y,(log(j / (1 - j))),k1)[k2],y)
            for k in ans:
                if k.is_real:
                    if LOOP3(log(i / (1 - i)),k, log(j / (1 - j)), k1)[k3]:
                        datax.append(i)
                        datay.append(np.float64(1 / (1 + exp(-k))))
                        dataz.append(j)

#三维有三个约束
def getlinedataxy3_3d(k1,k2,k3,k4):
    for i in m:
        for j in n:
            ans = solve(func3(log(i/(1-i)),log(j/(1-j)),z,k1)[k2], z)
            for k in ans:
                if k.is_real:
                    if LOOP3(log(i/(1-i)),log(j/(1-j)),k, k1)[k3] and LOOP3(log(i/(1-i)),log(j/(1-j)),k, k1)[k4]:
                        datax.append(i)
                        datay.append(j)
                        dataz.append(np.float64(1 / (1 + exp(-k))))

def getlinedatayz3_3d(k1,k2,k3,k4):
    for i in m:
        for j in n:
            ans = solve(func3(x, (log(i / (1 - i))), (log(j / (1 - j))),k1)[k2],x)
            for k in ans:
                if k.is_real:
                    if LOOP3(k,log(i / (1 - i)), log(j / (1 - j)), k1)[k3] and LOOP3(k,log(i / (1 - i)), log(j / (1 - j)), k1)[k4]:
                        datax.append(np.float64(1 / (1 + exp(-k))))
                        datay.append(i)
                        dataz.append(j)

def getlinedatazx3_3d(k1,k2,k3,k4):
    for i in m:
        for j in n:
            ans = solve(func3((log(i / (1 - i))), y,(log(j / (1 - j))),k1)[k2],y)
            for k in ans:
                if k.is_real:
                    if LOOP3(log(i / (1 - i)),k, log(j / (1 - j)), k1)[k3] and LOOP3(log(i / (1 - i)),k, log(j / (1 - j)), k1)[k4]:
                        datax.append(i)
                        datay.append(np.float64(1 / (1 + exp(-k))))
                        dataz.append(j)

#二维
#一个约束
# getlinedatax_2d(2)
# getlinedatay_2d(2)

#两个约束
# getlinedatax2_2d(13,1,2)
# getlinedatay2_2d(13,1,2)
# getlinedatax2_2d(13,2,1)
# getlinedatay2_2d(13,2,1)

#三个约束
#
# getlinedatax3_2d(6,1,2,3)
# getlinedatay3_2d(6,1,2,3)
# getlinedatax3_2d(6,2,1,3)
# getlinedatay3_2d(6,2,1,3)
# getlinedatax3_2d(6,3,1,2)
# getlinedatay3_2d(6,3,1,2)

#三维
#一个约束
# getlinedataxy_3d(14)
# getlinedatayz_3d(14)
# getlinedatazx_3d(14)

#两个约束
# getlinedataxy2_3d(4,1,2)
# getlinedatayz2_3d(4,1,2)
# getlinedatazx2_3d(4,1,2)
# getlinedataxy2_3d(4,2,1)
# getlinedatayz2_3d(4,2,1)
# getlinedatazx2_3d(4,2,1)

#三个约束
# getlinedataxy3_3d(3,1,2,3)
# getlinedatayz3_3d(3,1,2,3)
# getlinedatazx3_3d(3,1,2,3)
# getlinedataxy3_3d(3,2,1,3)
# getlinedatayz3_3d(3,2,1,3)
# getlinedatazx3_3d(3,2,1,3)
# getlinedataxy3_3d(3,3,1,2)
# getlinedatayz3_3d(3,3,1,2)
# getlinedatazx3_3d(3,3,1,2)


