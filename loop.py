import math
import numpy as np
ini_data2 = [[],[0.8,0.8],[0.5,0.5],[0.90,0.94],[0.90,0.5],[0.90,0.5],[0.6,0.90],[0.6,0.2],[0.6,0.1],[0.6,0.6],[0.6,0.6],[0.98,0.25],[0.732,0.74],[0.6,0.98]]
ini_data3 = [[],[0.5,0.8,0.5],[0.8,0.8,0.8],[0.5,0.5,0.5],[0.04,0.96,0.015],[0.8,0.8,0.8]]
#一维

def loop1(i,n):
    L = {#yuanyue1
        1:i >= 1 and i <= 3,   #循环16
        2:i**2 - 3*i + 2 <=0 , #自建循环
        3:i>=4,
        4:i>=1,
        #自建循环
        5:math.sin(i) >= 0 and i>=1 and i <= 6,
        6:math.sin(i) >= 0 and i>=1 and i <= 6

    }
    return L[n]

def update1(i,n):
    L = {#yuanyue1
        1:[5*i - i**2],   #循环16
        2:[5*i - i**2],
        3:[-2*i+4],
        4:[-i],
        #自建循环
        5:[-i],
        6:[i+1+math.cos(i)**2]

    }
    return L[n]


#二维
def loop2(i,j,n):
    L = {   #袁月循环
            1:i + j >= 1,           #自建循环
            2:i**2 + j**2 <= 1,    #循环1
            3:i>=1 and i<=j,     #zijianxunhuan
            4:i - j >= 1 and i + j >= 1 and i <=2,  #循环3改动
            5:i +j >= 1 and i <= 3 and j**2 <=1,      #循环4
            6:i >= 0 and j - 2*i >= 1 and j <= 3,         #循环5
            7:i >= j,                    #循环7
            8:i**2 - i +1 <= j**2,            #循环8
            9:i**2<=1 and j**2<=1,                   #循环9
            10:i >= 1 + j**2 - j,             #循环10
            11:math.pow(j,4) + 1 <= i*(j**2) and j <= -1,       #循环11
            12:i >= 1 and j**2 + 2*i <= 3*j,          #循环12
            13:i >= 0 and 1 + 4*(j**2) + 8*math.pow(i,3) <= math.pow(j,3) + j,        #循环13
            #不终止的例子
            15:i>=0,#8
            16:j>=i,#9
            17:i <= 5,#10
            18:4*i-5j>=0,#12
            19:10>=i,#13
            20:i-j>=0 and j>=0,#14
            21:i+j>=4,#15
            22:i+j>=3,#16
            23:4*i+j>=0,#17
            24:i >= 1 and j >= i,#18 添加的例子
            #可能终止的例子
            25:i > 0 and j >= i,#22
            26:i >= 0,#23
            27:i-j>0,#26
            28:i>0,#27
            29:i>0 and j**2<=1,#28
            30:i+j>0,#29
            31:i>0,#31

            #自建的例子
            32: i**2 + j**2 <= 1 and math.cos(i) >= 0,
            33: i**2 + j**2 <= 1,
            34: i**2 + j**2 <= 1 and math.cos(i) >= 0,

            35: 4*i+j>=1


        }

    return L[n]

def update2(i,j,n):
    L = {
        #yuanyue1
        1:[i-1,j-1],      #自建循环
        2:[i - j**2 + 1,j + i**2 -1], #循环1
        3:[i + j,-j], #循环2
        4:[i - 1,j - 1],#循环3`
        5:[-i,j**2 +j],#循环4
        6:[-i**2 - 4*(j**2) + 1,-i*j - 1],#循环5
        7:[i - 4*(j**2) - 4*j + 3,j - 2],#循环7
        8:[i**2 + j + 1,-j + 1],#循环8
        9:[i + j,j - 1],#循环9
        10:[-i,-j],#循环10
        11:[i + j - 1/j,j - 1],#循环11
        12:[1 + 1/(i**2),-i*j - 3*j + j**2 + 1],#循环12
        13:[-4*(j**2) + j - i**2,-1/(j + 1) - i*j],#循环13
        #下面是不终止的例子
        15: [i + j,-2*j],  # 8
        16: [i + j,j/2],  # 9
        17: [i - j,i + j],  # 10
        18: [2*i + 4*j,4*i],  # 12
        19: [-j,j + 1], # 13
        20: [3*i,j + 1],  # 14
        21: [3*i + j,2*j],  # 15
        22: [3*i - 2,2*j],  # 16
        23: [-2*i + 4*j,4*i],  # 17
        24: [1/16*i + 2 + math.exp(-i),1/16*j + 1],
        #例子
        25:[2*i,j+1],#22
        26:[i-2*j, j + 1],#23
        27:[-i+j, j + 1],#26
        28:[j, j - 1],#27
        29:[i+j-5, -2*j],#28
        30:[i-1, -2*j],#29
        31:[i+j, -j-1],#31

        32:[i**2 + 1,j**2 -1],
        33:[i-1-math.cos(i),j-1-math.cos(j)],
        34:[i-1-math.sin(i)**2,j-1-math.cos(i)**3],

        35:[-2*i+4*j,4*i]



    }
    return L[n]

#三维
def loop3(i,j,k,n):
    L = {
        1: math.pow(k, 4) + 1 <= i + j + k ** 2 and k <= i ** 2,       #循环6
        2: j >= 1 and j <= i and k >= 0,         #循环17
        3: i >= j and j >= 0 and i <= 1,          #循环18(自建循环)
        4: i >= j and j >= 0 and i <= 1,       #循环15
        5: i >= 0 and i + j >= 0, #循环14
        #另一份文章的循环
        6:i>0 and i+j>=0,#40
        7:i<j,#33
        8:i>0,#34
        9:i+j>=0 and i<=k,#35
        10:i>0 and i<=k,#36
        11:i>=0,#37
        12:i-j>0,#38

        13:i**2 + j**2 + k**2 <= 1 ,
        14:i**2 + j**2 + k**2 <= 1 and math.cos(i) >= 0
    }
    return L[n]

def update3(i,j,k,n):
    L = {
        1: [i + j, -k - 1, k**2 - 2*k],#循环6
        2: [-i, j, k + 1],#循环17
        3: [i + 1,j**2 + 2*i,-k],#循环18
        4: [i + 1,j - 1,-k],#循环15
        5: [i + j + (k - 1)/(k**2 + 1),-(k*(k+1))/(k**2 + 1),k**2], #循环14
        #
        6: [i + j + k,-k - 1,k],#40
        7: [i+1,k,k],#33
        8: [i+j,j+k,k-1],#34
        9: [2*i + j,j+1,k],#35
        10: [2*i + j,j+1,k],#36
        11: [i+j,k,-k-1],#37
        12: [-i+j,k,k+1],#38

        13: [i - 1 - math.cos(k), j - 1 - math.cos(i), k - 1 - math.sin(k)],
        14: [i - 1, j, math.sin(k)**3]
    }
    return L[n]

def cons1(n):
    L = {
        1:({'type': 'ineq', 'fun': lambda x: 3 - (-np.log((1 - x[0]) / x[0]))},
            {'type': 'ineq', 'fun': lambda x:(-np.log((1 - x[0]) / x[0])) - 1})
    }
    return L[n]

def cons2(n):
    L = {
        1:({'type': 'ineq', 'fun': lambda x: (-np.log((1 - x[0]) / x[0])) + (-np.log((1 - x[1]) / x[1])) - 1}),
        2:({'type': 'ineq', 'fun': lambda x: 1 - (-np.log((1 - x[0]) / x[0]))**2 - (-np.log((1 - x[1]) / x[1]))**2}),
        3: ({'type': 'ineq', 'fun': lambda x: (-np.log((1 - x[0]) / x[0])) + 6*(-np.log((1 - x[1]) / x[1])) - (-np.log((1 - x[1]) / x[1]))**2 - 10},
            {'type': 'ineq', 'fun': lambda x:4*(-np.log((1 - x[0]) / x[0])) + (-np.log((1 - x[1]) / x[1])) - (-np.log((1 - x[0]) / x[0]))**2 - 6}),
        4:({'type': 'ineq', 'fun': lambda x: (-np.log((1 - x[0]) / x[0])) - (-np.log((1 - x[1]) / x[1])) - 1},
           {'type': 'ineq', 'fun': lambda x: (-np.log((1 - x[0]) / x[0])) + (-np.log((1 - x[1]) / x[1])) - 1},
           {'type': 'ineq', 'fun': lambda x: 10 - (-np.log((1 - x[0]) / x[0]))}),
        5:({'type': 'ineq', 'fun': lambda x: (-np.log((1 - x[0]) / x[0])) + (-np.log((1 - x[1]) / x[1])) - 1},
           {'type': 'ineq', 'fun': lambda x: 3 - (-np.log((1 - x[0]) / x[0]))},
           {'type': 'ineq', 'fun': lambda x: 1 - (-np.log((1 - x[1]) / x[1]))**2}),
        6:({'type': 'ineq', 'fun': lambda x: (-np.log((1 - x[0]) / x[0])) - 1e-20},
           {'type': 'ineq', 'fun': lambda x: (-np.log((1 - x[1]) / x[1])) - 2*(-np.log((1 - x[0]) / x[0])) - 1}),
        7:({'type': 'ineq', 'fun': lambda x: (-np.log((1 - x[0]) / x[0])) - (-np.log((1 - x[1]) / x[1]))}),
        8:({'type': 'ineq', 'fun': lambda x:(-np.log((1 - x[1]) / x[1]))**2 - ((-np.log((1 - x[0]) / x[0]))**2 - (-np.log((1 - x[0]) / x[0])) + 1)}),
        9:({'type': 'ineq', 'fun': lambda x: (-np.log((1 - x[0]) / x[0])) - 1e-20}),
        10:({'type': 'ineq', 'fun': lambda x: (-np.log((1 - x[0]) / x[0])) - ((-np.log((1 - x[1]) / x[1]))**2 - (-np.log((1 - x[1]) / x[1])))}),
        11:({'type': 'ineq', 'fun': lambda x: (-np.log((1 - x[0]) / x[0]))*((-np.log((1 - x[1]) / x[1]))**2) - (math.pow((-np.log((1 - x[1]) / x[1])),4) + 1)},
           {'type': 'ineq', 'fun': lambda x: -1 - (-np.log((1 - x[1]) / x[1]))}),
        12:({'type': 'ineq', 'fun': lambda x: (-np.log((1 - x[0]) / x[0])) - 1},
           {'type': 'ineq', 'fun': lambda x: 3*(-np.log((1 - x[1]) / x[1])) - ((-np.log((1 - x[1]) / x[1]))**2 + 2*(-np.log((1 - x[0]) / x[0])))}),
        13:({'type': 'ineq', 'fun': lambda x: (-np.log((1 - x[0]) / x[0])) - 1e-20},
           {'type': 'ineq', 'fun': lambda x: math.pow((-np.log((1 - x[1]) / x[1])),3) + (-np.log((1 - x[1]) / x[1])) - (1 + 4*((-np.log((1 - x[1]) / x[1]))**2) + 8*math.pow((-np.log((1 - x[0]) / x[0])),3))}),
    }
    return L[n]

def cons3(n):
    L = {
        1: ({'type': 'ineq', 'fun':lambda x:(-np.log((1 - x[0]) / x[0])) + (-np.log((1 - x[1]) / x[1])) + (-np.log((1 - x[2]) / x[2])) ** 2 - (math.pow((-np.log((1 - x[2]) / x[2])), 4) + 1)},
            {'type': 'ineq', 'fun':lambda x:(-np.log((1 - x[0]) / x[0])) ** 2 - (-np.log((1 - x[2]) / x[2]))}),
        2: ({'type': 'ineq', 'fun':lambda x:(-np.log((1 - x[1]) / x[1])) - 1 },
            {'type': 'ineq', 'fun':lambda x: (-np.log((1 - x[0]) / x[0])) - (-np.log((1 - x[1]) / x[1]))},
            {'type': 'ineq', 'fun':lambda x: (-np.log((1 - x[2]) / x[2])) - 1e-20}),
        3: ({'type': 'ineq', 'fun':lambda x:(-np.log((1 - x[0]) / x[0])) - (-np.log((1 - x[1]) / x[1]))}),
        4: ({'type': 'ineq', 'fun':lambda x: 40 * (-np.log((1 - x[1]) / x[1])) - 5 * (-np.log((1 - x[0]) / x[0])) ** 2 + 4 * (-np.log((1 - x[2]) / x[2])) ** 2},
            {'type': 'ineq', 'fun':lambda x: -1 - (-np.log((1 - x[1]) / x[1])) - (-np.log((1 - x[2]) / x[2]))}),
        5: ({'type': 'ineq', 'fun':lambda x:(-np.log((1 - x[0]) / x[0])) - 1e-20 },
            {'type': 'ineq', 'fun':lambda x:(-np.log((1 - x[0]) / x[0])) + (-np.log((1 - x[1]) / x[1])) - 1e-20})
    }
    return L[n]


def x_range(n):
    L = {
        1: np.linspace(0.002,0.998,499),
        2: np.linspace(0.2,0.8,301),
        3: np.linspace(0.7,0.998,150),
        4: np.linspace(0.7,0.998,150),
        5: np.linspace(0.4, 0.998, 300),
        6: np.linspace(0.4, 0.998, 300),
        7: np.linspace(0.002, 0.998, 499),
        8: np.linspace(0.002, 0.998, 499),
        9: np.linspace(0.4, 0.998, 300),
        10: np.linspace(0.4, 0.998, 300),
        11: np.linspace(0.86, 0.998, 70),
        12: np.linspace(0.73, 0.76, 16),
        13: np.linspace(0.4, 0.998, 300),
        #三维
        14: np.linspace(0.01*4/3, 1-0.01*4/3, 74),
        15: np.linspace(0.6 + 0.01*4/3, 1-0.01*4/3, 28),#17
        16: np.linspace(0.01*4/3, 1-0.01*4/3, 74),
        17: np.linspace(0.01*4/3, 1-0.01*4/3, 74),
        18: np.linspace(0.4 + 0.01*4/3, 1-0.01*4/3, 43),


    }
    return L[n]

def y_range(n):
    L = {
        1: np.linspace(0.002, 0.998, 499),
        2: np.linspace(0.2,0.8,301),
        3: np.linspace(0.86, 0.998, 70),
        4: np.linspace(0.002, 0.998, 499),
        5: np.linspace(0.2, 0.8, 301),
        6: np.linspace(0.7,0.998,150),
        7: np.linspace(0.002, 0.998, 499),
        8: np.linspace(0.002, 0.998, 499),
        9: np.linspace(0.002, 0.998, 499),
        10: np.linspace(0.002, 0.998, 499),
        11: np.linspace(0.07, 0.276, 104),
        12: np.linspace(0.72, 0.9, 91),
        13: np.linspace(0.976, 0.998, 12),
        #三维
        14:np.linspace(0.01*4/3, 1-0.01*4/3, 74),
        15:np.linspace(0.6 + 0.01*4/3, 1-0.01*4/3, 28),
        16:np.linspace(0.01*4/3, 1-0.01*4/3, 74),
        17:np.linspace(0.4 + 0.01*4/3, 1-0.01*4/3, 43),
        18:np.linspace(0.01*4/3, 1-0.01*4/3, 74),
    }
    return L[n]

def z_range(n):
    L = {
        1:np.linspace(0.01*4/3, 1-0.01*4/3, 74),
        2:np.linspace(0.4 + 0.01*4/3, 1-0.01*4/3, 43),
        3:np.linspace(0.01*4/3, 1-0.01*4/3, 74),
        4:np.linspace(0.01*4/3, 0.4-0.01*4/3, 28),
        5:np.linspace(0.01*4/3, 1-0.01*4/3, 74),

    }
    return L[n]

