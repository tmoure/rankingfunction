import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from itertools import chain
from loop import *
from scipy.optimize import minimize
from sympy import *
import math
import tensorflow as tf
from getlinedata import*
D = []
D_sample_bound = []
D_sample_inside = []

# Recursive sampling
def hypercube(n,d):
    """
    :param n: dimension
    :param d: step
    :return: points
    """
    if n == 1:
        for p in np.arange(0, 1, d):
            yield [p]
    elif n > 1:
        for p in hypercube(n-1,d):
            for x in np.arange(0, 1, d):
                yield p + [x]

def sample(m):
    for i in hypercube(2, 0.002):
        D.append(i)
      # 初始化列表，存入打标签前的数据

    laterdata_f1 = []
    laterdata_f2 = []  # 初始化列表，存入打标签之后的数据
    for i in D:
        if loop2((-np.log((1 - i[0]) / i[0])),(-np.log((1 - i[1]) / i[1])),m):
            D_sample_inside.append(i)



def getdata1(i):
    predata_f1 = []  # 初始化列表，存入打标签前的数据

    laterdata_f1 = []
    laterdata_f2 = []  # 初始化列表，存入打标签之后的数据

    for f1 in i:
            # 在所有点中判断循环不等式条
        if loop1((-np.log((1 - f1) / f1)), 6):  # 调用loop函数中的循环
            predata_f1.append(f1)
              # 将满足的点存入之前初始化好的列表中

    np.random.seed(2)  # 随机数种子
    for i in range(len(predata_f1)):  # 分别将所得到的点打上所对应的标签
        m = np.random.random() % 0.5 + 0.5
        laterdata_f1.append([predata_f1[i], m])

        # 利用update函数做变量的更新
        [f1] = update1((-np.log((1 - predata_f1[i]) / predata_f1[i])),6)
        [f1_] = [1 / (1 + np.exp(-f1))]
        laterdata_f2.append([f1_, m - (np.random.random() % 0.5)])
    return predata_f1, laterdata_f1, laterdata_f2

def getdata2(i,j):

    predata_f1 = []
    predata_f2 = []

    #
    # for data in np.array(datax):
    #     predata_f1.append(data)
    # for data in np.array(datay):
    #     predata_f2.append(data)

    laterdata_f1 = []
    laterdata_f2 = []  # 初始化列表，存入打标签之后的数据

    for f1 in i:
        for f2 in j:
            # 在所有点中判断循环不等式条件
            if loop2((-np.log((1 - f1) / f1)), -np.log((1 - f2) / f2), 35) :  # 调用loop函数中的循环
                predata_f1.append(f1)
                predata_f2.append(f2)  # 将满足的点存入之前初始化好的列表中


    np.random.seed(2)  # 随机数种子
    for i in range(len(predata_f1)):  # 分别将所得到的点打上所对应的标签
        m = np.random.random() % 0.5 + 0.5

        laterdata_f1.append([predata_f1[i], predata_f2[i], m])

        # 利用update函数做变量的更新
        [f1, f2] = update2((-np.log((1 - predata_f1[i]) / predata_f1[i])), -np.log((1 - predata_f2[i]) / predata_f2[i]),35)
        [f1_, f2_] = [1 / (1 + np.exp(-f1)), 1 / (1 + np.exp(-f2))]
        #laterdata_f2.append([f1_, f2_, m - (np.random.random() % 0.2+0.3)])
        #laterdata_f2.append([f1_, f2_, m - (np.random.random() % 0.1+0.4)])
        laterdata_f2.append([f1_, f2_, m - 0.5])

    return predata_f1 ,predata_f2,laterdata_f1,laterdata_f2

def getdata3(i,j,k):

    predata_f1 = []
    predata_f2 = []  # 初始化列表，存入打标签前的数据
    predata_f3 = []

    laterdata_f1 = []
    laterdata_f2 = []  # 初始化列表，存入打标签之后的数据


    #边界打点
    for data in np.array(datax):
        predata_f1.append(data)
    for data in np.array(datay):
        predata_f2.append(data)
    for data in np.array(dataz):
        predata_f3.append(data)

    for f1 in i:
        for f2 in j:
            for f3 in k:
            # 在所有点中判断循环不等式条件
                if loop3((-np.log((1 - f1) / f1)), -np.log((1 - f2) / f2), -np.log((1 - f3) / f3),7):  # 调用loop函数中的循环
                    predata_f1.append(f1)
                    predata_f2.append(f2)
                    predata_f3.append(f3)   # 将满足的点存入之前初始化好的列表中

    np.random.seed(2)  # 随机数种子
    for i in range(len(predata_f1)):  # 分别将所得到的点打上所对应的标签
        m = np.random.random() % 0.5 + 0.5
        laterdata_f1.append([predata_f1[i], predata_f2[i], predata_f3[i],m])

        # 利用update函数做变量的更新
        [f1, f2, f3] = update3((-np.log((1 - predata_f1[i]) / predata_f1[i])), -np.log((1 - predata_f2[i]) / predata_f2[i]), -np.log((1 - predata_f3[i]) / predata_f3[i]),7)
        [f1_, f2_, f3_] = [1 / (1 + np.exp(-f1)), 1 / (1 + np.exp(-f2)), 1 / (1 + np.exp(-f3))]
        laterdata_f2.append([f1_, f2_, f3_, m - (np.random.random() % 0.2+0.3)])
    return predata_f1 ,predata_f2,predata_f3,laterdata_f1,laterdata_f2

def data_handling(m,n,i):
    #数据处理
    #将偏置项b加入输入数据之中（f1,f2,1）

    x1 = np.array(m)

    x2 = np.delete(x1,i,axis=1) #删除标签列

    x3 = np.ones(len(m)) #初始化：全部为1 长度与所得样本点个数一样

    f1_inputs = np.insert(x2,i,values= x3,axis= 1) #将上诉数据插入删除删除标签列后的数据之中
    #同理对样本点f2做相同操作
    x11 = np.array(n)
    x22 = np.delete(x11, i, axis=1)  # 删除标签列
    x33 = np.ones(len(m))  # 初始化：全部为1 长度与所得样本点个数一样
    f2_inputs = np.insert(x22, i, values=x33, axis=1)  # 将上诉数据插入删除删除标签列后的数据之中

    laterdata_f1_matrix = np.matrix(m)  # 将所得到的数据转化为矩阵形式
    laterdata_f2_matrix = np.matrix(n)

    f1_outputs = laterdata_f1_matrix[::,i]
    f2_outputs = laterdata_f2_matrix[::,i]

    return f1_inputs, f2_inputs, f1_outputs, f2_outputs
def gettraining_data():
    #一维
    #f1_ = np.linspace(0.001, 0.999, 999)
    #二维
    f1_ = np.linspace(0.002, 0.998, 499)
    f2_ = np.linspace(0.002, 0.998, 499)
    # f1_ = np.linspace(0.001, 0.999, 999)
    # f2_ = np.linspace(0.001, 0.999, 999)
    # f1_ = np.linspace(0.0002, 0.9998, 4999)
    # f2_ = np.linspace(0.0002, 0.9998, 4999)
    #三维
    # f1_ = np.linspace(0.01*4/3, 1-0.01*4/3, 74)
    # f2_ = np.linspace(0.01*4/3, 1-0.01*4/3, 74)
    # f3_ = np.linspace(0.01*4/3, 1-0.01*4/3, 74)# 在（0，1）中均匀取点

    # f1_ = np.linspace(0.01*2/3, 1-0.01*2/3, 149)
    # f2_ = np.linspace(0.01*2/3, 1-0.01*2/3, 149)
    # f3_ = np.linspace(0.01*2/3, 1-0.01*2/3, 149)

    # f1_ = np.linspace(0.005, 0.995, 199)
    # f2_ = np.linspace(0.005, 0.995, 199)
    # f3_ = np.linspace(0.005, 0.995, 199)


    #一维二维三维情况下调用getdata
    #predata_f1, laterdata_f1, laterdata_f2 = getdata1(f1_)
    predata_f1, predata_f2, laterdata_f1, laterdata_f2 = getdata2(f1_, f2_)
    #predata_f1 ,predata_f2,predata_f3,laterdata_f1,laterdata_f2 = getdata3(f1_,f2_,f3_)

    #三维散点图
    #u的标签
    # laterdata_f1_matrix = np.matrix(laterdata_f1) #将所得到的数据转化为矩阵形式
    #
    # x = laterdata_f1_matrix[:,0]    #绘制3d图
    # y = laterdata_f1_matrix[:,1]
    # z = laterdata_f1_matrix[:,2]
    #
    # plt.rcParams['font.sans-serif'] = ['SimHei'] #解决不能显示中文标题的问题
    # plt.rcParams['axes.unicode_minus'] = False
    #
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # plt.title('u的分布')
    # ax.set_zlabel('Z', fontdict={'size': 10, 'color': 'red'})
    # ax.set_ylabel('Y', fontdict={'size': 10, 'color': 'red'})
    # ax.set_xlabel('X', fontdict={'size': 10, 'color': 'red'})
    # ax.scatter(x, y, z,marker = '+',color = 'red')

    #u'的标签
    # laterdata_f2_matrix = np.matrix(laterdata_f2)
    # x1 = laterdata_f2_matrix[:,0]
    # y1 = laterdata_f2_matrix[:,1]
    # z1 = laterdata_f2_matrix[:,2]
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # plt.title('u1的分布')
    # ax.scatter(x1, y1, z1,marker = 'o',color = 'blue')
    # ax.set_zlabel('Z', fontdict={'size': 10, 'color': 'red'})
    # ax.set_ylabel('Y', fontdict={'size': 10, 'color': 'red'})
    # ax.set_xlabel('X', fontdict={'size': 10, 'color': 'red'})
    # plt.show()

    #将u与u'放入同一坐标系中
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(x1, y1, z1,marker = 'o',color = 'blue')
    # ax.scatter(x, y, z,marker = '+',color = 'red')
    # plt.title("u与u1的分布")
    # ax.set_zlabel('Z', fontdict={'size': 10, 'color': 'red'})
    # ax.set_ylabel('Y', fontdict={'size': 10, 'color': 'red'})
    # ax.set_xlabel('X', fontdict={'size': 10, 'color': 'red'})
    # plt.show()

    #一维样本点分布
    # plt.rcParams['font.sans-serif'] = ['SimHei'] #解决不能显示中文标题的问题
    # plt.rcParams['axes.unicode_minus'] = False
    # m = np.linspace(1,len(predata_f1),len(predata_f1)) #令横坐标为第几个点
    # plt.scatter(m,predata_f1,marker = "+",color = "red",label = "f1的采样分布")
    # plt.legend(loc = 'best')
    # plt.show()

    #二维样本点分布
    # plt.rcParams['font.sans-serif'] = ['SimHei'] #解决不能显示中文标题的问题
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.scatter(predata_f1,predata_f2,marker = "+",color = "red",label = "(u1,u2)")
    # plt.legend(loc = 'best')
    # plt.show()

    #三维样本点分布
    # predata = np.delete(laterdata_f1, 3, axis=1)
    # laterdata_f1_matrix = np.matrix(predata)  # 将所得到的数据转化为矩阵形式
    # x = laterdata_f1_matrix[:,0]    #绘制3d图
    # y = laterdata_f1_matrix[:,1]
    # z = laterdata_f1_matrix[:,2]
    #
    # plt.rcParams['font.sans-serif'] = ['SimHei'] #解决不能显示中文标题的问题
    # plt.rcParams['axes.unicode_minus'] = False
    #
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # plt.title('样本点(f1,f2,f3)的分布')
    # ax.set_zlabel('Z', fontdict={'size': 10, 'color': 'red'})
    # ax.set_ylabel('Y', fontdict={'size': 10, 'color': 'red'})
    # ax.set_xlabel('X', fontdict={'size': 10, 'color': 'red'})
    # ax.scatter(x, y, z,marker = '+',color = 'red')
    # plt.show()

    #处理数据成为可供神经网络输入
    f1_inputs, f2_inputs, f1_outputs, f2_outputs = data_handling(laterdata_f1,laterdata_f2,2) #最后一个参数是输入维度

    return f1_inputs.T,f2_inputs.T,f1_outputs.T,f2_outputs.T


#将数据划分为训练集以及测试集
def data_split(x,y,n):
    x = x.T
    y = list(chain.from_iterable(y.tolist()))
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=n, random_state=10)
    x_data = x_train.T
    x_test_data = x_test.T
    y_data = np.array(np.mat(y_train))
    y_test_data = np.array(np.mat(y_test))
    return x_data,x_test_data,y_data,y_test_data

f1_inputs,f2_inputs,f1_outputs,f2_outputs = gettraining_data()
# x_train,x_test,y_tarin,y_test = data_split(f1_inputs,f1_outputs,0.3)
# x1_train,x1_test,y1_tarin,y1_test = data_split(f2_inputs,f2_outputs,0.3)
#
x_data = np.hstack((f1_inputs, f2_inputs))  # 用于训练模型的数据
y_data = np.hstack((f1_outputs, f2_outputs))
# print("+++++++++++++++++++++++++++++++++++++")
# print(f1_inputs)
# print("+++++++++++++++++++++++++++++++++++++")
# print(f2_inputs)
# print("+++++++++++++++++++++++++++++++++++++")
# print(f1_outputs)
# print("+++++++++++++++++++++++++++++++++++++")
# print(f2_outputs)
# print("+++++++++++++++++++++++++++++++++++++")
# print(x_data)
# print("+++++++++++++++++++++++++++++++++++++")
# print(y_data)

#将样本点数据写入txt文件中
# sample_point = np.delete(f1_inputs.T,2,axis=1)
# np.savetxt(r'C:\Users\tanw\Desktop\sample_point.txt',sample_point,fmt='%f')

















