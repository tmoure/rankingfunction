import tensorflow as tf
import numpy as np
from traing_data import *
from itertools import chain
from sklearn.metrics import r2_score
import math
from countC import *

# 添加网络层
def add_layer(inputs, input_size, output_size, n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.cast(tf.Variable(tf.random_normal([output_size, input_size]), name='W'), dtype=tf.float64)
            tf.summary.histogram(layer_name + '/weights', weights)
        with tf.name_scope('wx'):
            wx = tf.matmul(weights, inputs)
            #添加dropout
            wx = tf.nn.dropout(wx,keep_prob)
            tf.summary.histogram(layer_name + '/wx', wx)

        if activation_function == None:
            outputs = wx
        else:
            outputs = activation_function(wx)
        tf.summary.histogram(layer_name + '/ouputs', outputs)
        return outputs, weights


# 导入训练数据以及测试数据
f1_inputs, f2_inputs, f1_outputs, f2_outputs = gettraining_data()
# x_train, x_test, y_train, y_test = data_split(f1_inputs, f1_outputs, 0.3)
# x1_train, x1_test, y1_train, y1_test = data_split(f2_inputs, f2_outputs, 0.3)

x_data = np.hstack((f1_inputs, f2_inputs))  # 用于训练模型的数据
y_data = np.hstack((f1_outputs, f2_outputs))

# x_data = np.hstack((x_train, x1_train))  # 用于训练模型的数据
# y_data = np.hstack((y_train, y1_train))

with tf.name_scope('inputs'):
    x_inputs = tf.placeholder(tf.float64, [3, None], name='x_input')
    y_outputs = tf.placeholder(tf.float64, [1, None], name='y_output')
    keep_prob = tf.placeholder(tf.float64,name = 'keep_prob')


#多层网络
# layer1,weights1 = add_layer(x_inputs,3,3,n_layer=1,activation_function= tf.nn.sigmoid)
# layer2,weights2 = add_layer(layer1,3,2,n_layer=2,activation_function= tf.nn.sigmoid)
# prediction, weights11 = add_layer(layer2, 2, 1, n_layer=6, activation_function=tf.nn.sigmoid)

#layer3,weights3 = add_layer(layer2,2,3,n_layer=3,activation_function= tf.nn.sigmoid)
# layer4,weights4 = add_layer(layer3,3,2,n_layer=4,activation_function= tf.nn.sigmoid)
# layer5,weights5 = add_layer(layer4,2,3,n_layer=5,activation_function= tf.nn.sigmoid)
# layer6,weights6 = add_layer(layer5,3,2,n_layer=6,activation_function= tf.nn.sigmoid)
# layer7,weights7 = add_layer(layer6,2,3,n_layer=7,activation_function= tf.nn.sigmoid)
# layer8,weights8 = add_layer(layer7,3,2,n_layer=8,activation_function= tf.nn.sigmoid)
# layer9,weights9 = add_layer(layer8,2,3,n_layer=9,activation_function= tf.nn.sigmoid)
# layer10,weight10 = add_layer(layer9,3,2,n_layer=10,activation_function= tf.nn.sigmoid)

#双层网络
layer1,weights1 = add_layer(x_inputs,3,3,n_layer=1,activation_function= tf.nn.sigmoid)
prediction, weights2 = add_layer(layer1, 3, 1, n_layer=3, activation_function=tf.nn.sigmoid)

# weights = weights1 + weights1
with tf.name_scope('loss'):
    # 使用交叉熵
    # loss = tf.reduce_mean(tf.reduce_sum(-y_data * tf.log(tf.clip_by_value(prediction, 1e-50, 1.0)) -
    #                                       (1 - y_data) * tf.log(tf.clip_by_value((1 - prediction), 1e-50, 1.0))))
    # 加入正则化
    # loss = tf.reduce_mean(tf.reduce_sum(-y_data * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)) -
    #   (1 - y_data) * tf.log(tf.clip_by_value((1 - prediction), 1e-10, 1.0)))) + tf.reduce_mean(tf.reduce_sum(tf.square(weights)))
    # clip_by_value函数将y限制在1e-10和1.0的范围内，防止出现log0的错误，即防止梯度消失或爆发

    # 使用均方误差
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.clip_by_value(prediction, 1e-50, 1.0) - y_data)))
    # 加入正则化
    # L1
    #loss = tf.reduce_mean(tf.reduce_sum(tf.abs(tf.clip_by_value(prediction, 1e-50, 1.0) - y_data)))
    # L2
    # loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.clip_by_value(prediction, 1e-50, 1.0) - y_data))) + tf.reduce_mean(tf.reduce_sum(tf.abs(weights)))

    #使用R-square
    #loss = 1 - tf.reduce_mean(tf.reduce_sum(tf.square(tf.clip_by_value(prediction, 1e-50, 1.0) - y_data)))/(np.var(y_data))

    #使用
    tf.summary.scalar('loss', loss)

# 使用交叉熵比使用均方误差迭代更慢
with tf.name_scope('train'):
    train_step = tf.train.AdadeltaOptimizer(learning_rate=0.15).minimize(loss)
    #train_step = tf.train.AdamOptimizer(learning_rate = 0.15).minimize(loss)
    #train_step = tf.train.RMSPropOptimizer(learning_rate=0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)


def compute_accuracy(x, y):
    global prediction

    y_pre = list(chain.from_iterable(list(sess.run(prediction, feed_dict={x_inputs: x,keep_prob:1}))))
    y_pre1 = list(chain.from_iterable(list(sess.run(prediction, feed_dict={x_inputs: y,keep_prob:1}))))

    num = 0

    for i in range(len(y_pre)):
        if y_pre[i] - y_pre1[i] >= 1e-3:
            num = num + 1

    accuracy = num / len(y_pre)

    # if(accuracy > 0.9998):
    #     for i in range(len(y_pre)):
    #         if y_pre[i] - y_pre1[i] < 1e-3*2:
    #         #np.set_printoptions(precision=10)
    #             print("++++++++++++++++++++++")
    #             print(i)
    #             print(f1_inputs[0][i])
    #             print(f1_inputs[1][i])
    #             #print(f1_inputs[2][i])
    #             print("+++++++++++++++")
    #             print(f2_inputs[0][i])
    #             print(f2_inputs[1][i])
    #             #print(f2_inputs[2][i])
    #             print("++++++++++++++++++++++")
    #             print(y_pre[i] - y_pre1[i])

    return accuracy

for i in range(50000):

    sess.run(train_step, feed_dict={x_inputs: x_data, y_outputs: y_data,keep_prob:1})
    if i % 100 == 0:
        np.set_printoptions(precision=16)

        result = sess.run(merged, feed_dict={x_inputs: x_data, y_outputs: y_data,keep_prob:1})
        writer.add_summary(result, i)
        print("++++++++++++++++++++++++")
        print("%s iterations" % i)
        print("weights of w_ji", sess.run(weights1))
        print("weights of w_i",sess.run(weights2))

        print("the value of c",countc2(sess.run(weights1),sess.run(weights2)))

        # print("第一层的输出为", sess.run(layer1,feed_dict={x_inputs:x_data,keep_prob:1}))
        # print("第二层的输出为",sess.run(prediction,feed_dict={x_inputs:x_data,keep_prob:1}))
        # print("------------------------")
        #print("损失值为",sess.run(loss, feed_dict={x_inputs: x_data, y_outputs: y_data,keep_prob:1}))
        #print("训练集准确率为", compute_accuracy(x_train, x1_train))
        print("the accuracy of test-set", compute_accuracy(f1_inputs, f2_inputs))



