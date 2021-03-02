import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib
import platform
import shutil
import time
import DNN_base


def test_intergral(bachsize=10):
    learning_rate = 0.001
    lr_decay = 0.01
    input_dim = 1
    out_dim = 1
    hidden_layers = (8, 8, 6, 6, 4)
    act_func = 'tanh'

    flag2beta = 'beta'
    flag2S = 'Seff'
    input_dim2beta = input_dim + 1
    W2beta, B2beta = DNN_base.initialize_NN_random_normal2(input_dim2beta, out_dim, hidden_layers, flag2beta)
    W2S, B2S = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2S)

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (0)):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            X = tf.placeholder(tf.float32, name='X_recv', shape=[bachsize, input_dim])      # * 行 1 列
            Y = tf.placeholder(tf.float32, name='X_recv', shape=[bachsize, input_dim])      # * 行 1 列
            beta = tf.placeholder(tf.float32, name='Beta', shape=[bachsize, input_dim])     # * 行 1 列
            in_learning_rate = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')

            XY = tf.concat([X, Y], axis=-1)
            Beta_NN = DNN_base.PDE_DNN(XY, W2beta, B2beta, hidden_layers, activate_name=act_func)
            # S_NN = DNN_base.PDE_DNN(Beta_NN, W2S, B2S, hidden_layers, activate_name=act_func)
            # dSbeta = tf.gradients(S_NN, Beta_NN)
            #
            # tf.assign(beta, Beta_NN)
            Seff_NN = DNN_base.PDE_DNN(Beta_NN, W2S, B2S, hidden_layers, activate_name=act_func)

            dBeta2X = tf.gradients(Beta_NN, X)
            dBeta2Y = tf.gradients(Beta_NN, Y)
            dSbeta = tf.gradients(Seff_NN, Beta_NN)

            Loss2Inter = tf.reduce_mean(tf.square(dBeta2X)) + tf.reduce_mean(tf.square(dBeta2Y)) + tf.reduce_mean(tf.square(dSbeta))

            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            train_Loss2NN = my_optimizer.minimize(Loss2Inter, global_step=global_steps)

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate
        for i_epoch in range(100+1):
            test_x_bach = np.reshape(np.linspace(0, 1, num=bachsize), [-1, 1])
            test_y_bach = np.reshape(np.linspace(0, 1, num=bachsize), [-1, 1])
            tmp_lr = tmp_lr * (1 - lr_decay)
            _, dbeta2x, dbeta2y, ds2beta = sess.run([train_Loss2NN, dBeta2X, dBeta2Y, dSbeta],
                                                    feed_dict={X: test_x_bach, Y: test_y_bach, in_learning_rate: tmp_lr})


def Bernoulli(z=None):
    # zbottom = z - 0.5
    # zb_flag = np.max(0, zbottom)
    R = np.zeros_like(z)
    R = [z>0.5]
    return R


def Bernoulli_tf(bachsize=10, input_dim=1):
    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (0)):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            Z = tf.placeholder(tf.float32, name='z_recv', shape=[bachsize, input_dim])
            BRR = tf.zeros_like(Z)
            BRR = [Z>0.5]
            # BR = tf.gather(BRR, [0])
            BR = tf.cast(BRR, dtype=tf.float32)

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        R = sess.run(BR, feed_dict={Z: np.random.random((10, 1))})
        print(R[0])
        # print(R[0] + 0)


if __name__ == "__main__":
    # size2batch = 10
    # test_intergral(bachsize=size2batch)

    # X = np.random.random((10, 1))
    # B = Bernoulli(z=X)
    # R = B[0] + 0
    # print(R)

    Bernoulli_tf()