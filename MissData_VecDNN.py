"""
@author: LXA
 Date: 2021年 2 月 20 日
"""
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib
import platform
import shutil
import DNN_data
import time
import DNN_base
import DNN_tools
import DNN_Log_Print
import plotData
import saveData


def Bernoulli(Z=None):
    BRR = [Z > 0.5]
    BR = tf.cast(BRR, dtype=tf.float32)
    return BR


def pi_star(t=None):
    pi = tf.exp(1 - t) / (1 + tf.exp(1 - t))
    return pi


def my_normal(t=None):
    z = (1/np.sqrt(2*np.pi))*tf.exp(-0.5*t*t)
    return z


def fYX(z=None, func_name=None):
    if str.upper(func_name)=='GAUSSIAN':
        Z = my_normal(t=z)
    return Z


def solve_Integral_Equa(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    outFile2para_name = '%s_%s.txt' % ('para2', 'beta')
    logfile_name = '%s%s.txt' % ('log2', R['activate_func'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')       # 在这个路径下创建并打开一个可写的 log_train.txt文件
    para_outFile = open(os.path.join(log_out_path, outFile2para_name), 'w') # 在这个路径下创建并打开一个可写的 para_outFile.txt文件
    DNN_Log_Print.dictionary_out2file(R, log_fileout)

    # 积分方程问题需要的设置
    batchsize2bS = R['batch_size2bS']
    batchsize2b = R['batch_size2y_b']
    batchsize2integral = R['batch_size2integral']
    wb_regular = R['regular_weight_biases']
    if R['train_model'] == 'Alter_train':
        init_lr2b = R['init_learning_rate2b']
        init_lr2S = R['init_learning_rate2S']
        lr_decay2b = R['learning_rate_decay2b']
        lr_decay2S = R['learning_rate_decay2S']
    else:
        lr_decay = R['learning_rate_decay']
        init_lr = R['init_learning_rate']
    hidden_layers = R['hidden_layers']
    act_func = R['activate_func']

    # ------- set the problem ---------
    data_indim = R['input_dim']
    para_dim = R['estimate_para_dim']

    # bNN是否和参数 beta 显式相关，即参数 beta 作为bNN的一个输入。是：数据维数+参数维数；否：数据维数
    if R['bnn2beta'] == 'explicitly_related':
        bNN_inDim = data_indim + para_dim
    else:
        bNN_inDim = data_indim

    # 初始化权重和和偏置的模式。bNN是一个整体的网络表示，还是分为几个独立的网格表示 b 的每一个分量 bi. b=(b0,b1,b2,......)
    if R['sub_networks'] == 'subDNNs':
        flag_b0 = 'bnn0'
        flag_b1 = 'bnn1'
        bNN_outDim = 1
        if R['name2network'] == 'DNN_Cos_C_Sin_Base':
            W2b0, B2b0 = DNN_base.initialize_NN_random_normal2_CS(bNN_inDim, bNN_outDim, hidden_layers,
                                                                  flag_b0, unit_const_Init1layer=False)
            W2b1, B2b1 = DNN_base.initialize_NN_random_normal2_CS(bNN_inDim, bNN_outDim, hidden_layers,
                                                                  flag_b1, unit_const_Init1layer=False)
        else:
            W2b0, B2b0 = DNN_base.initialize_NN_random_normal2(bNN_inDim, bNN_outDim, hidden_layers, flag_b0,
                                                               unit_const_Init1layer=False)
            W2b1, B2b1 = DNN_base.initialize_NN_random_normal2(bNN_inDim, bNN_outDim, hidden_layers, flag_b0,
                                                               unit_const_Init1layer=False)
    else:
        bNN_outDim = para_dim
        flag_b = 'bnn'
        if R['name2network'] == 'DNN_Cos_C_Sin_Base':
            W2b, B2b = DNN_base.initialize_NN_random_normal2_CS(bNN_inDim, bNN_outDim, hidden_layers, flag_b,
                                                                unit_const_Init1layer=False)
        else:
            W2b, B2b = DNN_base.initialize_NN_random_normal2(bNN_inDim, bNN_outDim, hidden_layers, flag_b,
                                                             unit_const_Init1layer=False)

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            X = tf.placeholder(tf.float32, name='X_recv', shape=[batchsize2bS, data_indim])
            Y = tf.placeholder(tf.float32, name='Y_recv', shape=[batchsize2bS, data_indim])
            R2XY = tf.placeholder(tf.float32, name='R2XY_recv', shape=[batchsize2bS, data_indim])
            tfOne = tf.placeholder(tf.float32, shape=[batchsize2bS, data_indim], name='tfOne')

            y = tf.placeholder(tf.float32, name='y_recv', shape=[batchsize2b, data_indim])
            t_inte = tf.placeholder(tf.float32, name='t_integral', shape=[batchsize2integral, data_indim])
            beta = tf.Variable(tf.random.uniform([1, para_dim]), dtype='float32', name='beta')
            # beta = tf.Variable([[0.25, -0.5]], dtype='float32', name='beta')
            # beta = tf.constant([[0.25, -0.5]], dtype=tf.float32, name='beta')

            inline_lr = tf.placeholder_with_default(input=1e-2, shape=[], name='inline_learning_rate')
            inline_lr2b = tf.placeholder_with_default(input=1e-2, shape=[], name='inline_learning_rate2b')
            inline_lr2S = tf.placeholder_with_default(input=1e-2, shape=[], name='inline_learning_rate2S')

            if R['bnn2beta'] == 'explicitly_related':
                Repeat_beta2t = tf.tile(beta, [batchsize2integral, 1])
                tInte_beta = tf.concat([t_inte, Repeat_beta2t], axis=-1)

                Repeat_beta2y = tf.tile(beta, [batchsize2b, 1])
                y_beta = tf.concat([y, Repeat_beta2y], axis=-1)

                Repeat_beta2Y = tf.tile(beta, [batchsize2bS, 1])
                Y_beta = tf.concat([Y, Repeat_beta2Y], axis=-1)
            else:
                tInte_beta = t_inte
                y_beta = y
                Y_beta = Y

            if R['sub_networks'] == 'subDNNs':
                if R['name2network'] == str('DNN'):
                    b0_NN2t = DNN_base.PDE_DNN(tInte_beta, W2b0, B2b0, hidden_layers, activate_name=act_func)
                    b1_NN2t = DNN_base.PDE_DNN(tInte_beta, W2b1, B2b1, hidden_layers, activate_name=act_func)
                    b0_NN2y = DNN_base.PDE_DNN(y_beta, W2b0, B2b0, hidden_layers, activate_name=act_func)
                    b1_NN2y = DNN_base.PDE_DNN(y_beta, W2b1, B2b1, hidden_layers, activate_name=act_func)
                    b0_NN2Y = DNN_base.PDE_DNN(Y_beta, W2b0, B2b0, hidden_layers, activate_name=act_func)
                    b1_NN2Y = DNN_base.PDE_DNN(Y_beta, W2b1, B2b1, hidden_layers, activate_name=act_func)
                elif R['name2network'] == 'DNN_Cos_C_Sin_Base':
                    freq = R['freqs']
                    b0_NN2t = DNN_base.PDE_DNN_Cos_C_Sin_Base(tInte_beta, W2b0, B2b0, hidden_layers, freq, activate_name=act_func)
                    b1_NN2t = DNN_base.PDE_DNN_Cos_C_Sin_Base(tInte_beta, W2b1, B2b1, hidden_layers, freq, activate_name=act_func)
                    b0_NN2y = DNN_base.PDE_DNN_Cos_C_Sin_Base(y_beta, W2b0, B2b0, hidden_layers, freq, activate_name=act_func)
                    b1_NN2y = DNN_base.PDE_DNN_Cos_C_Sin_Base(y_beta, W2b1, B2b1, hidden_layers, freq, activate_name=act_func)
                    b0_NN2Y = DNN_base.PDE_DNN_Cos_C_Sin_Base(Y_beta, W2b0, B2b0, hidden_layers, freq, activate_name=act_func)
                    b1_NN2Y = DNN_base.PDE_DNN_Cos_C_Sin_Base(Y_beta, W2b1, B2b1, hidden_layers, freq, activate_name=act_func)
                b_NN2t = tf.concat([b0_NN2t, b1_NN2t], axis=-1)
                b_NN2y = tf.concat([b0_NN2y, b1_NN2y], axis=-1)
                b_NN2Y = tf.concat([b0_NN2Y, b1_NN2Y], axis=-1)
            else:
                if R['name2network'] == str('DNN'):
                    b_NN2t = DNN_base.PDE_DNN(tInte_beta, W2b, B2b, hidden_layers, activate_name=act_func)
                    b_NN2y = DNN_base.PDE_DNN(y_beta, W2b, B2b, hidden_layers, activate_name=act_func)
                    b_NN2Y = DNN_base.PDE_DNN(Y_beta, W2b, B2b, hidden_layers, activate_name=act_func)
                elif R['name2network'] == 'DNN_Cos_C_Sin_Base':
                    freq = R['freqs']
                    b_NN2t = DNN_base.PDE_DNN_Cos_C_Sin_Base(tInte_beta, W2b, B2b, hidden_layers, freq, activate_name=act_func)
                    b_NN2y = DNN_base.PDE_DNN_Cos_C_Sin_Base(y_beta, W2b, B2b, hidden_layers, freq, activate_name=act_func)
                    b_NN2Y = DNN_base.PDE_DNN_Cos_C_Sin_Base(Y_beta, W2b, B2b, hidden_layers, freq, activate_name=act_func)

            # 如下代码说明中 K 代表y的数目， N 代表 X 和 Y 的数目，M 代表 t 的数目
            # 求 b 的代码，先处理等号左边，再处理等号右边
            OneX = tf.concat([tfOne, X], axis=-1)      # N 行 (1+dim) 列，N代表X数据数目
            betaT = tf.transpose(beta, perm=[1, 0])    # (1+dim) 行 1 列
            oneX_betaT= tf.matmul(OneX, betaT)         # N 行 1 列。每个 Xi 都会对应 beta。 beta 是 (1+dim) 行 1 列的
            tInte_beraXT = tf.transpose(t_inte, perm=[1, 0]) - oneX_betaT  # N 行 M 列，t_inte 是 M 行 dim 列的

            # b方程左边系数分母，关于t求和平均代替积分
            fYX_t = fYX(z=tInte_beraXT, func_name='gaussian')         # fY|X在t处的取值  # N 行 M 列
            phi_1 = tf.transpose(1 - pi_star(t=t_inte), perm=[1, 0])  # M 行 dim 列转置为 dim 行 M 列
            fYXt_1phi = tf.multiply(fYX_t, phi_1)                     # N 行 M 列
            fYXt_1phi_intergal = tf.reduce_mean(fYXt_1phi, axis=-1)
            fYXt_1phi_intergal = tf.reshape(fYXt_1phi_intergal, shape=[-1, 1])  # N 行 dim 列

            # b方程左边系数分子，关于t求和平均代替积分
            dbeta2fYX_t =fYX_t * tInte_beraXT                                             # N 行 M 列
            expand_dbeta2fYX_t = tf.expand_dims(dbeta2fYX_t, axis=1)                      # N 页 1 行 M 列
            tilde_Edbeta2fYX_t = tf.tile(expand_dbeta2fYX_t, [1, data_indim+1, 1])        # N 页 (1+dim) 行 M 列
            dfYX_beta_t = tf.multiply(tilde_Edbeta2fYX_t, tf.expand_dims(OneX, axis=-1))  # N 页 (1+dim) 行 M 列
            phi_t = tf.transpose(pi_star(t=t_inte), perm=[1, 0])           # M 行 1 列转置为 1 行 M 列
            dfYX_beta_phi_t = tf.multiply(dfYX_beta_t, phi_t)              # N 页 (1+dim) 行 M 列
            dfYX_beta_integral = tf.reduce_mean(dfYX_beta_phi_t, axis=-1)  # N 页 (1+dim) 行

            # b方程左边的系数
            ceof2left = dfYX_beta_integral / fYXt_1phi_intergal            # N 行 (1+dim) 列
            # b方程左边积分系数的匹配项
            yaux_beraXT = tf.transpose(y, perm=[1, 0]) - oneX_betaT        # N 行 K 列， y 是 K 行 data_dim 列
            fYX_y = fYX(z=yaux_beraXT, func_name='gaussian')               # fY|X在y处的取值, N 行 K 列
            expand_fYX_y = tf.expand_dims(fYX_y, axis=1)                   # N 页 1 行 K 列
            bleft_fYX_y = tf.multiply(tf.expand_dims(ceof2left, axis=-1), expand_fYX_y)    # N 页 (1+dim) 行 K 列

            # b方程左边的第一个关于beta的导数项， 即dfYX2beta(y,Xi,beta)
            dbeta2fYX_y = fYX_y * yaux_beraXT                                              # N 行 K 列
            expand_dbeta2fYX_y = tf.expand_dims(dbeta2fYX_y, axis=1)                       # N 页 1 行 K 列
            dfYX_beta_y = tf.multiply(expand_dbeta2fYX_y, tf.expand_dims(OneX, axis=-1))   # N 页 (1+dim) 行 K 列

            # b方程左边的组合结果
            sum2bleft = bleft_fYX_y + dfYX_beta_y       # N 页 (1+dim) 行 K 列
            bleft = tf.reduce_mean(sum2bleft, axis=0)   # 1+dim 行 K 列

            # b 方程的右边。右边的第一项，如下两行计算
            trans_b_NN2y = tf.transpose(b_NN2y, perm=[1, 0])    # K 行 2 列转置为 2 行 K 列,然后扩为 1 X 2 X K 的
            by_fYX_y = tf.multiply(tf.expand_dims(trans_b_NN2y, axis=0), tf.expand_dims(fYX_y, axis=1))  # N 页 1+dim 行 K 列

            # b方程右边的系数的分子
            expand_fYX_t = tf.tile(tf.expand_dims(fYX_t, axis=1), [1, 1+data_indim, 1])  # N 页 (1+data_indim) 行 M 列
            bt_fYXt = tf.multiply(tf.transpose(b_NN2t, perm=[1, 0]), expand_fYX_t)       # N 页 (1+data_indim) 行 M 列
            bt_fYXt_phi = tf.multiply(bt_fYXt, phi_t)                                    # N 页 (1+data_indim) 行 M 列
            bt_fYXt_phi_integral = tf.reduce_mean(bt_fYXt_phi, axis=-1)                  # N 行 (1+data_indim) 列

            # b方程右边的系数。分母为共用b方程左边的分母
            ceof2fight = bt_fYXt_phi_integral / fYXt_1phi_intergal                       # N 行 (1+data_indim) 列

            # b方程系数与匹配项的线性积
            expand_fYX_y_right = tf.expand_dims(fYX_y, axis=1)                           # N 页 1 行 K 列
            bright_fYX_y = tf.multiply(tf.expand_dims(ceof2fight, axis=-1), expand_fYX_y_right)  # N 页 (1+data_indim) 行 K 列

            # b方程右边的组合结果
            sum2right = by_fYX_y + bright_fYX_y           # N 页 (1+dim) 行 K 列
            bright = tf.reduce_mean(sum2right, axis=0)    # 1+dim 行 K 列

            # b方程左边和右边要相等，使用差的平方来估计
            loss2b = tf.reduce_mean(tf.reduce_mean(tf.square(bleft - bright), axis=-1))

            # 求 Seff 的代码
            Y_beraXT = Y - oneX_betaT                      # N 行 dim 列， Y 和 oneX_betaT 都是 N 行 dim 列的

            fYX_Y = fYX(z=Y_beraXT, func_name='gaussian')  # fY|X在t处的取值  # N 行 dim 列
            ceof2R = (R2XY/fYX_Y)                          # N 行 dim 列， R2XY 是 N 行 dim 列的
            dbeta2fYX_Y = tf.multiply(fYX_Y, Y_beraXT)     # N 行 dim 列
            dfYX_beta_Y = tf.multiply(dbeta2fYX_Y, OneX)   # fY|X(t)对beta的导数，N行(1+dim)列，OneX 是 N 行 (1+dim) 列

            Sbeta_star1 = tf.multiply(ceof2R, dfYX_beta_Y)                                      # N 行 (1+dim)

            Seff_temp = tf.reshape(tf.reduce_mean(fYXt_1phi_intergal, axis=-1), shape=[-1, 1])  # N 行 dim 列
            ceof21R = (1-R2XY)/Seff_temp                                                        # N 行 dim 列
            Sbeta_star2 = tf.multiply(ceof21R, dfYX_beta_integral)                              # N 行 (1+dim)

            Sbeta_star = Sbeta_star1 - Sbeta_star2                                   # N 行 (1+dim)

            Seff2 = tf.multiply(R2XY, b_NN2Y)                                        # N 行 (1+dim)

            Seff3 = tf.multiply(1-R2XY, bt_fYXt_phi_integral / fYXt_1phi_intergal)   # N 行 (1+dim)

            squareSeff = tf.square(Sbeta_star - Seff2 + Seff3)                       # N 行 (1+dim)

            loss2Seff = tf.reduce_mean(tf.reduce_mean(squareSeff, axis=0))

            if R['sub_networks'] == 'subDNNs':
                if R['regular_weight_model'] == 'L1':
                    regular_WB2b0 = DNN_base.regular_weights_biases_L1(W2b0, B2b0)
                    regular_WB2b1 = DNN_base.regular_weights_biases_L1(W2b1, B2b1)
                elif R['regular_weight_model'] == 'L2':
                    regular_WB2b0 = DNN_base.regular_weights_biases_L2(W2b0, B2b0)
                    regular_WB2b1 = DNN_base.regular_weights_biases_L2(W2b1, B2b1)
                else:
                    regular_WB2b0 = tf.constant(0.0)
                    regular_WB2b1 = tf.constant(0.0)
                regular_WB2b = regular_WB2b0 + regular_WB2b1
            else:
                if R['regular_weight_model'] == 'L1':
                    regular_WB2b = DNN_base.regular_weights_biases_L1(W2b, B2b)
                elif R['regular_weight_model'] == 'L2':
                    regular_WB2b = DNN_base.regular_weights_biases_L2(W2b, B2b)
                else:
                    regular_WB2b = tf.constant(0.0)

            penalty_WB = wb_regular * regular_WB2b

            if R['train_model'] == 'Alter_train':
                lossB = loss2b + penalty_WB
                lossSeff = loss2Seff + penalty_WB

                my_optimizer2b = tf.train.AdamOptimizer(inline_lr2b)
                my_optimizer2Seff = tf.train.AdamOptimizer(inline_lr2S)
                train_loss2b = my_optimizer2b.minimize(lossB, global_step=global_steps)
                train_loss2Seff = my_optimizer2Seff.minimize(lossSeff, global_step=global_steps)
            else:
                loss = 5*loss2b + 10*loss2Seff + penalty_WB  # 要优化的loss function

                my_optimizer = tf.train.AdamOptimizer(inline_lr)
                if R['train_group'] == 0:
                    train_my_loss = my_optimizer.minimize(loss, global_step=global_steps)
                if R['train_group'] == 1:
                    train_op1 = my_optimizer.minimize(loss2b, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(loss2Seff, global_step=global_steps)
                    train_op3 = my_optimizer.minimize(loss, global_step=global_steps)
                    train_my_loss = tf.group(train_op1, train_op2, train_op3)
                elif R['train_group'] == 2:
                    train_op1 = my_optimizer.minimize(loss2b, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(loss2Seff, global_step=global_steps)
                    train_my_loss = tf.group(train_op1, train_op2)

    t0 = time.time()
    loss_b_all, loss_seff_all, loss_all = [], [], []  # 空列表, 使用 append() 添加元素

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行

    X_batch = DNN_data.randnormal_mu_sigma(size=batchsize2bS, mu=0.5, sigma=0.5)
    Y_init = 0.25 - 0.5 * X_batch + np.reshape(np.random.randn(batchsize2bS, 1), (-1, 1))
    # y_batch = DNN_data.randnormal_mu_sigma(size=batchsize2aux, mu=0.0, sigma=1.5)
    # x_aux_batch = DNN_data.randnormal_mu_sigma(size=batchsize2aux, mu=0.5, sigma=0.5)
    # y_batch = 0.25 - 0.5 * x_aux_batch + np.reshape(np.random.randn(batchsize2aux, 1), (-1, 1))
    relate2XY = np.reshape(np.random.randint(0, 2, batchsize2bS), (-1, 1))
    Y_batch = np.multiply(Y_init, relate2XY)
    one2train = np.ones((batchsize2bS, 1))

    y_batch = DNN_data.rand_it(batch_size=batchsize2b, variable_dim=data_indim, region_a=-2, region_b=2)
    t_batch = DNN_data.rand_it(batch_size=batchsize2integral, variable_dim=data_indim, region_a=-2, region_b=2)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if R['train_model'] == 'Alter_train':
            tmp_lr2b = init_lr2b
            tmp_lr2S = init_lr2S
            for i_epoch in range(R['max_epoch'] + 1):
                if i_epoch % 10 == 0 and i_epoch != 0:  # 10, 20, 30, 40,.....
                    tmp_lr2S = tmp_lr2S * (1 - lr_decay2S)
                    _S, loss2seff_tmp, loss2b_tmp, p_WB, beta_temp = sess.run(
                        [train_loss2Seff, loss2Seff, loss2b, penalty_WB, beta],
                        feed_dict={X: X_batch, Y: Y_batch, R2XY: relate2XY, y: y_batch, tfOne: one2train,
                                   t_inte: t_batch, inline_lr2S: tmp_lr2S})
                else:  # 0,1,2,3,4,5,6,7,8,9, 11,12,13,....,19, 21,22,....
                    tmp_lr2b = tmp_lr2b * (1 - lr_decay2b)
                    _b, loss2b_tmp, loss2seff_tmp, p_WB, beta_temp = sess.run(
                        [train_loss2b, loss2b, loss2Seff, penalty_WB, beta],
                        feed_dict={X: X_batch, Y: Y_batch, R2XY: relate2XY, y: y_batch, tfOne: one2train,
                                   t_inte: t_batch, inline_lr2b: tmp_lr2b})
                loss_seff_all.append(loss2seff_tmp)
                loss_b_all.append(loss2b_tmp)
                if (i_epoch % 10) == 0:
                    DNN_tools.log_string('*************** epoch: %d*10 ****************' % int(i_epoch / 10),
                                         log_fileout)
                    DNN_tools.log_string('lossb for training: %.10f\n' % loss2b_tmp, log_fileout)
                    DNN_tools.log_string('lossS for training: %.10f\n' % loss2seff_tmp, log_fileout)
                    DNN_tools.log_string('Penalty2Weights_Bias for training: %.10f\n' % p_WB, log_fileout)
                if (i_epoch % 100) == 0:
                    print('**************** epoch: %d*100 *******************' % int(i_epoch / 100))
                    print('beta:[%f %f]' % (beta_temp[0, 0], beta_temp[0, 1]))
                    print('\n')
                    DNN_tools.log_string('*************** epoch: %d*100 *****************' % int(i_epoch / 100),
                                         para_outFile)
                    DNN_tools.log_string('beta:[%f, %f]' % (beta_temp[0, 0], beta_temp[0, 1]), para_outFile)
                    DNN_tools.log_string('\n', para_outFile)

            saveData.save_2trainLosses2mat(loss_b_all, loss_seff_all, data1Name='lossb', data2Name='lossSeff',
                                           actName=act_func, outPath=R['FolderName'])
            plotData.plotTrain_loss_1act_func(loss_b_all, lossType='loss_b', seedNo=R['seed'],
                                              outPath=R['FolderName'], yaxis_scale=True)
            plotData.plotTrain_loss_1act_func(loss_seff_all, lossType='loss_s', seedNo=R['seed'],
                                              outPath=R['FolderName'], yaxis_scale=True)
        else:
            tmp_lr = init_lr
            for i_epoch in range(R['max_epoch'] + 1):
                tmp_lr = tmp_lr * (1 - lr_decay)
                _, loss2b_tmp, loss2seff_tmp, loss_tmp, p_WB, beta_temp = sess.run(
                    [train_my_loss, loss2b, loss2Seff, loss, penalty_WB, beta],
                    feed_dict={X: X_batch, Y: Y_batch, R2XY: relate2XY, y: y_batch, tfOne: one2train, t_inte: t_batch,
                               inline_lr: tmp_lr})

                loss_b_all.append(loss2b_tmp)
                loss_seff_all.append(loss2seff_tmp)
                loss_all.append(loss_tmp)
                if (i_epoch % 10) == 0:
                    DNN_tools.log_string('*************** epoch: %d*10 ****************' % int(i_epoch / 10), log_fileout)
                    DNN_tools.log_string('lossb for training: %.10f\n' % loss2b_tmp, log_fileout)
                    DNN_tools.log_string('lossS for training: %.10f\n' % loss2seff_tmp, log_fileout)
                    DNN_tools.log_string('loss for training: %.10f\n' % loss_tmp, log_fileout)
                    DNN_tools.log_string('Penalty2Weights_Bias for training: %.10f\n' % p_WB, log_fileout)
                if (i_epoch % 100) == 0:
                    print('**************** epoch: %d*100 *******************'% int(i_epoch/100))
                    print('beta:[%f %f]' % (beta_temp[0, 0], beta_temp[0, 1]))
                    print('\n')
                    DNN_tools.log_string('*************** epoch: %d*100 *****************' % int(i_epoch/100), para_outFile)
                    DNN_tools.log_string('beta:[%f, %f]' % (beta_temp[0, 0], beta_temp[0, 1]), para_outFile)
                    DNN_tools.log_string('\n', para_outFile)

            saveData.save_3trainLosses2mat(loss_b_all, loss_seff_all, loss_all, data1Name='lossb', data2Name='lossSeff',
                                           data3Name='lossAll', actName=act_func, outPath=R['FolderName'])
            plotData.plotTrain_loss_1act_func(loss_b_all, lossType='loss_b', seedNo=R['seed'], outPath=R['FolderName'],
                                              yaxis_scale=True)
            plotData.plotTrain_loss_1act_func(loss_seff_all, lossType='loss_s', seedNo=R['seed'], outPath=R['FolderName'],
                                              yaxis_scale=True)
            plotData.plotTrain_loss_1act_func(loss_all, lossType='loss_all', seedNo=R['seed'], outPath=R['FolderName'],
                                              yaxis_scale=True)


if __name__ == "__main__":
    R={}
    # -------------------------------------- CPU or GPU 选择 -----------------------------------------------
    R['gpuNo'] = 0
    # 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # -1代表使用 CPU 模式
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置当前使用的GPU设备仅为第 0 块GPU, 设备名称为'/gpu:0'
    if platform.system() == 'Windows':
        os.environ["CDUA_VISIBLE_DEVICES"] = "%s" % (R['gpuNo'])
    else:
        print('-------------------------------------- linux -----------------------------------------------')
        # Linux终端没有GUI, 需要添加如下代码，而且必须添加在 import matplotlib.pyplot 之前，否则无效。
        matplotlib.use('Agg')

        if tf.test.is_gpu_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 设置当前使用的GPU设备仅为第 0,1,2,3 块GPU, 设备名称为'/gpu:0'
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # ------------------------------------------- 文件保存路径设置 ----------------------------------------
    store_file = 'missdata'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    R['seed'] = np.random.randint(1e5)
    seed_str = str(R['seed'])                     # int 型转为字符串型
    FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
    R['FolderName'] = FolderName
    if not os.path.exists(FolderName):
        print('--------------------- FolderName with path-----------------:', FolderName)
        os.mkdir(FolderName)

    # ----------------------------------------  复制并保存当前文件 -----------------------------------------
    if platform.system() == 'Windows':
        tf.compat.v1.reset_default_graph()
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # ---------------------------- Setup of laplace equation ------------------------------
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    R['activate_stop'] = int(step_stop_flag)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    R['max_epoch'] = 200000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    R['PDE_type'] = 'Integral_Eq'
    R['eqs_name'] = 'missdata'

    R['input_dim'] = 1                         # 输入数据维数，即问题的维数(几元问题)
    R['output_dim'] = 1                        # 输出维数
    R['estimate_para_dim'] = 2                 # 待估计参数的维数

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of DNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 训练集的设置
    R['batch_size2bS'] = 2000                # 训练数据的批大小
    # R['batch_size2bS'] = 5000                  # 训练数据的批大小
    R['batch_size2integral'] = 200             # 求积分时训练数据的批大小
    R['batch_size2y_b'] = 200                  # 求 b 时关于 y 变量训练数据的批大小

    R['optimizer_name'] = 'Adam'                # 优化器

    # R['train_model'] = 'Alter_train'          # 交替训练模式
    R['train_model'] = 'Combine_train'          # 组合训练模式
    if R['train_model'] == 'Alter_train':
        R['train_group'] = 0
        R['init_learning_rate2b'] = 1e-2  # 学习率
        R['learning_rate_decay2b'] = 5e-3  # 学习率 decay
        R['init_learning_rate2S'] = 1e-3  # 学习率
        R['learning_rate_decay2S'] = 1e-3  # 学习率 decay
    else:
        # R['train_group'] = 0
        R['train_group'] = 1
        # R['train_group'] = 2
        R['init_learning_rate'] = 1e-2  # 学习率
        R['learning_rate_decay'] = 5e-3  # 学习率 decay

    # R['regular_weight_model'] = 'L0'
    # R['regular_weight_model'] = 'L1'
    R['regular_weight_model'] = 'L2'
    # R['regular_weight_biases'] = 0.000                    # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.1                      # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.01                     # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.001                    # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.0005                   # Regularization parameter for weights
    R['regular_weight_biases'] = 0.0001  # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.0025                 # Regularization parameter for weights

    # &&&&&&&&&&&&&&&&&&& 使用的网络模型 &&&&&&&&&&&&&&&&&&&&&&&&&&&
    R['name2network'] = 'DNN'
    # R['name2network'] = 'DNN_BN'
    # R['name2network'] = 'DNN_scale'
    # R['name2network'] = 'DNN_adapt_scale'
    # R['name2network'] = 'DNN_FourierBase'
    # R['name2network'] = 'DNN_Cos_C_Sin_Base'

    if R['name2network'] != 'DNN':
        # 网络的频率范围设置
        # R['freqs'] = np.arange(1, 101)
        R['freqs'] = np.concatenate(([1, 1, 1, 1], np.arange(1, 20)), axis=0)
        # R['freqs'] = np.concatenate(([1, 1, 1, 1, 1, 1, 1, 1, 1], np.arange(1, 20)), axis=0)

    # &&&&&&&&&&&&&&&&&&&&&& 隐藏层的层数和每层神经元数目 &&&&&&&&&&&&&&&&&&&&&&&&&&&&
    if R['name2network'] == 'DNN_Cos_C_Sin_Base':
        R['hidden_layers'] = (30, 20, 10)
        # R['hidden_layers'] = (30, 20, 20, 10)
        # R['hidden_layers'] = (30, 20, 10, 10, 5)
        # R['hidden_layers'] = (100, 50, 30, 30, 20)
        # R['hidden_layers'] = (100, 100, 80, 80, 60)  # 1*200+200*100+100*80+80*80+80*60+60*1= 39460 个参数
        # R['hidden_layers'] = (125, 100, 80, 80, 60)  # 1*250+250*100+100*80+80*80+80*60+60*1= 44510 个参数
        # R['hidden_layers'] = (100, 120, 80, 80, 80)  # 1*200+200*120+120*80+80*80+80*80+80*1= 46680 个参数
        # R['hidden_layers'] = (125, 120, 80, 80, 80)  # 1*250+250*120+120*80+80*80+80*80+80*1= 52730 个参数
        # R['hidden_layers'] = (100, 150, 100, 100, 80)  # 1*200+200*150+150*100+100*100+100*80+80*1= 63280 个参数
        # R['hidden_layers'] = (125, 150, 100, 100, 80)  # 1*250+250*150+150*100+100*100+100*80+80*1= 70830 个参数
    else:
        R['hidden_layers'] = (30, 20, 10)
        # R['hidden_layers'] = (30, 20, 20, 10)
        # R['hidden_layers'] = (30, 20, 10, 10, 5)
        # R['hidden_layers'] = (100, 50, 30, 30, 20)
        # R['hidden_layers'] = (400, 300, 300, 200, 100, 100, 50)
        # R['hidden_layers'] = (500, 400, 300, 300, 200, 100)
        # R['hidden_layers'] = (500, 400, 300, 200, 200, 100)

    # &&&&&&&&&&&&&&&&&&& 激活函数的选择 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['activate_func'] = 'relu'
    R['activate_func'] = 'tanh'
    # R['activate_func'] = 'sintanh'
    # R['activate_func']' = leaky_relu'
    # R['activate_func'] = 'srelu'
    # R['activate_func'] = 's2relu'
    # R['activate_func'] = 'elu'
    # R['activate_func'] = 'selu'
    # R['activate_func'] = 'phi'

    R['bnn2beta'] = 'explicitly_related'  # bNN 和参数 beta 显式相关，即参数 beta 作为bNN的一个输入
    # R['bnn2beta'] = 'implicitly_related'    # bNN 和参数 beta 隐式相关，即参数 beta 不作为bNN的一个输入
    # bNN是一个整体的网络表示，还是分为几个独立的网格表示 b 的每一个分量 bi. b=(b0,b1,b2,......)
    R['sub_networks'] = 'subDNNs'
    # R['sub_networks'] = 'oneDNN'

    solve_Integral_Equa(R)

