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
import plotData
import saveData


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout):
    DNN_tools.log_string('PDE type for problem: %s\n' % (R_dic['PDE_type']), log_fileout)
    DNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['eqs_name']), log_fileout)

    if R_dic['activate_stop'] != 0:
        DNN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        DNN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']), log_fileout)

    DNN_tools.log_string('Network model of solving problem: %s\n' % str(R_dic['model']), log_fileout)
    DNN_tools.log_string('Activate function for network: %s\n' % str(R_dic['activate_func']), log_fileout)
    if R_dic['model'] != 'DNN':
        DNN_tools.log_string('The frequency to neural network: %s\n' % (R_dic['freqs']), log_fileout)

    if (R_dic['optimizer_name']).title() == 'Adam':
        DNN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        DNN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']), log_fileout)

    DNN_tools.log_string('Init learning rate: %s\n' % str(R_dic['learning_rate']), log_fileout)

    DNN_tools.log_string('Decay to learning rate: %s\n' % str(R_dic['learning_rate_decay']), log_fileout)

    DNN_tools.log_string('hidden layer:%s\n' % str(R_dic['hidden_layers']), log_fileout)

    DNN_tools.log_string('Batch-size 2 integral: %s\n' % str(R_dic['batch_size2integral']), log_fileout)
    DNN_tools.log_string('Batch-size 2 auxiliary: %s\n' % str(R_dic['batch_size2auxiliary']), log_fileout)


def Bernoulli(Z=None):
    BRR = [Z > 0.5]
    BR = tf.cast(BRR, dtype=tf.float32)
    return BR


def phi_star(t=None):
    phi = tf.exp(1 - t) / (1 + tf.exp(1 - t))
    return phi


def my_normal(t=None):
    z = (1/np.sqrt(2*np.pi))*tf.exp(-0.5*t*t)
    return z


def solve_Integral_Equa(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s_%s.txt' % ('log2', R['activate_func'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    dictionary_out2file(R, log_fileout)

    # 积分方程问题需要的设置
    batchsize = R['batch_size2integral']
    batchsize2aux = R['batch_size2auxiliary']
    wb_regular = R['regular_weight_biases']
    lr_decay = R['learning_rate_decay']
    learning_rate = R['learning_rate']
    hidden_layers = R['hidden_layers']
    act_func = R['activate_func']

    # ------- set the problem ---------
    data_indim = R['input_dim']
    data_outdim = R['output_dim']
    para_dim = R['estimate_para_dim']

    # 初始化权重和和偏置的模式
    flag2bnn = 'bnn'
    input_dim = 1
    if R['model'] == 'PDE_DNN_Cos_C_Sin_Base':
        W2b, B2b = DNN_base.initialize_NN_random_normal2_CS(input_dim, para_dim, hidden_layers, flag2bnn)
    else:
        W2b, B2b = DNN_base.initialize_NN_random_normal2(input_dim, para_dim, hidden_layers, flag2bnn)

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            X = tf.placeholder(tf.float32, name='X_recv', shape=[batchsize, data_indim])
            Y = tf.placeholder(tf.float32, name='Y_recv', shape=[batchsize, data_indim])
            R2XY = tf.placeholder(tf.float32, name='R2XY_recv', shape=[batchsize, data_indim])
            y_aux = tf.placeholder(tf.float32, name='y_auxiliary', shape=[batchsize2aux, 1])
            beta = tf.Variable(0.1 * tf.random.uniform([1, para_dim]), dtype='float32', name='beta')
            tfOne = tf.placeholder(tf.float32, shape=[1, 1], name='tfOne')
            inline_lr = tf.placeholder_with_default(input=1e-5, shape=[], name='inline_learning_rate')

            # 供选择的网络模式
            if R['model'] == str('DNN'):
                b_NN2y = DNN_base.PDE_DNN(y_aux, W2b, B2b, hidden_layers, activate_name=act_func)
                # b_NN2Y = DNN_base.PDE_DNN(Y, W2b, B2b, hidden_layers, activate_name=act_func)
            elif R['model'] == 'DNN_scale':
                freq = R['freqs']
                b_NN2y = DNN_base.PDE_DNN_scale(y_aux, W2b, B2b, hidden_layers, freq, activate_name=act_func)
                # b_NN2Y = DNN_base.PDE_DNN_scale(Y, W2b, B2b, hidden_layers, freq, activate_name=act_func)
            elif R['model'] == 'DNN_adapt_scale':
                freq = R['freqs']
                b_NN2y = DNN_base.PDE_DNN_adapt_scale(y_aux, W2b, B2b, hidden_layers, freq, activate_name=act_func)
                # b_NN2Y = DNN_base.PDE_DNN_adapt_scale(Y, W2b, B2b, hidden_layers, freq, activate_name=act_func)
            elif R['model'] == 'DNN_FourierBase':
                freq = R['freqs']
                b_NN2y = DNN_base.PDE_DNN_FourierBase(y_aux, W2b, B2b, hidden_layers, freq, activate_name=act_func)
                # b_NN2Y = DNN_base.PDE_DNN_FourierBase(Y, W2b, B2b, hidden_layers, freq, activate_name=act_func)
            elif R['model'] == 'DNN_Cos_C_Sin_Base':
                freq = R['freqs']
                b_NN2y = DNN_base.PDE_DNN_Cos_C_Sin_Base(y_aux, W2b, B2b, hidden_layers, freq, activate_name=act_func)
                # b_NN2Y = DNN_base.PDE_DNN_Cos_C_Sin_Base(Y, W2b, B2b, hidden_layers, freq, activate_name=act_func)
            elif R['model'] == 'DNN_WaveletBase':
                freq = R['freqs']
                b_NN2y = DNN_base.PDE_DNN_WaveletBase(y_aux, W2b, B2b, hidden_layers, freq, activate_name=act_func)
                # b_NN2Y = DNN_base.PDE_DNN_WaveletBase(Y, W2b, B2b, hidden_layers, freq, activate_name=act_func)

            # 下面怎么把循环改为向量操作呢？
            sum2bleft = tf.zeros(shape=[1, 1], dtype=tf.float32, name='01')
            sum2bright = tf.zeros(shape=[1, 1], dtype=tf.float32, name='02')
            for i in range(batchsize):
                Xtemp = tf.reshape(X[i], shape=[1, 1])
                OneX = tf.concat([tfOne, Xtemp], axis=-1)   # 1 行 (1+dim) 列
                XiTrans = tf.transpose(OneX, [1, 0])

                fYX_y = my_normal(t=y_aux-tf.matmul(beta, XiTrans))
                dfYX_beta = my_normal(t=y_aux-tf.matmul(beta, XiTrans))*(y_aux-tf.matmul(beta, XiTrans)) * OneX

                # beta 是 1 行 para_dim 列
                fyx_1minus_phi_integral = tf.reduce_mean(fYX_y * (1 - phi_star(t=y_aux)), axis=0)
                dfyx_phi_integral = tf.reduce_mean(dfYX_beta * phi_star(t=y_aux), axis=0)
                ceof_vec2left = dfyx_phi_integral/fyx_1minus_phi_integral
                sum2bleft = sum2bleft + dfYX_beta + ceof_vec2left*fYX_y

                b_fyx_phi_integral = tf.reduce_mean(b_NN2y*fYX_y*phi_star(t=y_aux), axis=0)
                ceof_vec2right = b_fyx_phi_integral / fyx_1minus_phi_integral
                sum2bright = sum2bright + b_NN2y * fYX_y + ceof_vec2right * fYX_y

            bleft = sum2bleft / batchsize
            bright = sum2bright / batchsize

            loss2b = tf.reduce_mean(tf.reduce_mean(tf.square(bleft - bright), axis=0))

            sum2Seff = tf.zeros(shape=[1, 1], dtype=tf.float32, name='03')
            for i in range(batchsize):
                Xtemp = tf.reshape(X[i], shape=[1, 1])
                OneX = tf.concat([tfOne, Xtemp], axis=-1)  # 1 行 (1+dim) 列
                XiTrans = tf.transpose(OneX, [1, 0])

                fYX2y = my_normal(t=y_aux - tf.matmul(beta, XiTrans))
                Yi = tf.reshape(Y[i], shape=[1, 1])
                fYX2Y = my_normal(t=Yi-tf.matmul(beta, XiTrans))

                dfYX_beta2Y = my_normal(t=Yi-tf.matmul(beta, XiTrans))*(Yi-tf.matmul(beta, XiTrans)) * OneX
                dfYX_beta2y = my_normal(t=y_aux - tf.matmul(beta, XiTrans)) * (y_aux - tf.matmul(beta, XiTrans)) * OneX

                fyx_1minus_phi_integral = tf.reduce_mean(fYX2y * (1 - phi_star(t=y_aux)), axis=0)
                dfyx_phi_integral = tf.reduce_mean(dfYX_beta2y * phi_star(t=y_aux), axis=0)
                fyx_b_phi_integral = tf.reduce_mean(fYX2y * b_NN2y * phi_star(t=y_aux), axis=0)

                R2XY_i = tf.reshape(R2XY[i], shape=[1, -1])
                Seff1 = (R2XY_i/fYX2Y) * dfYX_beta2Y - ((1-R2XY_i)/fyx_1minus_phi_integral) * dfyx_phi_integral
                if R['model'] == str('DNN'):
                    b_NN2yi = DNN_base.PDE_DNN(Yi, W2b, B2b, hidden_layers, activate_name=act_func)
                else:
                    assert(R['model'] == str('DNN'))
                Seff2 = R2XY_i * tf.reshape(b_NN2yi, shape=[1, -1])
                Seff3 = (1-R2XY_i) * (fyx_b_phi_integral/fyx_1minus_phi_integral)
                Seff = Seff1 - Seff2 + Seff3
                sum2Seff = sum2Seff + Seff

            loss2s_temp = sum2Seff/batchsize
            loss2Seff = tf.reduce_mean(tf.square(loss2s_temp))

            if R['regular_weight_model'] == 'L1':
                regular_WB2b = DNN_base.regular_weights_biases_L1(W2b, B2b)
            elif R['regular_weight_model'] == 'L2':
                regular_WB2b = DNN_base.regular_weights_biases_L2(W2b, B2b)
            else:
                regular_WB2b = tf.constant(0.0)

            penalty_WB = wb_regular * regular_WB2b
            loss = loss2b + loss2Seff + penalty_WB       # 要优化的loss function

            my_optimizer = tf.train.AdamOptimizer(inline_lr)
            if R['train_group'] == 1:
                train_op1 = my_optimizer.minimize(loss2b, global_step=global_steps)
                train_op2 = my_optimizer.minimize(loss2Seff, global_step=global_steps)
                train_op3 = my_optimizer.minimize(loss, global_step=global_steps)
                train_my_loss = tf.group(train_op1, train_op2, train_op3)
            else:
                train_my_loss = my_optimizer.minimize(loss, global_step=global_steps)

    t0 = time.time()
    loss_b_all, loss_seff_all, loss_all = [], [], []  # 空列表, 使用 append() 添加元素

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行

    x_batch = DNN_data.randnormal_mu_sigma(size=batchsize, mu=0.5, sigma=0.5)
    y_init = 0.25 - 0.5 * x_batch + np.reshape(np.random.randn(batchsize, 1), (-1, 1))
    y_aux_batch = np.reshape(np.random.uniform(0, 1, batchsize2aux), (-1, 1))
    # y_aux_batch = np.reshape(np.random.randn(batchsize2aux, 1), (-1, 1))
    # x_aux_batch = DNN_data.randnormal_mu_sigma(size=batchsize2aux, mu=0.5, sigma=0.5)
    # y_aux_batch = 0.25 - 0.5 * x_aux_batch + np.reshape(np.random.randn(batchsize2aux, 1), (-1, 1))
    relate2XY = np.reshape(np.random.randint(0, 2, batchsize), (-1, 1))
    y_batch = np.multiply(y_init, relate2XY)
    one2train = np.ones((1, 1))

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate
        for i_epoch in range(R['max_epoch'] + 1):
            # x_batch = DNN_data.randnormal_mu_sigma(size=batchsize, mu=0.5, sigma=0.5)
            # y_batch = 0.25 - 0.5*x_batch + np.random.randn(batchsize, 1)
            # y_aux_batch = np.reshape(np.random.uniform(0, 1, batchsize2aux), (-1, 1))
            # relate2XY = np.reshape(np.random.randint(0, 1, batchsize), (-1, 1))
            # one2train = np.ones((1, 1))
            tmp_lr = tmp_lr * (1 - lr_decay)
            _, loss2b_tmp, loss2seff_tmp, loss_tmp, p_WB, beta_temp = sess.run(
                [train_my_loss, loss2b, loss2Seff, loss, penalty_WB, beta],
                feed_dict={X: x_batch, Y: y_batch, R2XY: relate2XY, y_aux: y_aux_batch, tfOne: one2train,
                           inline_lr: tmp_lr})

            loss_b_all.append(loss2b_tmp)
            loss_seff_all.append(loss2seff_tmp)
            loss_all.append(loss_tmp)

            DNN_tools.log_string('loss for training: %.10f\n' % loss_tmp, log_fileout)
            if (i_epoch % 100) == 0:
                print('****************** %d **********************'% int(i_epoch/100))
                print(beta_temp)
                print('\n')

        saveData.save_trainLoss2mat_1actFunc(loss_b_all, loss_seff_all, loss_all, actName=act_func, outPath=R['FolderName'])
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
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    R['seed'] = np.random.randint(1e5)
    seed_str = str(R['seed'])                     # int 型转为字符串型
    FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
    R['FolderName'] = FolderName
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
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

    R['input_dim'] = 1                                    # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1                                   # 输出维数
    R['estimate_para_dim'] = 2

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of DNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 训练集的设置
    R['batch_size2integral'] = 200              # 内部训练数据的批大小
    # R['batch_size2integral'] = 100            # 内部训练数据的批大小
    # R['batch_size2integral'] = 5              # 内部训练数据的批大小
    # R['batch_size2auxiliary'] = 100
    R['batch_size2auxiliary'] = 50

    R['optimizer_name'] = 'Adam'                # 优化器
    R['learning_rate'] = 1e-3                   # 学习率
    R['learning_rate_decay'] = 1e-4             # 学习率 decay
    # R['train_group'] = 1
    R['train_group'] = 0

    R['regular_weight_model'] = 'L0'
    # R['regular_weight_model'] = 'L1'
    # R['regular_weight_model'] = 'L2'
    R['regular_weight_biases'] = 0.000                    # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.001                  # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.0025                 # Regularization parameter for weights

    # &&&&&&&&&&&&&&&&&&& 使用的网络模型 &&&&&&&&&&&&&&&&&&&&&&&&&&&
    R['model'] = 'DNN'
    # R['model'] = 'DNN_BN'
    # R['model'] = 'DNN_scale'
    # R['model'] = 'DNN_adapt_scale'
    # R['model'] = 'DNN_FourierBase'
    # R['model'] = 'DNN_Cos_C_Sin_Base'
    # R['model'] = 'DNN_WaveletBase'

    if R['model'] != 'DNN':
        # 网络的频率范围设置
        R['freqs'] = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)

    # &&&&&&&&&&&&&&&&&&&&&& 隐藏层的层数和每层神经元数目 &&&&&&&&&&&&&&&&&&&&&&&&&&&&
    if R['model'] == 'DNN_Cos_C_Sin_Base':
        # R['hidden_layers'] = (30, 20, 10, 10, 5)
        R['hidden_layers'] = (100, 50, 30, 30, 20)
        # R['hidden_layers'] = (100, 100, 80, 80, 60)  # 1*200+200*100+100*80+80*80+80*60+60*1= 39460 个参数
        # R['hidden_layers'] = (125, 100, 80, 80, 60)  # 1*250+250*100+100*80+80*80+80*60+60*1= 44510 个参数
        # R['hidden_layers'] = (100, 120, 80, 80, 80)  # 1*200+200*120+120*80+80*80+80*80+80*1= 46680 个参数
        # R['hidden_layers'] = (125, 120, 80, 80, 80)  # 1*250+250*120+120*80+80*80+80*80+80*1= 52730 个参数
        # R['hidden_layers'] = (100, 150, 100, 100, 80)  # 1*200+200*150+150*100+100*100+100*80+80*1= 63280 个参数
        # R['hidden_layers'] = (125, 150, 100, 100, 80)  # 1*250+250*150+150*100+100*100+100*80+80*1= 70830 个参数
    else:
        # R['hidden_layers'] = (30, 20, 10, 10, 5)
        R['hidden_layers'] = (100, 50, 30, 30, 20)
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

    solve_Integral_Equa(R)

