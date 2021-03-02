"""
@author: LXA
 Date: 2020 年 5 月 31 日
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
    if R['PDE_type'] == 'p_laplace':
        DNN_tools.log_string('The order of p-Laplace: %s\n' % (R_dic['order2laplace']), log_fileout)
        DNN_tools.log_string('The epsilon to p-laplace: %f\n' % (R_dic['epsilon']), log_fileout)

    if R_dic['activate_stop'] != 0:
        DNN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        DNN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']), log_fileout)

    if R_dic['variational_loss'] == 1:
        DNN_tools.log_string('Loss function: variational loss\n', log_fileout)
    else:
        DNN_tools.log_string('Loss function: original function loss\n', log_fileout)

    DNN_tools.log_string('Network model of solving problem: %s\n' % str(R_dic['model']), log_fileout)
    DNN_tools.log_string('The frequency to neural network: %s\n' % (R_dic['freqs']), log_fileout)
    DNN_tools.log_string('Activate function for network: %s\n' % str(R_dic['activate_func']), log_fileout)

    if (R_dic['optimizer_name']).title() == 'Adam':
        DNN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        DNN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']), log_fileout)

    DNN_tools.log_string('Init learning rate: %s\n' % str(R_dic['learning_rate']), log_fileout)

    DNN_tools.log_string('Decay to learning rate: %s\n' % str(R_dic['learning_rate_decay']), log_fileout)

    DNN_tools.log_string('hidden layer:%s\n' % str(R_dic['hidden_layers']), log_fileout)

    DNN_tools.log_string('Batch-size 2 integral: %s\n' % str(R_dic['batch_size2integral']), log_fileout)

    DNN_tools.log_string('Initial boundary penalty: %s\n' % str(R_dic['init_boundary_penalty']), log_fileout)
    if R['activate_penalty2bd_increase'] == 1:
        DNN_tools.log_string('The penalty of boundary will increase with training going on.\n', log_fileout)
    elif R['activate_penalty2bd_increase'] == 2:
        DNN_tools.log_string('The penalty of boundary will decrease with training going on.\n', log_fileout)
    else:
        DNN_tools.log_string('The penalty of boundary will keep unchanged with training going on.\n', log_fileout)


def Bernoulli(Z=None):
    BRR = [Z > 0.5]
    BR = tf.cast(BRR, dtype=tf.float32)
    return BR


def solve_Integral_Equa(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s_%s.txt' % ('log2', R['activate_func'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    dictionary_out2file(R, log_fileout)

    # 积分方程问题需要的设置
    batchsize = R['batch_size2integral']
    wb_regular = R['regular_weight_biases']                # Regularization parameter for weights and biases
    lr_decay = R['learning_rate_decay']
    learning_rate = R['learning_rate']
    hidden_layers = R['hidden_layers']
    act_func = R['activate_func']

    # ------- set the problem ---------
    input_dim = R['input_dim']
    out_dim = R['output_dim']

    # 初始化权重和和偏置的模式
    flag2beta = 'beta'
    flag2Seff = 'Seff'
    flag2b = 'b'
    if R['model'] == 'PDE_DNN_Cos_C_Sin_Base':
        W2beta, B2beta = DNN_base.initialize_NN_random_normal2_CS(input_dim+2, out_dim, hidden_layers, flag2beta)
        # W2Seff, B2Seff = DNN_base.initialize_NN_random_normal2_CS(input_dim, out_dim, hidden_layers, flag2Seff)
        W2b, B2b = DNN_base.initialize_NN_random_normal2_CS(input_dim+1, out_dim, hidden_layers, flag2b)
    else:
        W2beta, B2beta = DNN_base.initialize_NN_random_normal2(input_dim+2, out_dim, hidden_layers, flag2beta)
        # W2Seff, B2Seff = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2Seff)
        W2b, B2b = DNN_base.initialize_NN_random_normal2(input_dim+1, out_dim, hidden_layers, flag2b)

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            tfOne = tf.placeholder(tf.float32, name='tfOne', shape=[batchsize, input_dim])  # * 行 1 列
            X = tf.placeholder(tf.float32, name='X_recv', shape=[batchsize, input_dim])                # * 行 1 列
            Y = tf.placeholder(tf.float32, name='Y_recv', shape=[batchsize, input_dim])      # * 行 1 列
            R = tf.placeholder(tf.float32, name='R_recv', shape=[batchsize, input_dim])
            beta = tf.placeholder(tf.float32, name='R_recv', shape=[batchsize, para_dim])
            in_learning_rate = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')
            train_opt = tf.placeholder_with_default(input=True, shape=[], name='train_opt')

            OneX = tf.concat([tfOne, X], axis=-1)
            OneXY = tf.concat([X, Y], axis=-1)
            # 供选择的网络模式
            if R['model'] == 'DNN':
                Beta_NN = DNN_base.PDE_DNN(OneXY, W2beta, B2beta, hidden_layers, activate_name=act_func)
                # Seff_NN = DNN_base.PDE_DNN(Beta_NN, W2Seff, B2Seff, hidden_layers, activate_name=act_func)
                betaY = tf.concat([Beta_NN, Y], axis=-1)
                b_NN = DNN_base.PDE_DNN(betaY, W2b, B2b, hidden_layers, activate_name=act_func)
            elif R['model'] == 'DNN_scale':
                freq = R['freqs']
                Beta_NN = DNN_base.PDE_DNN_scale(OneXY, W2beta, B2beta, hidden_layers, freq, activate_name=act_func)
                # Seff_NN = DNN_base.PDE_DNN_scale(Beta_NN, W2Seff, B2Seff, hidden_layers, freq, activate_name=act_func)
                betaY = tf.concat([Beta_NN, Y], axis=-1)
                b_NN = DNN_base.PDE_DNN_scale(betaY, W2b, B2b, hidden_layers, freq, activate_name=act_func)
            elif R['model'] == 'DNN_adapt_scale':
                freq = R['freqs']
                Beta_NN = DNN_base.PDE_DNN_adapt_scale(OneXY, W2beta, B2beta, hidden_layers, freq, activate_name=act_func)
                betaY = tf.concat([Beta_NN, Y], axis=-1)
                b_NN = DNN_base.PDE_DNN_adapt_scale(betaY, W2b, B2b, hidden_layers, freq, activate_name=act_func)
            elif R['model'] == 'DNN_FourierBase':
                freq = R['freqs']
                Beta_NN = DNN_base.PDE_DNN_FourierBase(OneXY, W2beta, B2beta, hidden_layers, freq, activate_name=act_func)
                betaY = tf.concat([Beta_NN, Y], axis=-1)
                b_NN = DNN_base.PDE_DNN_FourierBase(betaY, W2b, B2b, hidden_layers, freq, activate_name=act_func)
            elif R['model'] == 'DNN_Cos_C_Sin_Base':
                freq = R['freqs']
                Beta_NN = DNN_base.PDE_DNN_Cos_C_Sin_Base(OneXY, W2beta, B2beta, hidden_layers, freq, activate_name=act_func)
                betaY = tf.concat([Beta_NN, Y], axis=-1)
                b_NN = DNN_base.PDE_DNN_Cos_C_Sin_Base(betaY, W2b, B2b, hidden_layers, freq, activate_name=act_func)
            elif R['model'] == 'DNN_WaveletBase':
                freq = R['freqs']
                Beta_NN = DNN_base.PDE_DNN_WaveletBase(OneXY, W2beta, B2beta, hidden_layers, freq, activate_name=act_func)
                betaY = tf.concat([Beta_NN, Y], axis=-1)
                b_NN = DNN_base.PDE_DNN_WaveletBase(betaY, W2b, B2b, hidden_layers, freq, activate_name=act_func)

            phi2Y = tf.exp(1-Y)/(1+tf.exp(1-Y))
            fYX = Beta_NN - 0.5*X + tf.random_normal([batchsize,input_dim], mean=0.0, stddev=1)
            fX = X

            dfYX_beta = tf.gradients(fYX, Beta_NN)
            # dfYX_beta =
            tempbl1 = dfYX_beta*phi2Y
            tempbl2 = fYX*(1-phi2Y)
            ceof2bl = tf.reduce_mean(tempbl1)/tf.reduce_mean(tempbl2)
            lossbl = (dfYX_beta + ceof2bl*fYX)*fX

            bfYX = b_NN * fYX
            tempbr1 = bfYX * phi2Y
            tempbr2 = fYX * (1 - phi2Y)
            ceof2br = tf.reduce_mean(tempbr1) / tf.reduce_mean(tempbr2)

            lossbr = (bfYX + ceof2br * fYX) * fX

            loss2b = tf.square(lossbl - lossbr)

            R = Bernoulli(Z=phi2Y)
            stemp1 = (R/fYX)*dfYX_beta
            stemp2 = (1-R)/(tf.reduce_mean(fYX*(1-phi2Y)))*tf.reduce_mean(dfYX_beta*phi2Y)
            stemp3 = R*b_NN
            sstemp1 = tf.reduce_mean(fYX*b_NN*phi2Y)
            sstemp2 = tf.reduce_mean(fYX * (1- phi2Y))
            stemp4 = (1-R)*(sstemp1/sstemp2)

            loss2Seff = tf.reduce_mean(tf.square(stemp1 - stemp2 - stemp3 + stemp4))

            if R['regular_weight_model'] == 'L1':
                regular_WB2Beta = DNN_base.regular_weights_biases_L1(W2beta, B2beta)    # 正则化权重和偏置 L1正则化
                regular_WB2b = DNN_base.regular_weights_biases_L1(W2b, B2b)
            elif R['regular_weight_model'] == 'L2':
                regular_WB2Beta = DNN_base.regular_weights_biases_L2(W2beta, B2beta)    # 正则化权重和偏置 L2正则化
                regular_WB2b = DNN_base.regular_weights_biases_L2(W2b, B2b)
            else:
                regular_WB2Beta = tf.constant(0.0)                                         # 无正则化权重参数
                regular_WB2b = tf.constant(0.0)

            penalty_WB = wb_regular * (regular_WB2Beta + regular_WB2b)
            loss = loss2b + loss2Seff + penalty_WB       # 要优化的loss function

            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            if R['train_group'] == 1:
                train_op1 = my_optimizer.minimize(loss2b, global_step=global_steps)
                train_op2 = my_optimizer.minimize(loss2Seff, global_step=global_steps)
                train_op3 = my_optimizer.minimize(loss, global_step=global_steps)
                train_my_loss = tf.group(train_op1, train_op2, train_op3)
            else:
                train_my_loss = my_optimizer.minimize(loss, global_step=global_steps)

    t0 = time.time()
    loss_it_all, loss_bd_all, loss_all, train_mse_all, train_rel_all = [], [], [], [], []  # 空列表, 使用 append() 添加元素
    test_mse_all, test_rel_all = [], []
    testing_epoch = []

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate

        for i_epoch in range(R['max_epoch'] + 1):
            x_batch = DNN_data.randnormal_mu_sigma(size=batchsize, mu=0.5, sigma=0.5)
            y_batch = DNN_data.randnormal_mu_sigma(size=batchsize, mu=0.5, sigma=0.5) + np.random.randn(size=[batchsize, 1])
            tmp_lr = tmp_lr * (1 - lr_decay)

            _, loss_it_tmp, loss_bd_tmp, loss_tmp, train_mse_tmp, train_res_tmp, p_WB = sess.run(
                [train_my_loss, loss2b, loss2Seff, loss, penalty_WB],
                feed_dict={X: x_batch, Y: y_batch, in_learning_rate: tmp_lr})

            loss_it_all.append(loss_it_tmp)
            loss_bd_all.append(loss_bd_tmp)
            loss_all.append(loss_tmp)
            train_mse_all.append(train_mse_tmp)
            train_rel_all.append(train_res_tmp)
            if i_epoch % 1000 == 0:
                run_times = time.time() - t0
                DNN_tools.print_and_log_train_one_epoch(
                    i_epoch, run_times, tmp_lr, temp_penalty_bd, p_WB, loss_it_tmp, loss_bd_tmp, loss_tmp, train_mse_tmp,
                    train_res_tmp, log_out=log_fileout)

                # ---------------------------   test network ----------------------------------------------
                testing_epoch.append(i_epoch / 1000)
                train_option = False
                u_true2test, u_predict2test = sess.run([U_true, U_NN], feed_dict={X_it: test_x_bach, train_opt: train_option})
                mse2test = np.mean(np.square(u_true2test - u_predict2test))
                test_mse_all.append(mse2test)
                res2test = mse2test / np.mean(np.square(u_true2test))
                test_rel_all.append(res2test)

                DNN_tools.print_and_log_test_one_epoch(mse2test, res2test, log_out=log_fileout)

        # -----------------------  save training results to mat files, then plot them ---------------------------------
        saveData.save_trainLoss2mat_1actFunc(loss_it_all, loss_bd_all, loss_all, actName=act_func, outPath=R['FolderName'])
        plotData.plotTrain_loss_1act_func(loss_it_all, lossType='loss_it', seedNo=R['seed'], outPath=R['FolderName'])
        plotData.plotTrain_loss_1act_func(loss_bd_all, lossType='loss_bd', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])

        saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=act_func, outPath=R['FolderName'])
        plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=act_func, seedNo=R['seed'],
                                             outPath=R['FolderName'], yaxis_scale=True)

        # ----------------------  save testing results to mat files, then plot them --------------------------------
        saveData.save_2testSolus2mat(u_true2test, u_predict2test, actName='utrue', actName1=act_func, outPath=R['FolderName'])
        plotData.plot_2solutions2test(u_true2test, u_predict2test, coord_points2test=test_x_bach,
                                      batch_size2test=test_batch_size, seedNo=R['seed'], outPath=R['FolderName'],
                                      subfig_type=R['subfig_type'])

        saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=act_func, outPath=R['FolderName'])
        plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, testing_epoch, actName=act_func,
                                  seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)


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
    store_file = 'laplace1d'
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

    # R['PDE_type'] = 'general_laplace'
    # R['eqs_name'] = 'PDE1'
    # R['eqs_name'] = 'PDE2'
    # R['eqs_name'] = 'PDE3'
    # R['eqs_name'] = 'PDE4'
    # R['eqs_name'] = 'PDE5'
    # R['eqs_name'] = 'PDE6'
    # R['eqs_name'] = 'PDE7'

    R['PDE_type'] = 'p_laplace'
    R['eqs_name'] = 'multi_scale'

    if R['PDE_type'] == 'general_laplace':
        R['epsilon'] = 0.1
        R['order2laplace'] = 2
    elif R['PDE_type'] == 'p_laplace':
        # 频率设置
        epsilon = input('please input epsilon =')         # 由终端输入的会记录为字符串形式
        R['epsilon'] = float(epsilon)                     # 字符串转为浮点

        # 问题幂次
        order2p_laplace = input('please input the order(a int number) to p-laplace:')
        order = float(order2p_laplace)
        R['order2laplace'] = order

    R['input_dim'] = 1                                    # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1                                   # 输出维数

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of DNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 训练集的设置
    R['batch_size2integral'] = 3000                       # 内部训练数据的批大小

    # 装载测试数据模式和画图
    R['plot_ongoing'] = 0
    R['subfig_type'] = 1
    R['testData_model'] = 'loadData'

    R['variational_loss'] = 1  # PDE变分

    R['optimizer_name'] = 'Adam'                          # 优化器
    R['learning_rate'] = 2e-4                             # 学习率
    R['learning_rate_decay'] = 5e-5                       # 学习率 decay
    # R['train_group'] = 1
    R['train_group'] = 0

    R['weight_biases_model'] = 'general_model'            # 权重和偏置生成模式
    # R['weight_biases_model'] = 'phase_shift_model'

    R['regular_weight_model'] = 'L0'
    # R['regular_weight_model'] = 'L1'
    # R['regular_weight_model'] = 'L2'
    R['regular_weight_biases'] = 0.000                    # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.001                  # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.0025                 # Regularization parameter for weights

    # 边界的惩罚处理方式,以及边界的惩罚因子
    R['activate_penalty2bd_increase'] = 1
    # R['init_boundary_penalty'] = 1000                   # Regularization parameter for boundary conditions
    R['init_boundary_penalty'] = 100                      # Regularization parameter for boundary conditions

    # 网络的频率范围设置
    R['freqs'] = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)

    # &&&&&&&&&&&&&&&&&&& 使用的网络模型 &&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['model'] = 'PDE_DNN'
    # R['model'] = 'PDE_DNN_BN'
    R['model'] = 'PDE_DNN_scale'
    # R['model'] = 'PDE_DNN_adapt_scale'
    # R['model'] = 'PDE_DNN_FourierBase'
    # R['model'] = 'PDE_DNN_Cos_C_Sin_Base'
    # R['model'] = 'PDE_DNN_WaveletBase'
    # R['model'] = 'PDE_CPDNN'

    # &&&&&&&&&&&&&&&&&&&&&& 隐藏层的层数和每层神经元数目 &&&&&&&&&&&&&&&&&&&&&&&&&&&&
    if R['model'] == 'PDE_DNN_Cos_C_Sin_Base':
        if R['order2laplace'] == 2:
            if R['epsilon'] == 0.1:
                R['hidden_layers'] = (100, 100, 80, 80, 60)  # 1*200+200*100+100*80+80*80+80*60+60*1= 39460 个参数
            else:
                R['hidden_layers'] = (125, 100, 80, 80, 60)  # 1*250+250*100+100*80+80*80+80*60+60*1= 44510 个参数
        elif R['order2laplace'] == 5:
            if R['epsilon'] == 0.1:
                R['hidden_layers'] = (100, 120, 80, 80, 80)  # 1*200+200*120+120*80+80*80+80*80+80*1= 46680 个参数
            else:
                R['hidden_layers'] = (125, 120, 80, 80, 80)  # 1*250+250*120+120*80+80*80+80*80+80*1= 52730 个参数
        elif R['order2laplace'] == 8:
            if R['epsilon'] == 0.1:
                R['hidden_layers'] = (100, 150, 100, 100, 80)  # 1*200+200*150+150*100+100*100+100*80+80*1= 63280 个参数
            else:
                R['hidden_layers'] = (125, 150, 100, 100, 80)  # 1*250+250*150+150*100+100*100+100*80+80*1= 70830 个参数
    else:
        if R['order2laplace'] == 2:
            if R['epsilon'] == 0.1:
                R['hidden_layers'] = (200, 100, 80, 80, 60)  # 1*200+200*100+100*80+80*80+80*60+60*1= 39460 个参数
            else:
                R['hidden_layers'] = (250, 100, 80, 80, 60)  # 1*250+250*100+100*80+80*80+80*60+60*1= 44510 个参数
        elif R['order2laplace'] == 5:
            if R['epsilon'] == 0.1:
                R['hidden_layers'] = (200, 120, 80, 80, 80)  # 1*200+200*120+120*80+80*80+80*80+80*1= 46680 个参数
            else:
                R['hidden_layers'] = (250, 120, 80, 80, 80)  # 1*250+250*120+120*80+80*80+80*80+80*1= 52730 个参数
        elif R['order2laplace'] == 8:
            if R['epsilon'] == 0.1:
                R['hidden_layers'] = (200, 150, 100, 100, 80)  # 1*200+200*150+150*100+100*100+100*80+80*1= 63280 个参数
            else:
                R['hidden_layers'] = (250, 150, 100, 100, 80)  # 1*250+250*150+150*100+100*100+100*80+80*1= 70830 个参数
        else:
            R['hidden_layers'] = (300, 200, 150, 150, 100, 50, 50)
            # R['hidden_layers'] = (400, 300, 300, 200, 100, 100, 50)
            # R['hidden_layers'] = (500, 400, 300, 300, 200, 100)
            # R['hidden_layers'] = (500, 400, 300, 200, 200, 100)
            # R['hidden_layers'] = (500, 400, 300, 300, 200, 100, 100)
            # R['hidden_layers'] = (500, 300, 200, 200, 100, 100, 50)
            # R['hidden_layers'] = (1000, 800, 600, 400, 200)
            # R['hidden_layers'] = (1000, 500, 400, 300, 300, 200, 100, 100)
            # R['hidden_layers'] = (2000, 1500, 1000, 500, 250)

    # &&&&&&&&&&&&&&&&&&& 激活函数的选择 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['activate_func'] = 'relu'
    # R['activate_func'] = 'tanh'
    # R['activate_func'] = 'sintanh'
    # R['activate_func']' = leaky_relu'
    # R['activate_func'] = 'srelu'
    R['activate_func'] = 's2relu'
    # R['activate_func'] = 's3relu'
    # R['activate_func'] = 'csrelu'
    # R['activate_func'] = 'gauss'
    # R['activate_func'] = 'singauss'
    # R['activate_func'] = 'metican'
    # R['activate_func'] = 'modify_mexican'
    # R['activate_func'] = 'leaklysrelu'
    # R['activate_func'] = 'slrelu'
    # R['activate_func'] = 'elu'
    # R['activate_func'] = 'selu'
    # R['activate_func'] = 'phi'

    solve_Integral_Equa(R)

