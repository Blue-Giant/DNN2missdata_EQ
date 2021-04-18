import DNN_tools


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout):
    DNN_tools.log_string('Type for problem: %s\n' % (R_dic['PDE_type']), log_fileout)
    DNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['eqs_name']), log_fileout)

    if R_dic['activate_stop'] != 0:
        DNN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        DNN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']), log_fileout)

    if R_dic['sub_networks'] == 'subDNNs':
        DNN_tools.log_string('Using some subnetworks for solving problem\n', log_fileout)
    else:
        DNN_tools.log_string('Using one network for solving problem\n', log_fileout)

    if R_dic['bnn2beta'] == 'explicitly_related':
        DNN_tools.log_string('bNN 和参数 beta 显式相关, 即参数 beta 作为bNN的一个输入\n', log_fileout)
    else:
        DNN_tools.log_string('bNN 和参数 beta 隐式相关, 即参数 beta 不作为bNN的一个输入\n', log_fileout)

    DNN_tools.log_string('Network-name of solving problem: %s\n' % str(R_dic['name2network']), log_fileout)
    DNN_tools.log_string('Activate function for network: %s\n' % str(R_dic['activate_func']), log_fileout)
    if R_dic['name2network'] != 'DNN':
        DNN_tools.log_string('The frequency to neural network: %s\n' % (R_dic['freqs']), log_fileout)

    if (R_dic['optimizer_name']).title() == 'Adam':
        DNN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        DNN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']), log_fileout)
    if R_dic['train_model'] == 'Combine_train':
        DNN_tools.log_string('Training model: Combine training\n', log_fileout)

        DNN_tools.log_string('Init learning rate: %s\n' % str(R_dic['init_learning_rate']), log_fileout)
        DNN_tools.log_string('Decay to learning rate: %s\n' % str(R_dic['learning_rate_decay']), log_fileout)

        if R_dic['train_group'] == 0:
            DNN_tools.log_string('Training total loss \n', log_fileout)
        elif R_dic['train_group'] == 1:
            DNN_tools.log_string('Training total loss + parts loss \n', log_fileout)
        elif R_dic['train_group'] == 2:
            DNN_tools.log_string('Training parts loss \n', log_fileout)
    else:
        DNN_tools.log_string('Training model: Alternate training\n', log_fileout)
        DNN_tools.log_string('Init learning rate for b: %s\n' % str(R_dic['init_learning_rate2b']), log_fileout)
        DNN_tools.log_string('Init learning rate for Seff: %s\n' % str(R_dic['init_learning_rate2S']), log_fileout)
        DNN_tools.log_string('Decay to learning rate for b: %s\n' % str(R_dic['learning_rate_decay2b']), log_fileout)
        DNN_tools.log_string('Decay to learning rate for Seff: %s\n' % str(R_dic['learning_rate_decay2S']), log_fileout)

    DNN_tools.log_string('hidden layer:%s\n' % str(R_dic['hidden_layers']), log_fileout)

    DNN_tools.log_string('Batch-size for solving B and Seff(data size) : %s\n' % str(R_dic['batch_size2bS']), log_fileout)
    DNN_tools.log_string('Batch-size for solving b 2 y : %s\n' % str(R_dic['batch_size2y_b']), log_fileout)
    DNN_tools.log_string('Batch-size for solving integral function: %s\n' % str(R_dic['batch_size2integral']), log_fileout)


def print_and_log_train_one_epoch(i_epoch, run_time, tmp_lr, temp_penalty_bd, pwb, loss_it_tmp,
                                  loss_bd_tmp, loss_tmp, train_mse_tmp, train_res_tmp, log_out=None):
    # 将运行结果打印出来
    print('train epoch: %d, time: %.3f' % (i_epoch, run_time))
    print('learning rate: %f' % tmp_lr)
    print('boundary penalty: %f' % temp_penalty_bd)
    print('weights and biases with  penalty: %f' % pwb)
    print('loss_it for training: %.10f' % loss_it_tmp)
    print('loss_bd for training: %.10f' % loss_bd_tmp)
    print('loss for training: %.10f' % loss_tmp)
    print('solution mean square error for training: %.10f' % train_mse_tmp)
    print('solution residual error for training: %.10f\n' % train_res_tmp)

    DNN_tools.log_string('train epoch: %d,time: %.3f' % (i_epoch, run_time), log_out)
    DNN_tools.log_string('learning rate: %f' % tmp_lr, log_out)
    DNN_tools.log_string('boundary penalty: %f' % temp_penalty_bd, log_out)
    DNN_tools.log_string('weights and biases with  penalty: %f' % pwb, log_out)
    DNN_tools.log_string('loss_it for training: %.10f' % loss_it_tmp, log_out)
    DNN_tools.log_string('loss_bd for training: %.10f' % loss_bd_tmp, log_out)
    DNN_tools.log_string('loss for training: %.10f' % loss_tmp, log_out)
    DNN_tools.log_string('solution mean square error for training: %.10f' % train_mse_tmp, log_out)
    DNN_tools.log_string('solution residual error for training: %.10f\n' % train_res_tmp, log_out)


def print_and_log_test_one_epoch(mse2test, res2test, log_out=None):
    # 将运行结果打印出来
    print('mean square error of predict and real for testing: %.10f' % mse2test)
    print('residual error of predict and real for testing: %.10f\n' % res2test)

    DNN_tools.log_string('mean square error of predict and real for testing: %.10f' % mse2test, log_out)
    DNN_tools.log_string('residual error of predict and real for testing: %.10f\n\n' % res2test, log_out)