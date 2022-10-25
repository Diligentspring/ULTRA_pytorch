import os

# models = ['PBM_eta0', 'PBM_eta0d5', 'PBM_eta1', 'PBM_eta2', 'PBM_eta3', 'DCM_beta0d6_eta0d5',
#           'DCM_beta0d6_eta1', 'DCM_beta0d6_eta2', 'DCM_beta1_eta0d5', 'DCM_beta1_eta1', 'DCM_beta1_eta2']
models = ['PBM_eta0d5', 'PBM_eta1', 'PBM_eta2', 'DCM_beta1_eta0d5', 'DCM_beta0d6_eta1', 'DCM_beta0d6_eta2',
          'CBCM_w0d2_eta0d5', 'CBCM_w0d4_eta0d75', 'CBCM_w0d6_eta1']
#batchsizes = [256, 512, 1024, 2048]
batchsizes = [256]
# lrs = [0.1, 0.05, 0.01, 0.001]
lrs = [0.1, 0.05, 0.01]
runs = [1, 2, 3, 4, 5]

for run in runs:
    for m in models:
        for b in batchsizes:
            for l in lrs:
                file = open('train_pl_0d7_click_run{}_result.txt'.format(run), 'a')
                # file = open('train_sample_1_svm_prs_result.txt', 'a')
                # rs = os.popen("python main.py --data_dir ./ULTRE_IPS/ --train_data_prefix {} --model_dir \
                # ./ULTRE_IPS/model/{}_{}_{}  --setting_file ./example/offline_setting/naive_algorithm_directlabel_exp_settings.json \
                # --batch_size {} --ln {}".format(m, m, b, l, b, l))

                rs = os.popen("python main.py --data_dir ./ULTRE_train_pl_0d7_click/ --train_data_prefix {} --model_dir \
                            ./ULTRE_train_pl_0d7_click/model_run{}/{}_DLAPBM_{}_{} --setting_file ./ULTRE_train_pl_0_click/dla_exp_settings.json \
                            --batch_size {} --ln {} --seed {} --max_train_iteration 10000".format(m, run, m, b, l, b, l, run))

                # rs = os.popen("python main.py --data_dir ./ULTRE_train_sample_1_svm_click/ --train_data_prefix {} --model_dir \
                #                         ./ULTRE_train_sample_1_svm_click/model_run{}/{}_clickMLP_{}_{} --setting_file ./ULTRE_train_pl_0_click/naive_exp_settings.json \
                #                         --batch_size {} --ln {} --seed {} --max_train_iteration 10000".format(m, run, m, b, l, b, l, run))

                # rs = os.popen("python main.py --data_dir ./ULTRE_train_sample_1_svm_click/ --train_data_prefix {} --model_dir \
                #                                  ./ULTRE_train_sample_1_svm_click/model_run1/{}_prs_{}_{} --setting_file ./ULTRE_train_pl_-2_click/prs_exp_settings.json \
                #                                  --batch_size {} --ln {} --max_train_iteration 10000".format(m, m, b, l, b, l))

                result = str(rs.read())[-200:-1]
                rs.close()
                # file.write(m + '_clickMLP' + '_' + str(b) + '_' + str(l) + '\n' + result + '\n\n')
                # file.write(m + '_DCMDLA' + '_' + str(b) + '_' + str(l) + '\n' + result + '\n\n')
                file.write(m+'_PBMDLA'+'_'+str(b)+'_'+str(l)+'\n'+result+'\n\n')
                file.close()

# for m in models:
#     for b in batchsizes:
#         for l in lrs:
#             file = open('train_sample_1_svm_click_result.txt', 'a')
#             # rs = os.popen("python main.py --data_dir ./ULTRE_IPS/ --train_data_prefix {} --model_dir \
#             # ./ULTRE_IPS/model/{}_{}_{}  --setting_file ./example/offline_setting/naive_algorithm_directlabel_exp_settings.json \
#             # --batch_size {} --ln {}".format(m, m, b, l, b, l))
#
#             rs = os.popen("python main.py --data_dir ./ULTRE_train_sample_1_svm_click/ --train_data_prefix {} --model_dir \
#                         ./ULTRE_train_sample_1_svm_click/model/{}_DLAPBM_{}_{} --setting_file ./ULTRE_train_pl_0_click/dla_exp_settings.json \
#                         --batch_size {} --ln {} --seed {} --max_train_iteration 10000".format(m, m, b, l, b, l, r))
#
#             # rs = os.popen("python main.py --data_dir ./ULTRE_train_pl_0_click/ --train_data_prefix {} --model_dir \
#             #                         ./ULTRE_train_pl_0_click/model_run5/{}_clickMLP_{}_{} --setting_file ./ULTRE_train_pl_0_click/naive_exp_settings.json \
#             #                         --batch_size {} --ln {} --max_train_iteration 10000".format(m, m, b, l, b, l))
#
#             # rs = os.popen("python main.py --data_dir ./ULTRE_train_pl_0_click/ --train_data_prefix {} --model_dir \
#             #                                  ./ULTRE_train_pl_0_click/model_run2/{}_prs_{}_{} --setting_file ./ULTRE_train_pl_-2_click/prs_exp_settings.json \
#             #                                  --batch_size {} --ln {} --max_train_iteration 10000".format(m, m, b, l, b, l))
#
#             result = str(rs.read())[-200:-1]
#             rs.close()
#             file.write(m + '_PBMDLA' + '_' + str(b) + '_' + str(l) + '\n' + result + '\n\n')
#             # file.write(m+'_prs'+'_'+str(b)+'_'+str(l)+'\n'+result+'\n\n')
#             file.close()

# for m in models:
#     for b in batchsizes:
#         for l in lrs:
#             file = open('train_pl_2_prs_run2_result.txt', 'a')
#
#             rs = os.popen("python main.py --data_dir ./ULTRE_train_sample_1_svm_click/ --train_data_prefix {} --model_dir \
#                                     ./ULTRE_train_sample_1_svm_click/model/{}_DLAPBM_{}_{} --setting_file ./ULTRE_train_pl_0_click/dla_exp_settings.json \
#                                     --batch_size {} --ln {} --max_train_iteration 10000".format(m, m, b, l, b, l))
#             result = str(rs.read())[-200:-1]
#             rs.close()
#             file.write(m+'_prs'+'_'+str(b)+'_'+str(l)+'\n'+result+'\n\n')
#             file.close()
#
# for m in models:
#     for b in batchsizes:
#         for l in lrs:
#             file = open('train_pl_-2_prs_run2_result.txt', 'a')
#
#             rs = os.popen("python main.py --data_dir ./ULTRE_train_sample_1_svm_click/ --train_data_prefix {} --model_dir \
#                                     ./ULTRE_train_sample_1_svm_click/model/{}_DLAPBM_{}_{} --setting_file ./ULTRE_train_pl_0_click/dla_exp_settings.json \
#                                     --batch_size {} --ln {} --max_train_iteration 10000".format(m, m, b, l, b, l))
#             result = str(rs.read())[-200:-1]
#             rs.close()
#             file.write(m+'_prs'+'_'+str(b)+'_'+str(l)+'\n'+result+'\n\n')
#             file.close()