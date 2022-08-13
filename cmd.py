import os

# models = ['PBM_eta0', 'PBM_eta0d5', 'PBM_eta1', 'PBM_eta2', 'PBM_eta3', 'DCM_beta0d6_eta0d5',
#           'DCM_beta0d6_eta1', 'DCM_beta0d6_eta2', 'DCM_beta1_eta0d5', 'DCM_beta1_eta1', 'DCM_beta1_eta2']

models = ['PBM_eta0', 'PBM_eta0d5', 'PBM_eta1', 'PBM_eta2', 'DCM_beta1_eta0d5', 'DCM_beta0d6_eta1', 'DCM_beta0d6_eta2']

#batchsizes = [256, 512, 1024, 2048]
batchsizes = [256]
# lrs = [0.1, 0.05, 0.01, 0.001]
lrs = [0.1, 0.05, 0.01]
for m in models:
    for b in batchsizes:
        for l in lrs:
            file = open('train_pl_0_click_run2_result.txt', 'a')
            # rs = os.popen("python main.py --data_dir ./ULTRE_IPS/ --train_data_prefix {} --model_dir \
            # ./ULTRE_IPS/model/{}_{}_{}  --setting_file ./example/offline_setting/naive_algorithm_directlabel_exp_settings.json \
            # --batch_size {} --ln {}".format(m, m, b, l, b, l))
            rs = os.popen("python main.py --data_dir ./ULTRE_train_pl_0_click/ --train_data_prefix {} --model_dir \
                        ./ULTRE_train_pl_0_click/model_run2/{}_DLAPBM_{}_{} --setting_file ./ULTRE_train_pl_0_click/dla_exp_settings.json \
                        --batch_size {} --ln {} --max_train_iteration 10000".format(m, m, b, l, b, l))
            result = str(rs.read())[-200:-1]
            rs.close()
            file.write(m+'_PBMDLA'+'_'+str(b)+'_'+str(l)+'\n'+result+'\n\n')
            file.close()


for m in models:
    for b in batchsizes:
        for l in lrs:
            file = open('train_pl_2_click_run2_result.txt', 'a')
            # rs = os.popen("python main.py --data_dir ./ULTRE_IPS/ --train_data_prefix {} --model_dir \
            # ./ULTRE_IPS/model/{}_{}_{}  --setting_file ./example/offline_setting/naive_algorithm_directlabel_exp_settings.json \
            # --batch_size {} --ln {}".format(m, m, b, l, b, l))
            rs = os.popen("python main.py --data_dir ./ULTRE_train_pl_2_click/ --train_data_prefix {} --model_dir \
                        ./ULTRE_train_pl_2_click/model_run2/{}_DLAPBM_{}_{} --setting_file ./ULTRE_train_pl_2_click/dla_exp_settings.json \
                        --batch_size {} --ln {} --max_train_iteration 10000".format(m, m, b, l, b, l))
            result = str(rs.read())[-200:-1]
            rs.close()
            file.write(m+'_PBMDLA'+'_'+str(b)+'_'+str(l)+'\n'+result+'\n\n')
            file.close()

for m in models:
    for b in batchsizes:
        for l in lrs:
            file = open('train_pl_-2_click_run2_result.txt', 'a')
            # rs = os.popen("python main.py --data_dir ./ULTRE_IPS/ --train_data_prefix {} --model_dir \
            # ./ULTRE_IPS/model/{}_{}_{}  --setting_file ./example/offline_setting/naive_algorithm_directlabel_exp_settings.json \
            # --batch_size {} --ln {}".format(m, m, b, l, b, l))
            rs = os.popen("python main.py --data_dir ./ULTRE_train_pl_-2_click/ --train_data_prefix {} --model_dir \
                        ./ULTRE_train_pl_-2_click/model_run2/{}_DLAPBM_{}_{} --setting_file ./ULTRE_train_pl_-2_click/dla_exp_settings.json \
                        --batch_size {} --ln {} --max_train_iteration 10000".format(m, m, b, l, b, l))
            result = str(rs.read())[-200:-1]
            rs.close()
            file.write(m+'_PBMDLA'+'_'+str(b)+'_'+str(l)+'\n'+result+'\n\n')
            file.close()