import os

models = ['log_oraclePBM_clipped']

#batchsizes = [256, 512, 1024, 2048]
batchsizes = [256]
lrs = [0.1, 0.05, 0.01, 0.001]

for m in models:
    for b in batchsizes:
        for l in lrs:
            file = open('log_IPS_result.txt', 'a')
            # rs = os.popen("python main.py --data_dir ./ULTRE_IPS/ --train_data_prefix {} --model_dir \
            # ./ULTRE_IPS/model/{}_{}_{}  --setting_file ./example/offline_setting/naive_algorithm_directlabel_exp_settings.json \
            # --batch_size {} --ln {}".format(m, m, b, l, b, l))
            rs = os.popen("python main.py --data_dir ./ULTRE_log_IPS/ --train_data_prefix {} --model_dir \
                        ./ULTRE_log_IPS/model/{}_{}_{} --setting_file ./example/offline_setting/naive_algorithm_directlabel_exp_settings.json \
                        --batch_size {} --ln {} --max_train_iteration 100000".format(m, m, b, l, b, l))
            result = str(rs.read())[-100:-1]
            rs.close()
            file.write(m+'_'+str(b)+'_'+str(l)+'\n'+result+'\n\n')
            file.close()

