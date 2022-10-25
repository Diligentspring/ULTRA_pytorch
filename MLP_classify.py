import sys
import re
import os
import numpy as np

runs = ['', '_run2', '_run3', '_run4', '_run5']
# runs = ['_run1', '_run2', '_run3', '_run4', '_run5']
ws = ['-2', '0', '2']

for r in runs:
    for w in ws:
        result_path = 'train_pl_{}_click{}_result.txt'.format(w, r)
        # result_path = sys.argv[1]
        result_file = open(result_path, 'r', encoding='utf-8')

        lines = result_file.readlines()
        #print(lines)

        rule_score = 'ndcg_5:(.*),'
        rule_step = ',step (.*)\n'

        model_ssdic = {}
        flag = 1 #flag为1表示目前是一个snippet的第一行
        for i in range(len(lines)):
            line = lines[i]
            if flag ==1:
                valid_score = 0
                valid_step = 0
                training_valid_score = 0
                training_valid_step = 0
                model = line.strip('\n').strip('_256_0.1').strip('_256_0.05').strip('_256_0.01')
                if model not in model_ssdic.keys():
                    model_ssdic[model] = [[], [], [], []]
                # print(model)
                flag = 0
            else:
                score = re.findall(rule_score, line)
                step = re.findall(rule_step, line)
                # print(score)
                # print(step)
                if score and step:
                    if valid_score == 0:
                        valid_score = float(score[0])
                        valid_step = int(step[0])
                        model_ssdic[model][0].append(valid_score)
                        model_ssdic[model][1].append(valid_step)
                    else:
                        training_valid_score = float(score[0])
                        training_valid_step = int(step[0])
                        model_ssdic[model][2].append(training_valid_score)
                        model_ssdic[model][3].append(training_valid_step)
                if line == '\n':
                    flag = 1
        print(model_ssdic)

        for k in model_ssdic.keys():
            max_valid_step = model_ssdic[k][1][np.argmax(model_ssdic[k][0])]
            if np.argmax(model_ssdic[k][0]) == 0:
                lr = '0.1'
            elif np.argmax(model_ssdic[k][0]) == 1:
                lr = '0.05'
            elif np.argmax(model_ssdic[k][0]) == 2:
                lr = '0.01'
            model_split = k.strip().split('_')
            model_reset = ''
            algorithm = 'DLA'
            if model_split[-1] == 'PBMDLA':
                for i in range(len(model_split)):
                    if i != len(model_split)-1 :
                        model_reset = model_reset + model_split[i] + '_'
                    else:
                        model_reset = model_reset + 'DLAPBM'
            elif model_split[-1] == 'DCMDLA':
                for i in range(len(model_split)):
                    if i != len(model_split)-1 :
                        model_reset = model_reset + model_split[i] + '_'
                    else:
                        model_reset = model_reset + 'DLADCM'
            elif model_split[-1] == 'clickMLP':
                model_reset = k
                algorithm = 'NavieAlgorithm'
            rs = os.popen("python main_2.py --train_data_prefix train --setting_file ./ULTRE_train_pl_0_click/dla_exp_settings.json \
            --batch_size 300 --test_only True --data_dir ./ULTRE_train_pl_-2_click/ \
            --model_dir /data/work/niuzechun/ULTRE_train_pl_{}_click/model{}/{}_256_{}/ultra.learning_algorithm.{}.ckpt{}".format(w, r, model_reset, lr, algorithm, max_valid_step))
            print(rs.read())
            rs.close()

            max_training_valid_step = model_ssdic[k][3][np.argmax(model_ssdic[k][2])]
            if np.argmax(model_ssdic[k][2]) == 0:
                lr = '0.1'
            elif np.argmax(model_ssdic[k][2]) == 1:
                lr = '0.05'
            elif np.argmax(model_ssdic[k][2]) == 2:
                lr = '0.01'
            print(max_training_valid_step)
            rs=os.popen("python main_training-test.py --test_data_prefix training_test --setting_file ./ULTRE_train_pl_0_click/dla_exp_settings.json \
                       --batch_size 700 --test_only True --data_dir ./ULTRE_train_pl_0_click/ \
                       --model_dir /data/work/niuzechun/ULTRE_train_pl_{}_click/model{}/{}_256_{}/ultra.learning_algorithm.{}.ckpt{}".format(w, r, model_reset, lr, algorithm, max_training_valid_step))
            print(rs.read())
            rs.close()

# python main_2.py --train_data_prefix train --setting_file ./ULTRE_train_pl_0_click/dla_exp_settings.json --batch_size 300 \
# --test_only True --data_dir ./ULTRE_train_pl_-2_click/ \
# --model_dir /data/work/niuzechun/ULTRE_train_pl_-2_click/model_run4/PBM_eta0_DLAPBM_256_0.05/ultra.learning_algorithm.DLA.ckpt1700

# python main_training-test.py --test_data_prefix training_test --setting_file ./ULTRE_train_pl_0_click/dla_exp_settings.json \
# --batch_size 700 --test_only True --data_dir ./ULTRE_train_pl_0_click/ \
# --model_dir ./ULTRE_train_pl_0_click/model_run/PBM_eta0_DLADCM_256_0.1/ultra.learning_algorithm.DLA.ckpt6050
