import numpy as np
import sys
import xlwt
import os

def DCG(label_list):
    dcgsum = 0
    for i in range(len(label_list)):
        dcg = (2**label_list[i] - 1)/np.log2(i+2)
        dcgsum += dcg
    return dcgsum


#ndcg 计算
def NDCG(label_list, top_n):
    #没有设定topn
    if top_n==None:
        dcg = DCG(label_list)
        ideal_list = sorted(label_list, reverse=True)
        ideal_dcg = DCG(ideal_list)
        if ideal_dcg == 0:
            return 0
        return dcg/ideal_dcg
    #设定top n
    else:
        dcg = DCG(label_list[0:top_n])
        ideal_list = sorted(label_list, reverse=True)
        ideal_dcg = DCG(ideal_list[0:top_n])
        if ideal_dcg == 0:
            return 0
        return dcg/ideal_dcg

# weights = ['-2', '0', '2']
# runs = ['', '_run2', '_run3', '_run4', '_run5']

weights = ['-2']
# runs = ['', '_run2']
#runs = ['_run3']
#runs = ['_run4', '_run5']
# runs = ['']
runs = ['_run1', '_run2', '_run3', '_run4', '_run5']

models = ['PBM_eta0d5', 'PBM_eta1', 'PBM_eta2', 'DCM_beta1_eta0d5', 'DCM_beta0d6_eta1', 'DCM_beta0d6_eta2', \
          'CBCM_w0d2_eta0d5', 'CBCM_w0d4_eta0d75', 'CBCM_w0d6_eta1']
# algorithms = ['DLAPBM', 'DLADCM', 'clickMLP']

workbook = xlwt.Workbook(encoding='utf-8')
worksheet = workbook.add_sheet("training_test")
worksheet.write(0, 0, "production-ranker")
worksheet.write(0, 1, "click-model")
worksheet.write(0, 2, "position-bias")
worksheet.write(0, 3, "ULTR-model")
worksheet.write(0, 4, "query")
worksheet.write(0, 5, "ndcg@5")
row_count = 1

for w in weights:
    for r in runs:
        # path = 'classify_training_test/ULTRE_train_pl_{}_click/model{}'.format(w, r)
        path = 'classify_training_test/ULTRE_train_sample_1_svm_click/model{}'.format(r)
        for filename in os.listdir(path):
            print('_'.join(filename.strip().split('_')[:-3]))
            print('_'.join(filename.strip().split('_')[:-3]) in models)
            if '_'.join(filename.strip().split('_')[:-3]) not in models:
                continue

            if filename.strip().split('_')[-3] != 'prs':
                continue

            prefile = open(path+'/'+filename, "r")
            label_file = open('/home/niuzechun/svm_rank/click/training_test.labels', "r", encoding="utf-8")
            print(w + r + filename)
            ndcg = 0.0
            for i in range(700):
                rel = []
                for j in range(10):
                    rel.append(float(prefile.readline().strip('\n')))
                    #rel.append(random.randint(0,9))

                # rank = np.argsort(rel)
                rank=(sorted(range(len(rel)), key=lambda k: rel[k], reverse=True))

                #print(rel)
                #print(rank)
                line = label_file.readline().strip('\n')
                qid = line.split(':')[0]
                removeqid = line.split(':')[1]
                label = removeqid.strip().split(' ')
                #print(label)
                prelabel = []
                for k in range(10):
                    prelabel.append(int(label[rank[k]]))
                #print(prelabel)
                #print(NDCG(prelabel, 5))
                query_ndcg = NDCG(prelabel, 5)
                ndcg = query_ndcg + ndcg

                worksheet.write(row_count, 0, 'PL{}'.format(w))

                click_model_name = filename.split('_')[0]
                worksheet.write(row_count, 1, click_model_name)

                para1 = filename.split('_')[1]
                para2 = filename.split('_')[2]
                if para1 == 'w0d2' or para1 == 'beta1' or para1 == 'eta0d5':
                    bias_name = 'low'
                elif para1 == 'w0d4' or para1 == 'beta0d6' and para2 == 'eta1' or para1 == 'eta1':
                    bias_name = 'mid'
                elif para1 == 'w0d6' or para1 == 'beta0d6' and para2 == 'eta2' or para1 == 'eta2':
                    bias_name = 'high'
                worksheet.write(row_count, 2, bias_name)

                model_name = filename.split('_')[-3]
                worksheet.write(row_count, 3, model_name)

                worksheet.write(row_count, 4, qid)

                worksheet.write(row_count, 5, query_ndcg)
                row_count = row_count + 1

            ave_ndcg = ndcg / 700
            print(str(ave_ndcg))
            label_file.close()

workbook.save("ANOVA_training_test_query_train_sample_1_svm_prs.xls")




