"""Training and testing unbiased learning to rank algorithms.

See the following paper for more information about different algorithms.

    * Qingyao Ai, Keping Bi, Cheng Luo, Jiafeng Guo, W. Bruce Croft. 2018. Unbiased Learning to Rank with Unbiased Propensity Estimation. In Proceedings of SIGIR '18

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import ultra
import requests
import numpy as np

from ultra.learning_algorithm.dla_dcm import DLA_DCM
from ultra.learning_algorithm.dla_dcmq import DLA_DCMq
from ultra.learning_algorithm.dla_dcma import DLA_DCMa
from ultra.learning_algorithm.dla_ubm import DLA_UBM
from ultra.learning_algorithm.dla_pbm import DLA_PBM
from ultra.learning_algorithm.dla_pbm_em import DLA_PBM_EM
from ultra.learning_algorithm.dla_mcm_em import DLA_MCM_EM
from ultra.learning_algorithm.dla_mcma import DLA_MCMa
from ultra.learning_algorithm.pdgd import PDGD

# rank list size should be read from data
parser = argparse.ArgumentParser(description='Pipeline commandline argument')
parser.add_argument("--data_dir", type=str, default="./tests/data/", help="The directory of the experimental dataset.")
parser.add_argument("--train_data_prefix", type=str, default="train", help="The name prefix of the training data "
                                                                           "in data_dir.")
parser.add_argument("--valid_data_prefix", type=str, default="valid", help="The name prefix of the validation data in "
                                                                           "data_dir.")
parser.add_argument("--test_data_prefix", type=str, default="test",
                    help="The name prefix of the test data in data_dir.")
parser.add_argument("--model_dir", type=str, default="./tests/tmp_model/", help="The directory for model and "
                                                                                "intermediate outputs.")
parser.add_argument("--output_dir", type=str, default="./tests/tmp_output/", help="The directory to output results.")

parser.add_argument("--click_model_dir", type=str, default=None,
                    help="The directory that contains labels produced by the click model")
parser.add_argument("--data_format", type=str, default="ULTRA", help="The format of the data")
# model
parser.add_argument("--setting_file", type=str, default="./example/offline_setting/dla_exp_settings.json",
                    help="A json file that contains all the settings of the algorithm.")

# general training parameters
parser.add_argument("--batch_size", type=int, default=256,
                    help="Batch size to use during training.")
parser.add_argument("--max_list_cutoff", type=int, default=0,
                    help="The maximum number of top documents to consider in each rank list (0: no limit).")
parser.add_argument("--selection_bias_cutoff", type=int, default=10,
                    help="The maximum number of top documents to be shown to user "
                         "(which creates selection bias) in each rank list (0: no limit).")
parser.add_argument("--max_train_iteration", type=int, default=10000,
                    help="Limit on the iterations of training (0: no limit).")
parser.add_argument("--start_saving_iteration", type=int, default=0,
                    help="The minimum number of iterations before starting to test and save models. "
                         "(0: no limit).")
parser.add_argument("--steps_per_checkpoint", type=int, default=50,
                    help="How many training steps to do per checkpoint.")

parser.add_argument("--test_while_train", type=bool, default=False,
                    help="Set to True to test models during the training process.")
parser.add_argument("--test_only", type=bool, default=False,
                    help="Set to True for testing models only.")
parser.add_argument("--ln", type=float, default=0.05,
                    help="Learning rate.")

args = parser.parse_args()


def create_model(exp_settings, data_set):
    """Create model and initialize or load parameters in session.

        Args:
            exp_settings: (dictionary) The dictionary containing the model settings.
            data_set: (Raw_data) The dataset used to build the input layer.
    """

    model = ultra.utils.find_class(exp_settings['learning_algorithm'])(data_set, exp_settings)
    try:
        checkpoint_path = os.path.join(args.model_dir, "%s.ckpt" % exp_settings['learning_algorithm'])
        ckpt = torch.load(checkpoint_path)
        print("Reading model parameters from %s" % checkpoint_path)
        model.model.load_state_dict(ckpt)
        model.model.eval()
    except FileNotFoundError:
        print("Created model with fresh parameters.")
    return model


def train(exp_settings):
    # Prepare data.
    print("Reading data in %s" % args.data_dir)
    train_set = ultra.utils.read_data(args.data_dir, args.train_data_prefix, args.click_model_dir, args.max_list_cutoff)
    ultra.utils.find_class(exp_settings['train_input_feed']).preprocess_data(train_set,
                                                                             exp_settings['train_input_hparams'],
                                                                             exp_settings)
    valid_set = ultra.utils.read_data(args.data_dir, args.valid_data_prefix, args.click_model_dir, args.max_list_cutoff)
    ultra.utils.find_class(exp_settings['train_input_feed']).preprocess_data(valid_set,
                                                                             exp_settings['train_input_hparams'],
                                                                             exp_settings)
    print("Train Rank list size %d" % train_set.rank_list_size)
    print("Valid Rank list size %d" % valid_set.rank_list_size)
    exp_settings['max_candidate_num'] = max(train_set.rank_list_size, valid_set.rank_list_size)

    test_set = ultra.utils.read_data(args.data_dir, args.test_data_prefix, args.max_list_cutoff)
    ultra.utils.find_class(exp_settings['train_input_feed']).preprocess_data(test_set,
                                                                             exp_settings['train_input_hparams'],
                                                                             exp_settings)
    print("Test Rank list size %d" % test_set.rank_list_size)

    if 'selection_bias_cutoff' not in exp_settings:  # check if there is a limit on the number of items per training query.
        exp_settings['selection_bias_cutoff'] = args.selection_bias_cutoff if args.selection_bias_cutoff > 0 else \
            exp_settings['max_candidate_num']

    exp_settings['selection_bias_cutoff'] = min(exp_settings['selection_bias_cutoff'],
                                                exp_settings['max_candidate_num'])
    print('Users can only see the top %d documents for each query in training.' % exp_settings['selection_bias_cutoff'])

    # Pad data
    # train_set.pad(exp_settings['max_candidate_num'])
    # valid_set.pad(exp_settings['max_candidate_num'])

    # Create model based on the input layer.

    exp_settings['ln'] = args.ln

    print("Creating model...")
    # model = create_model(exp_settings, train_set)
    #model = DLA_PBM(train_set, exp_settings)
    model = PDGD(train_set, exp_settings)

    #checkpoint_path = os.path.join(args.model_dir + '_512_0.05', "%s.ckpt4600" % exp_settings['learning_algorithm'])

    # checkpoint_path = os.path.join(args.model_dir, "%s.ckpt48" % exp_settings['learning_algorithm'])
    # ckpt = torch.load(checkpoint_path)
    # print("Reading model parameters from %s" % checkpoint_path)
    # model.model.load_state_dict(ckpt)

    # model.print_info()

    # Create data feed
    train_input_feed = ultra.utils.find_class(exp_settings['train_input_feed'])(model, args.batch_size,
                                                                                exp_settings['train_input_hparams'])
    valid_input_feed = ultra.utils.find_class(exp_settings['valid_input_feed'])(model, args.batch_size,
                                                                                exp_settings['valid_input_hparams'])
    test_input_feed = ultra.utils.find_class(exp_settings['test_input_feed'])(model, args.batch_size,
                                                                                exp_settings['test_input_hparams'])

    # Create tensorboard summarizations.
    train_writer = torch.utils.tensorboard.SummaryWriter(log_dir=args.model_dir + '/train_log')
    valid_writer = torch.utils.tensorboard.SummaryWriter(log_dir=args.model_dir + '/valid_log')
    test_writer = None
    if args.test_while_train:
        test_writer = torch.utils.tensorboard.SummaryWriter(log_dir=args.model_dir + '/test_log')

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    best_perf = None
    best_step = None
    print("max_train_iter: ", args.max_train_iteration)
    #while True:
    for _ in range(2537): #3722
        # Get a batch and make a step.
        start_time = time.time()
        # input_feed, info_map = train_input_feed.get_batch(train_set, check_validation=True,
        #                                                data_format=args.data_format)

        #生成train_ranking_lists
        docid_inputs, letor_features, labels = [], [], []
        rank_list_idxs = []
        for i in range(840):
            rank_list_idxs.append(i)
            train_input_feed.prepare_true_labels_with_index_ULTRA(train_set, i,
                                                docid_inputs, letor_features, labels, True)
        local_batch_size = len(docid_inputs)
        letor_features_length = len(letor_features)
        for i in range(local_batch_size):
            for j in range(train_input_feed.rank_list_size):
                if docid_inputs[i][j] < 0:
                    docid_inputs[i][j] = letor_features_length

        batch_docid_inputs = []
        batch_labels = []
        for length_idx in range(train_input_feed.rank_list_size):
            # Batch encoder inputs are just re-indexed docid_inputs.
            batch_docid_inputs.append(
                np.array([docid_inputs[batch_idx][length_idx]
                          for batch_idx in range(local_batch_size)], dtype=np.float32))
            # Batch decoder inputs are re-indexed decoder_inputs, we create
            # labels.
            batch_labels.append(
                np.array([labels[batch_idx][length_idx]
                          for batch_idx in range(local_batch_size)], dtype=np.float32))
        # Create input feed map
        input_feed = {}
        input_feed[train_input_feed.model.letor_features_name] = np.array(letor_features)
        for l in range(train_input_feed.rank_list_size):
            input_feed[train_input_feed.model.docid_inputs_name[l]] = batch_docid_inputs[l]
            input_feed[train_input_feed.model.labels_name[l]] = batch_labels[l]
        # Create info_map to store other information
        info_map = {
            'rank_list_idxs': rank_list_idxs,
            'input_list': docid_inputs,
            'click_list': labels,
            'letor_features': letor_features
        }

        # Compute ranking scores with input_feed
        rank_scores = train_input_feed.model.validation(input_feed, True)[1]
        if train_input_feed.model.is_cuda_avail:
            rank_scores = rank_scores.cpu()
        #print(rank_scores)


        # Rerank documents and collect clicks
        letor_features_length = len(input_feed[train_input_feed.model.letor_features_name])
        local_batch_size = len(input_feed[train_input_feed.model.docid_inputs_name[0]])
        for i in range(local_batch_size):
            # Get valid doc index
            #valid_idx = train_input_feed.max_candidate_num - 1
            valid_idx = 9
            while valid_idx > -1:
                if input_feed[train_input_feed.model.docid_inputs_name[valid_idx]][i] < letor_features_length:  # a valid doc
                    break
                valid_idx -= 1
            list_len = valid_idx + 1
            #print(list_len)


            def plackett_luce_sampling(score_list):
                # Sample document ranking
                scores = score_list[:list_len]
                scores = scores - max(scores)
                #exp_scores = np.exp(self.hparams.tau * scores)
                exp_scores = np.exp(1 * scores)
                exp_scores = exp_scores.numpy()
                probs = exp_scores / np.sum(exp_scores)
                re_list = np.random.choice(np.arange(list_len),
                                           replace=False,
                                           p=probs,
                                           size=np.count_nonzero(probs))
                # Append unselected documents to the end
                used_indexs = set(re_list)
                unused_indexs = []
                for tmp_index in range(list_len):
                    if tmp_index not in used_indexs:
                        unused_indexs.append(tmp_index)
                re_list = np.append(re_list, unused_indexs).astype(int)
                return re_list

            rerank_list = plackett_luce_sampling(rank_scores[i])
            #print(rerank_list)


            # Rerank documents
            new_docid_list = np.zeros(list_len)
            new_label_list = np.zeros(list_len)
            for j in range(list_len):
                new_docid_list[j] = input_feed[train_input_feed.model.docid_inputs_name[rerank_list[j]]][i]
                new_label_list[j] = input_feed[train_input_feed.model.labels_name[rerank_list[j]]][i]



            # Collect clicks online
            # click_list = None
            #
            # click_list, _, _ = self.click_model.sampleClicksForOneList(
            #     new_label_list[:self.rank_list_size])
            # sample_count = 0
            # while check_validation and sum(
            #         click_list) == 0 and sample_count < self.MAX_SAMPLE_ROUND_NUM:
            #     click_list, _, _ = self.click_model.sampleClicksForOneList(
            #         new_label_list[:self.rank_list_size])
            #     sample_count += 1

            #update input_feed
            for j in range(list_len):
                input_feed[train_input_feed.model.docid_inputs_name[j]][i] = new_docid_list[j]
                if j < train_input_feed.rank_list_size:
                    input_feed[train_input_feed.model.labels_name[j]][i] = new_label_list[j]
                else:
                    input_feed[train_input_feed.model.labels_name[j]][i] = 0
        #print(local_batch_size)
        train_ranking_lists = {}
        for i in range(local_batch_size):
            train_ranking_list = []
            for j in range(list_len):
                train_ranking_list.append(str(int(input_feed[train_input_feed.model.labels_name[j]][i])))
            train_ranking_lists[str(int(input_feed[train_input_feed.model.labels_name[0]][i]/10))] = train_ranking_list
        #print(train_ranking_lists)

        #生成valid_ranking_lists
        rerank_scores = []
        it = 0
        count_batch = 0.0
        batch_size_list = []
        while it < len(valid_set.initial_list):
            v_input_feed, valid_info_map = valid_input_feed.get_next_batch(it, valid_set, check_validation=False)
            _, output_logits, summary = model.validation(v_input_feed)
            batch_size_list.append(len(valid_info_map['input_list']))
            for x in range(batch_size_list[-1]):
                rerank_scores.append(output_logits[x])
            it += batch_size_list[-1]
            count_batch += 1.0
            # print("Testing {:.0%} finished".format(float(it) / len(test_set.initial_list)), end="\r", flush=True)
        # get rerank indexes with new scores
        rerank_lists = []
        for i in range(len(rerank_scores)):
            scores = rerank_scores[i]
            rerank_lists.append(sorted(range(len(scores)), key=lambda k: scores[k], reverse=True))
        # print(rerank_scores)
        # print(len(rerank_lists))
        # print(rerank_lists)

        qdfile = open('ULTRE_online/valid/valid.init_list', 'r')
        query = []
        for line in qdfile:
            if (line != ''):
                q = int(line.strip().split(':')[0])
                query.append(q)
        qdfile.close()

        valid_ranking_lists = {}
        for i in range(60):
            valid_ranking_list = []
            q = query[i]
            for j in range(10):
                valid_ranking_list.append(str(query[i] * 10 + rerank_lists[i][j]))
            valid_ranking_lists[str(q)] = valid_ranking_list
        #print(valid_ranking_lists)

        # 生成test_ranking_lists
        rerank_scores = []
        it = 0
        count_batch = 0.0
        batch_size_list = []
        while it < len(test_set.initial_list):
            t_input_feed, test_info_map = test_input_feed.get_next_batch(it, test_set,
                                                                               check_validation=False)
            _, output_logits, summary = model.validation(t_input_feed)
            batch_size_list.append(len(test_info_map['input_list']))
            for x in range(batch_size_list[-1]):
                rerank_scores.append(output_logits[x])
            it += batch_size_list[-1]
            count_batch += 1.0
            # print("Testing {:.0%} finished".format(float(it) / len(test_set.initial_list)), end="\r", flush=True)
        # get rerank indexes with new scores
        rerank_lists = []
        for i in range(len(rerank_scores)):
            scores = rerank_scores[i]
            rerank_lists.append(sorted(range(len(scores)), key=lambda k: scores[k], reverse=True))
        # print(rerank_scores)
        # print(len(rerank_lists))
        # print(rerank_lists)

        qdfile = open('ULTRE_online/test/test.init_list', 'r')
        query = []
        for line in qdfile:
            if (line != ''):
                q = int(line.strip().split(':')[0])
                query.append(q)
        qdfile.close()

        test_ranking_lists = {}
        for i in range(300):
            test_ranking_list = []
            q = query[i]
            for j in range(10):
                test_ranking_list.append(str(query[i] * 10 + rerank_lists[i][j]))
            test_ranking_lists[str(q)] = test_ranking_list
        #print(test_ranking_lists)

        train_ranking_json_str = json.dumps(train_ranking_lists)
        valid_ranking_json_str = json.dumps(valid_ranking_lists)
        test_ranking_json_str = json.dumps(test_ranking_lists)

        time.sleep(1)

        res_flag = False
        while(res_flag == False):
            try:
                res = requests.post('http://ultre.online/online_service/', \
                                    data={'team_name': 'RUCIR21', 'team_code': 'dI3B16fQFeVmkTyX', \
                                          'model_name': 'PDGD', 'session_num': 32, \
                                          'click_model': 'dcm', \
                                          'train_ranking_lists': train_ranking_json_str, \
                                          'valid_ranking_lists': valid_ranking_json_str, \
                                          'test_ranking_lists': test_ranking_json_str})
                res_flag = True
            except:
                reg_flag = False
                time.sleep(10)

        # get response from Online Service API
        #print(res.text)
        #res.content.decode("utf-8")
        response = json.loads(res.text)
        response_status = response["status"]
        if response_status == 1:
            print('response wrong!')
        response_run_time = response["run_time"]
        response_remain_session_num = response["remain_session_num"]
        response_click_model = response["click_model"]
        sessions = json.loads(response["sessions"])
        #print(sessions)

        tr_docid_inputs, tr_letor_features, tr_labels = [], [], []
        for k in sessions.keys():
            qid = k.split('_')[0]
            clicks = sessions[k]
            if sum(clicks) == 0:
                continue
            base = len(tr_letor_features)
            for x in range(train_input_feed.rank_list_size):
                tr_letor_features.append(
                    train_set.features[int(train_ranking_lists[qid][x])])
            tr_docid_inputs.append(list([base + x for x in range(train_input_feed.rank_list_size)]))
            tr_labels.append(clicks)

        if len(tr_labels) == 0:
            continue

        local_batch_size = len(tr_docid_inputs)
        letor_features_length = len(tr_letor_features)
        for i in range(local_batch_size):
            for j in range(train_input_feed.rank_list_size):
                if tr_docid_inputs[i][j] < 0:
                    tr_docid_inputs[i][j] = letor_features_length

        batch_docid_inputs = []
        batch_labels = []
        for length_idx in range(train_input_feed.rank_list_size):
            # Batch encoder inputs are just re-indexed docid_inputs.
            batch_docid_inputs.append(
                np.array([tr_docid_inputs[batch_idx][length_idx]
                          for batch_idx in range(local_batch_size)], dtype=np.float32))
            # Batch decoder inputs are re-indexed decoder_inputs, we create
            # labels.
            batch_labels.append(
                np.array([tr_labels[batch_idx][length_idx]
                          for batch_idx in range(local_batch_size)], dtype=np.float32))
        # Create input feed map
        tr_input_feed = {}
        tr_input_feed[train_input_feed.model.letor_features_name] = np.array(tr_letor_features)
        for l in range(train_input_feed.rank_list_size):
            tr_input_feed[train_input_feed.model.docid_inputs_name[l]] = batch_docid_inputs[l]
            tr_input_feed[train_input_feed.model.labels_name[l]] = batch_labels[l]
        # Create info_map to store other information
        tr_info_map = {
            'rank_list_idxs': rank_list_idxs,
            'input_list': tr_docid_inputs,
            'click_list': tr_labels,
            'letor_features': tr_letor_features
        }
        #print(tr_input_feed)

        step_loss, _, summary = model.train(tr_input_feed)
        # break
        step_time += (time.time() - start_time) / args.steps_per_checkpoint
        loss += step_loss / args.steps_per_checkpoint
        current_step += 1
        # print("Training at step %s" % model.global_step, summary)
        train_writer.add_scalars("Train_loss", summary, model.global_step)

        # Once in a while, we save checkpoint, print statistics, and run evals.
        if current_step % args.steps_per_checkpoint == 0:
            # Print statistics for the previous epoch.
            print("global step %d learning rate %.4f step-time %.2f loss "
                  "%.4f" % (model.global_step, model.learning_rate,
                            step_time, loss))
            previous_losses.append(loss)

            # parafile = open(args.model_dir + '/para.txt', 'a')
            # parafile.write('current_step:' + str(current_step) + '\n')
            # for i in range(len(model.propensity_parameter)):
            #     parafile.write(str(model.propensity_parameter[i].data) + ' ')
            #     if i % 10 == 9:
            #         parafile.write('\n')
            # parafile.close()
            #
            # para2file = open(args.model_dir + '/para2.txt', 'a')
            # para2file.write('current_step:' + str(current_step) + '\n')
            # for i in range(len(model.propensity_para)):
            #     para2file.write(str(model.propensity_para[i].data) + ' ')
            #     if i % 10 == 9:
            #         para2file.write('\n')
            # para2file.close()

            # relfile = open(args.model_dir + '/rel.txt', 'a')
            # relfile.write('current_step:' + str(current_step) + '\n')
            # relfile.write(str(model.train_rel.data) + '\n')
            # relfile.close()

            # satfile = open(args.model_dir + '/sat.txt', 'a')
            # satfile.write('current_step:' + str(current_step) + '\n')
            # satfile.write(str(model.train_sat.data) + '\n')
            # satfile.close()

            # Validate model
            def validate_model(data_set, data_input_feed):
                it = 0
                count_batch = 0.0
                summary_list = []
                batch_size_list = []
                while it < len(data_set.initial_list):
                    input_feed, info_map = data_input_feed.get_next_batch(
                        it, data_set, check_validation=False, data_format=args.data_format)
                    _, _, summary = model.validation(input_feed)
                    summary_list.append(summary)
                    batch_size_list.append(len(info_map['input_list']))
                    it += batch_size_list[-1]
                    count_batch += 1.0
                return ultra.utils.merge_Summary(summary_list, batch_size_list)
                # return summary_list

            valid_summary = validate_model(valid_set, valid_input_feed)
            valid_writer.add_scalars('Validation_Summary', valid_summary, model.global_step)
            for key, value in valid_summary.items():
                print(key, value)

            if args.test_while_train:
                test_summary = validate_model(test_set, test_input_feed)
                test_writer.add_scalars('Validation Summary while training', valid_summary, model.global_step)
                for key, value in test_summary.items:
                    print(key, value)

            # Save checkpoint if the objective metric on the validation set is better
            if "objective_metric" in exp_settings:
                for key, value in valid_summary.items():
                    if key == exp_settings["objective_metric"]:
                        if current_step >= args.start_saving_iteration:
                            if best_perf == None or best_perf < value:
                                checkpoint_path = os.path.join(args.model_dir,
                                                               "%s.ckpt" % str(
                                                                   exp_settings['learning_algorithm']) + str(
                                                                   model.global_step))
                                torch.save(model.model.state_dict(), checkpoint_path)
                                best_perf = value
                                best_step = model.global_step
                                print('Save model, valid %s:%.4f,step %d' % (key, best_perf, best_step))
                                break
                            else:
                                checkpoint_path = os.path.join(args.model_dir,
                                                               "%s.ckpt" % str(
                                                                   exp_settings['learning_algorithm']) + str(
                                                                   model.global_step))
                                torch.save(model.model.state_dict(), checkpoint_path)
                            print('best valid %s:%.4f,step %d' % (key, best_perf, best_step))

            # Save checkpoint if there is no objective metric
            if best_perf == None and current_step > args.start_saving_iteration:
                checkpoint_path = os.path.join(args.model_dir, "%s.ckpt" % exp_settings['learning_algorithm'])
                torch.save(model.model.state_dict(), checkpoint_path)
            if loss == float('inf'):
                break

            step_time, loss = 0.0, 0.0
            sys.stdout.flush()

            if args.max_train_iteration > 0 and current_step > args.max_train_iteration:
                print("current_step: ", current_step)
                break
    train_writer.close()
    valid_writer.close()
    if args.test_while_train:
        test_writer.close()


# def test(exp_settings):
#     # Load test data.
#     print("Reading data in %s" % args.data_dir)
#     test_set = ultra.utils.read_data(args.data_dir, args.test_data_prefix, args.click_model_dir, args.max_list_cutoff)
#     ultra.utils.find_class(exp_settings['train_input_feed']).preprocess_data(test_set,
#                                                                              exp_settings['train_input_hparams'],
#                                                                              exp_settings)
#     exp_settings['max_candidate_num'] = test_set.rank_list_size
#
#     if 'selection_bias_cutoff' not in exp_settings:  # check if there is a limit on the number of items per training query.
#         exp_settings['selection_bias_cutoff'] = args.selection_bias_cutoff if args.selection_bias_cutoff > 0 else \
#             exp_settings['max_candidate_num']
#     exp_settings['selection_bias_cutoff'] = min(exp_settings['selection_bias_cutoff'],
#                                                 exp_settings['max_candidate_num'])
#     print('Users can only see the top %d documents for each query in training.' % exp_settings['selection_bias_cutoff'])
#
#     test_set.pad(exp_settings['max_candidate_num'])
#
#     # Create model and load parameters.
#     model = create_model(exp_settings, test_set)
#
#     # Create input feed
#     test_input_feed = ultra.utils.find_class(exp_settings['test_input_feed'])(model, args.batch_size,
#                                                                               exp_settings['test_input_hparams'])
#
#     test_writer = SummaryWriter(log_dir = args.model_dir + '/test_log')
#
#     rerank_scores = []
#     summary_list = []
#     # Start testing.
#
#     it = 0
#     count_batch = 0.0
#     batch_size_list = []
#     while it < len(test_set.initial_list):
#         input_feed, info_map = test_input_feed.get_next_batch(it, test_set, check_validation=False)
#         _, output_logits, summary = model.validation(input_feed)
#         summary_list.append(summary)
#         batch_size_list.append(len(info_map['input_list']))
#         for x in range(batch_size_list[-1]):
#             rerank_scores.append(output_logits[x])
#         it += batch_size_list[-1]
#         count_batch += 1.0
#         print("Testing {:.0%} finished".format(float(it) / len(test_set.initial_list)), end="\r", flush=True)
#
#     print("\n[Done]")
#     test_summary = ultra.utils.merge_Summary(summary_list, batch_size_list)
#     print("  eval: %s" % (
#         ' '.join(['%s:%.3f' % (key, value) for key,value in test_summary.items()])
#     ))
#
#     # get rerank indexes with new scores
#     rerank_lists = []
#     for i in range(len(rerank_scores)):
#         scores = rerank_scores[i]
#         rerank_lists.append(sorted(range(len(scores)), key=lambda k: scores[k], reverse=True))
#
#     if not os.path.exists(args.output_dir):
#         os.makedirs(args.output_dir)
#     ultra.utils.output_ranklist(test_set, rerank_scores, args.output_dir, args.test_data_prefix)
#
#     return


def test(exp_settings):
    # Load test data.
    print("Reading data in %s" % args.data_dir)
    test_set = ultra.utils.read_data(args.data_dir, args.test_data_prefix, args.click_model_dir, args.max_list_cutoff)
    ultra.utils.find_class(exp_settings['train_input_feed']).preprocess_data(test_set,
                                                                             exp_settings['train_input_hparams'],
                                                                             exp_settings)
    exp_settings['max_candidate_num'] = test_set.rank_list_size

    if 'selection_bias_cutoff' not in exp_settings:  # check if there is a limit on the number of items per training query.
        exp_settings['selection_bias_cutoff'] = args.selection_bias_cutoff if args.selection_bias_cutoff > 0 else \
            exp_settings['max_candidate_num']
    exp_settings['selection_bias_cutoff'] = min(exp_settings['selection_bias_cutoff'],
                                                exp_settings['max_candidate_num'])
    print('Users can only see the top %d documents for each query in training.' % exp_settings['selection_bias_cutoff'])

    # test_set.pad(exp_settings['max_candidate_num'])

    exp_settings['ln'] = args.ln
    # Create model and load parameters.
    #model = create_model(exp_settings, test_set)
    model = PDGD(test_set, exp_settings)
    checkpoint_path = os.path.join(args.model_dir + '_32_0.1', "%s.ckpt5" % exp_settings['learning_algorithm'])
    ckpt = torch.load(checkpoint_path)
    print("Reading model parameters from %s" % checkpoint_path)
    model.model.load_state_dict(ckpt)
    model.model.eval()

    # Create input feed
    test_input_feed = ultra.utils.find_class(exp_settings['test_input_feed'])(model, args.batch_size,
                                                                              exp_settings['test_input_hparams'])

    test_writer = SummaryWriter(log_dir=args.model_dir + '/test_log')

    rerank_scores = []
    summary_list = []
    # Start testing.

    it = 0
    count_batch = 0.0
    batch_size_list = []
    while it < len(test_set.initial_list):
        input_feed, info_map = test_input_feed.get_next_batch(it, test_set, check_validation=False)
        _, output_logits, summary = model.validation(input_feed)
        summary_list.append(summary)
        batch_size_list.append(len(info_map['input_list']))
        for x in range(batch_size_list[-1]):
            rerank_scores.append(output_logits[x])
        it += batch_size_list[-1]
        count_batch += 1.0
        print("Testing {:.0%} finished".format(float(it) / len(test_set.initial_list)), end="\r", flush=True)

    print("\n[Done]")
    test_summary = ultra.utils.merge_Summary(summary_list, batch_size_list)
    print("  eval: %s" % (
        ' '.join(['%s:%.3f' % (key, value) for key, value in test_summary.items()])
    ))

    # get rerank indexes with new scores
    rerank_lists = []
    for i in range(len(rerank_scores)):
        scores = rerank_scores[i]
        rerank_lists.append(sorted(range(len(scores)), key=lambda k: scores[k], reverse=True))
    print(rerank_scores)
    print(len(rerank_lists))
    ofile = open('ULTRE_online/output/FUSION_PDGD.txt', 'w')
    qdfile = open('ULTRE_online/test/test.init_list', 'r')
    query = []
    for line in qdfile:
        if (line != ''):
            q = int(line.strip().split(':')[0])
            query.append(q)
    qdfile.close()
    for i in range(300):
        ofile.write(str(query[i]) + ':')
        for j in range(10):
            ofile.write(str(query[i] * 10 + rerank_lists[i][j]))
            if j < 9:
                ofile.write(' ')
        if i < 299:
            ofile.write('\n')
    ofile.close()
    print(rerank_lists)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    ultra.utils.output_ranklist(test_set, rerank_scores, args.output_dir, args.test_data_prefix)

    return


def main(_):
    exp_settings = json.load(open(args.setting_file))
    if args.test_only:
        test(exp_settings)
    else:
        train(exp_settings)


if __name__ == "__main__":
    argv = sys.argv
    main(argv)
