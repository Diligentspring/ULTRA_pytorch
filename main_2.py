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
import copy
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import ultra
import numpy as np
import random

from ultra.learning_algorithm.dla_dcm import DLA_DCM
from ultra.learning_algorithm.dla_dcmq import DLA_DCMq
from ultra.learning_algorithm.dla_dcma import DLA_DCMa
from ultra.learning_algorithm.dla_ubm import DLA_UBM
from ultra.learning_algorithm.dla_pbm import DLA_PBM
from ultra.learning_algorithm.dla_pbm_em import DLA_PBM_EM
from ultra.learning_algorithm.dla_mcm_em import DLA_MCM_EM
from ultra.learning_algorithm.dla_mcma import DLA_MCMa
from ultra.learning_algorithm.prs_rank_modify import PRSrank_modify

# rank list size should be read from data
parser = argparse.ArgumentParser(description='Pipeline commandline argument')
parser.add_argument("--data_dir", type=str, default="./tests/data/", help="The directory of the experimental dataset.")
parser.add_argument("--train_data_prefix", type=str, default="train", help="The name prefix of the training data "
                                                                         "in data_dir.")
parser.add_argument("--valid_data_prefix", type=str, default="valid", help="The name prefix of the validation data in "
                                                                         "data_dir.")
parser.add_argument("--training_valid_data_prefix", type=str, default="training_valid", help="The name prefix of the training-validation data in "
                                                                         "data_dir.")
parser.add_argument("--test_data_prefix", type=str, default="test", help="The name prefix of the test data in data_dir.")
parser.add_argument("--model_dir", type=str, default="./tests/tmp_model/", help="The directory for model and "
                                                                              "intermediate outputs.")
parser.add_argument("--output_dir", type=str, default="./tests/tmp_output/", help="The directory to output results.")

parser.add_argument("--click_model_dir", type=str, default=None, help="The directory that contains labels produced by the click model")
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
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        # os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    setup_seed(1)

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
    training_valid_set = ultra.utils.read_data(args.data_dir, args.training_valid_data_prefix, args.click_model_dir, args.max_list_cutoff)
    ultra.utils.find_class(exp_settings['train_input_feed']).preprocess_data(training_valid_set,
                                                                             exp_settings['train_input_hparams'],
                                                                             exp_settings)

    print("Train Rank list size %d" % train_set.rank_list_size)
    print("Valid Rank list size %d" % valid_set.rank_list_size)
    exp_settings['max_candidate_num'] = max(train_set.rank_list_size, valid_set.rank_list_size)
    test_set = None
    if args.test_while_train:
        test_set = ultra.utils.read_data(args.data_dir, args.test_data_prefix, args.max_list_cutoff)
        ultra.utils.find_class(exp_settings['train_input_feed']).preprocess_data(test_set,
                                                                                 exp_settings['train_input_hparams'],
                                                                                 exp_settings)
        print("Test Rank list size %d" % test_set.rank_list_size)
        exp_settings['max_candidate_num'] = max(test_set.rank_list_size, exp_settings['max_candidate_num'])
        test_set.pad(exp_settings['max_candidate_num'])

    if 'selection_bias_cutoff' not in exp_settings:  # check if there is a limit on the number of items per training query.
        exp_settings['selection_bias_cutoff'] = args.selection_bias_cutoff if args.selection_bias_cutoff > 0 else \
            exp_settings['max_candidate_num']

    exp_settings['selection_bias_cutoff'] = min(exp_settings['selection_bias_cutoff'],
                                                exp_settings['max_candidate_num'])
    print('Users can only see the top %d documents for each query in training.' % exp_settings['selection_bias_cutoff'])

    # Pad data
    #train_set.pad(exp_settings['max_candidate_num'])
    #valid_set.pad(exp_settings['max_candidate_num'])

    # Create model based on the input layer.

    exp_settings['ln'] = args.ln
    exp_settings['train_data_prefix'] = args.train_data_prefix
    exp_settings['model_dir'] = args.model_dir

    print("Creating model...")
    # model = create_model(exp_settings, train_set)
    # model = DLA_DCM(train_set, exp_settings)
    model = PRSrank_modify(train_set, exp_settings)
    # model.print_info()

    # Create data feed
    train_input_feed = ultra.utils.find_class(exp_settings['train_input_feed'])(model, args.batch_size,
                                                                                exp_settings['train_input_hparams'])
    valid_input_feed = ultra.utils.find_class(exp_settings['valid_input_feed'])(model, args.batch_size,
                                                                                exp_settings['valid_input_hparams'])
    training_valid_input_feed = ultra.utils.find_class(exp_settings['valid_input_feed'])(model, args.batch_size,
                                                                                exp_settings['valid_input_hparams'])
    test_input_feed = None
    if args.test_while_train:
        test_input_feed = ultra.utils.find_class(exp_settings['test_input_feed'])(model, args.batch_size,
                                                                                  exp_settings[
                                                                                      'test_input_hparams'])

    # Create tensorboard summarizations.
    train_writer = torch.utils.tensorboard.SummaryWriter(log_dir=args.model_dir + '/train_log')
    valid_writer = torch.utils.tensorboard.SummaryWriter(log_dir=args.model_dir + '/valid_log')
    training_valid_writer = torch.utils.tensorboard.SummaryWriter(log_dir=args.model_dir + '/training_valid_log')
    test_writer = None
    if args.test_while_train:
        test_writer = torch.utils.tensorboard.SummaryWriter(log_dir=args.model_dir + '/test_log')

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    best_perf = None
    best_step = None
    training_valid_best_perf = None
    training_valid_best_step = None
    best_loss = None
    loss_best_step = None
    print("max_train_iter: ", args.max_train_iteration)
    while True:
        # Get a batch and make a step.
        start_time= time.time()
        input_feed, info_map = train_input_feed.get_batch(train_set, check_validation=True, data_format=args.data_format)

        # print(input_feed)
        # print(info_map)
        #break

        step_loss, _, summary = model.train(input_feed)
        #break
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

            parafile = open(args.model_dir+'/para.txt', 'a')
            parafile.write('current_step:'+str(current_step)+'\n')
            for i in range(len(model.propensity_parameter)):
                parafile.write(str(model.propensity_parameter[i].data)+' ')
                if i%10 ==9:
                    parafile.write('\n')
            parafile.close()

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
                    #summary_list.append(summary)
                    # deep copy the summary dict
                    summary_list.append(copy.deepcopy(summary))
                    batch_size_list.append(len(info_map['input_list']))
                    it += batch_size_list[-1]
                    count_batch += 1.0
                return ultra.utils.merge_Summary(summary_list, batch_size_list)
                # return summary_list

            valid_summary = validate_model(valid_set, valid_input_feed)
            valid_writer.add_scalars('Validation_Summary', valid_summary, model.global_step)
            for key,value in valid_summary.items():
                # print(key, value)
                print("%s %.4f" % (key, value))

            def training_validate_model(data_set, data_input_feed):
                it = 0
                count_batch = 0.0
                summary_list = []
                batch_size_list = []
                while it < len(data_set.initial_list):
                    input_feed, info_map = data_input_feed.get_next_batch(
                        it, data_set, check_validation=False, data_format=args.data_format)
                    _, _, summary = model.validation(input_feed)
                    # summary_list.append(summary)
                    # deep copy the summary dict
                    summary_list.append(copy.deepcopy(summary))
                    batch_size_list.append(len(info_map['input_list']))
                    it += batch_size_list[-1]
                    count_batch += 1.0
                return ultra.utils.merge_Summary(summary_list, batch_size_list)

            training_valid_summary = training_validate_model(training_valid_set, training_valid_input_feed)
            training_valid_writer.add_scalars('Training_Validation_Summary', training_valid_summary, model.global_step)
            for key, value in training_valid_summary.items():
                # print(key, value)
                print("%s %.4f" % (key, value))


            if args.test_while_train:
                test_summary = validate_model(test_set, test_input_feed)
                test_writer.add_scalars('Validation Summary while training', valid_summary, model.global_step)
                for key, value in test_summary.items:
                    print(key, value)

            # if current_step % (5 * args.steps_per_checkpoint) == 0:
            #     if best_loss == None or best_loss > loss:
            #         checkpoint_path = os.path.join(args.model_dir,
            #                                        "%s.ckpt" % str(exp_settings['learning_algorithm']) + str(
            #                                            model.global_step))
            #         torch.save(model.model.state_dict(), checkpoint_path)
            #
            #         best_loss = loss
            #         loss_best_step = model.global_step
            #     print('best loss:%.4f,step %d' % (best_loss, loss_best_step))

            # Save checkpoint if the objective metric on the validation set is better
            if "objective_metric" in exp_settings:
                for key,value in valid_summary.items():
                    if key == exp_settings["objective_metric"]:
                        if current_step >= args.start_saving_iteration:
                            if best_perf == None or best_perf < value:
                                checkpoint_path = os.path.join(args.model_dir,
                                                               "%s.ckpt" % str(exp_settings['learning_algorithm'])+str(model.global_step))
                                torch.save(model.model.state_dict(), checkpoint_path)
                                best_perf = value
                                best_step = model.global_step
                                print('Save model, valid %s:%.4f,step %d' % (key, best_perf, best_step))
                                break
                            print('best valid %s:%.4f,step %d' % (key, best_perf, best_step))

            # Save checkpoint if the objective metric on the training_validation set is better
            if "objective_metric" in exp_settings:
                for key, value in training_valid_summary.items():
                    if key == exp_settings["objective_metric"]:
                        if current_step >= args.start_saving_iteration:
                            if training_valid_best_perf == None or training_valid_best_perf < value:
                                checkpoint_path = os.path.join(args.model_dir,
                                                               "%s.ckpt" % str(exp_settings[
                                                                                   'learning_algorithm']) + str(
                                                                   model.global_step))
                                torch.save(model.model.state_dict(), checkpoint_path)
                                training_valid_best_perf = value
                                training_valid_best_step = model.global_step
                                print('Save model, training_valid %s:%.4f,step %d' % (key, training_valid_best_perf, training_valid_best_step))
                                break
                            print('best training_valid %s:%.4f,step %d' % (key, training_valid_best_perf, training_valid_best_step))

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

    #test_set.pad(exp_settings['max_candidate_num'])

    exp_settings['ln'] = args.ln
    # Create model and load parameters.
    # model = create_model(exp_settings, test_set)
    model = DLA_PBM(test_set, exp_settings)
    # model = PRSrank_modify(test_set, exp_settings)

    #checkpoint_path = os.path.join(args.model_dir+'_256_0.1', "%s.ckpt7900" % exp_settings['learning_algorithm'])
    checkpoint_path = args.model_dir
    ckpt = torch.load(checkpoint_path)
    print("Reading model parameters from %s" % checkpoint_path)
    model.model.load_state_dict(ckpt)
    model.model.eval()

    # Create input feed
    test_input_feed = ultra.utils.find_class(exp_settings['test_input_feed'])(model, args.batch_size,
                                                                              exp_settings['test_input_hparams'])

    # test_writer = SummaryWriter(log_dir=args.model_dir + '/test_log')

    rerank_scores = []
    summary_list = []
    # Start testing.

    it = 0
    count_batch = 0.0
    batch_size_list = []
    while it < len(test_set.initial_list):
        input_feed, info_map = test_input_feed.get_next_batch(it, test_set, check_validation=False)
        _, output_logits, summary = model.validation(input_feed)
        #summary_list.append(summary)
        # deep copy the summary dict
        summary_list.append(copy.deepcopy(summary))
        batch_size_list.append(len(info_map['input_list']))
        for x in range(batch_size_list[-1]):
            rerank_scores.append(output_logits[x])
        it += batch_size_list[-1]
        count_batch += 1.0
        print("Testing {:.0%} finished".format(float(it) / len(test_set.initial_list)), end="\r", flush=True)

    print("\n[Done]")
    test_summary = ultra.utils.merge_Summary(summary_list, batch_size_list)
    print("  eval: %s" % (
        ' '.join(['%s:%.4f' % (key, value) for key, value in test_summary.items()])
    ))

    # get rerank indexes with new scores
    rerank_lists = []
    for i in range(len(rerank_scores)):
        scores = rerank_scores[i]
        rerank_lists.append(sorted(range(len(scores)), key=lambda k: scores[k], reverse=True))
    # print(rerank_scores)
    # print(rerank_lists)

    # ofile = open('ULTRE_IPS/output/PBM_MCM.txt', 'w')
    # qdfile = open('ULTRE_rel/test/test.init_list', 'r')
    # query = []
    # for line in qdfile:
    #     if (line != ''):
    #         q = int(line.strip().split(':')[0])
    #         query.append(q)
    # qdfile.close()
    # for i in range(300):
    #     ofile.write(str(query[i])+':')
    #     for j in range(10):
    #         ofile.write(str(query[i]*10+rerank_lists[i][j]))
    #         if j<9:
    #             ofile.write(' ')
    #     if i<299:
    #         ofile.write('\n')
    # ofile.close()
    # print(rerank_lists)

    qdfile = open('ULTRE_train_pl_0_click/test/test.init_list', 'r')
    query = []
    for line in qdfile:
        if (line != ''):
            q = int(line.strip().split(':')[0])
            query.append(q)
    qdfile.close()

    def DCG(label_list):
        dcgsum = 0
        for i in range(len(label_list)):
            dcg = (2 ** label_list[i] - 1) / np.log2(i + 2)
            dcgsum += dcg
        return dcgsum

    # ndcg 计算
    def NDCG(label_list, top_n):
        # 没有设定topn
        if top_n == None:
            dcg = DCG(label_list)
            ideal_list = sorted(label_list, reverse=True)
            ideal_dcg = DCG(ideal_list)
            if ideal_dcg == 0:
                return 0
            return dcg / ideal_dcg
        # 设定top n
        else:
            dcg = DCG(label_list[0:top_n])
            ideal_list = sorted(label_list, reverse=True)
            ideal_dcg = DCG(ideal_list[0:top_n])
            if ideal_dcg == 0:
                return 0
            return dcg / ideal_dcg

    labelfile = open('ULTRE_train_pl_0_click/test/test.labels', 'r')
    label_dict = {}
    for line in labelfile:
        if line != '':
            line_split = line.strip().split(':')
            qid = int(line_split[0])
            labels = line_split[1].strip().split(' ')
            for i in range(10):
                label_dict[qid * 10 + i] = int(labels[i])
    labelfile.close()

    # classify_dir = args.model_dir.strip('./data/work/niuzechun/').split('/')
    classify_dir = args.model_dir.strip('./').split('/')
    classify_file = open('classify_test/{}/{}/{}'.format(classify_dir[0],classify_dir[1],classify_dir[2]), 'w')

    ndcg_5 = 0.0
    for i in range(300):
        labels = []
        for j in range(10):
            labels.append(label_dict[query[i]*10+rerank_lists[i][j]])
            classify_file.write(str(rerank_scores[i][j].item())+'\n')
        ndcg_5 = NDCG(labels, 5) + ndcg_5
        # print(labels)
        # print(NDCG(labels, 5))

    classify_file.close()

    ave_ndcg_5 = ndcg_5 / 300
    print('NDCG@5: ' + str(ave_ndcg_5))
    #
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)
    # ultra.utils.output_ranklist(test_set, rerank_scores, args.output_dir, args.test_data_prefix)

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
