import argparse
import torch
import os
from hyper_search_configs import *
import math
import threading
import subprocess
import pickle as pkl
import time
import numpy as np
from itertools import product

# Code burrowed from : https://stackoverflow.com/questions/984941/python-subprocess-popen-from-a-thread
class MyThreadClass(threading.Thread):
	def __init__(self, command_list):
		self.stdout = None
		self.stderr = None
		self.command_list = command_list
		threading.Thread.__init__(self)

	def run(self):
		for cmd in self.command_list:
			os.system(cmd)

'''
For Paper Results [Larger Hyper-param Space]
python hyperparam_search.py -task citation_intent -base-spconfig jointbasic           -patience 20 -grad-accum-steps 4 -exp-name JOINT-BASIC -gpu-list "[0]" -hyperconfig partial_big_1   -runthreads -pure-transform
python hyperparam_search.py -task citation_intent -base-spconfig xlnet                -patience 20 -grad-accum-steps 4 -exp-name XLNET       -gpu-list "[1]" -hyperconfig partial_onetask -runthreads -pure-transform
python hyperparam_search.py -task citation_intent -base-spconfig tapt                 -patience 20 -grad-accum-steps 4 -exp-name TAPT        -gpu-list "[2]" -hyperconfig partial_onetask -runthreads -pure-transform
python hyperparam_search.py -task citation_intent -base-spconfig gpt                  -patience 20 -grad-accum-steps 4 -exp-name GPT         -gpu-list "[2]" -hyperconfig partial_onetask -runthreads -pure-transform
python hyperparam_search.py -task citation_intent -base-spconfig citation.supervised  -patience 20 -grad-accum-steps 4 -exp-name SUPERVISED  -gpu-list "[3]" -hyperconfig partial_big     -runthreads -pure-transform


Best HyperConfig
python hyperparam_search.py -task citation_intent -base-spconfig jointbasic           -patience 20 -grad-accum-steps 4 -exp-name JOINT-BASIC -gpu-list "[0]" -hyperconfig ct_best_joint -runthreads -pure-transform
python hyperparam_search.py -task citation_intent -base-spconfig xlnet                -patience 20 -grad-accum-steps 4 -exp-name XLNET       -gpu-list "[1]" -hyperconfig ct_best_xlnet -runthreads -pure-transform
python hyperparam_search.py -task citation_intent -base-spconfig tapt                 -patience 20 -grad-accum-steps 4 -exp-name TAPT        -gpu-list "[2]" -hyperconfig ct_best_tapt  -runthreads -pure-transform
python hyperparam_search.py -task citation_intent -base-spconfig citation.supervised  -patience 20 -grad-accum-steps 4 -exp-name SUPERVISED  -gpu-list "[3]" -hyperconfig ct_best_ours  -runthreads -pure-transform
python hyperparam_search.py -task citation_intent -base-spconfig gpt                  -patience 20 -grad-accum-steps 4 -exp-name GPT         -gpu-list "[2]" -hyperconfig ct_best_gpt   -runthreads -pure-transform
'''

def add_hyperparam_options(parser):
	parser.add_argument('-task', type=str, default='sciie')
	parser.add_argument('-base-spconfig', type=str, default='vbasic1')
	parser.add_argument('-warmup-frac', type=float, default=0.06)
	parser.add_argument('-step-meta-every', type=int, default=3)
	parser.add_argument('-patience', type=int, default=10)
	parser.add_argument('-classfdp', type=float, default=0.3)
	parser.add_argument('-iters', type=int, default=150)
	parser.add_argument('-dev-ft-iters', type=int, default=10)
	parser.add_argument('-devbsz', type=int, default=32)
	parser.add_argument('-devlr', type=float, default=1e-2)
	parser.add_argument('-tokentform-temp', type=float, default=0.5)
	parser.add_argument('-base-wd', type=float, default=0.01)
	parser.add_argument('-dev-wd', type=float, default=0.1)
	parser.add_argument('-classf-wd', type=float, default=0.1)
	parser.add_argument('-lr', type=float, default=1e-4)
	parser.add_argument('-grad-accum-steps', type=int, default=2)
	parser.add_argument('-num-seeds', type=int, default=3)
	parser.add_argument('-output-dir', type=str, default='autoaux_outputs')
	parser.add_argument('-exp-name', type=str, default='hyperparamSearch')
	parser.add_argument('-logdir', type=str, default='HyperParamLogs')
	parser.add_argument('-runthreads', action='store_true')
	parser.add_argument('-gpu-list', type=str)
	parser.add_argument('-hyperconfig', type=str, default="full", choices=CONFIG_NAMES)
	parser.add_argument('-do-retrain', action='store_true')
	parser.add_argument('-pure-transform', action='store_true')


def get_task_info(args):
	if args.task == 'citation_intent':
		return CITATION_INTENT
	elif args.task == 'sciie':
		return SCIIE
	elif args.task == 'chemprot':
		return CHEMPROT
	elif args.task == 'hyperpartisan':
		return HYPERPARTISAN
	elif args.task == 'SemEval2016Task6':
		return SemEval2016Task6
	elif args.task == 'PERSPECTRUM':
		return PERSPECTRUM

def get_all_hyperconfigs(config_dict):
	all_hypers = []
	all_configs = list(product(*list(config_dict.values())))
	all_keys = list(config_dict.keys())
	all_hyperconfigs = [{k: v for k, v in zip(all_keys, config)} for config in all_configs]
	return all_hyperconfigs

def has_been_run(fldr):
	has_saved_model = os.path.exists(os.path.join(fldr, 'modelWAuxTasks.pth'))
	has_saved_searchOpts =  os.path.exists(os.path.join(fldr, 'searchOpts.pth'))
	if has_saved_model and has_saved_searchOpts:
		print('SKIPPING. ALREADY RUN : ', os.path.basename(fldr))
		return True
	return False

# Todo [ldery] - can clean this up a bit better !!!
def get_base_runstring(args, gpuid, config, task_info):
	config['wfrac'] = args.warmup_frac
	hyper_id = ".".join(["{}={}".format(k, v) for k, v in config.items()])
	pergpubsz = int(config['auxbsz'] / args.grad_accum_steps)
	primiterbsz = int(config['primbsz'] / args.grad_accum_steps)

	logdir = "{}/MassLaunch/{}/{}/{}".format(args.logdir, args.exp_name, args.task, hyper_id)
	os.makedirs(logdir, exist_ok=True)
	this_output_dir = "{}/{}".format(args.output_dir, args.exp_name)
	pure_transform_str = '-pure-transform' if args.pure_transform else ''

	run_commands, outdirs = [], []
	for seed in range(args.num_seeds):
		logfile = "{}/seed={}.txt".format(logdir, seed)
		outputdir ="{}/{}/{}/seed={}".format(this_output_dir, args.task, hyper_id, seed)

		os.makedirs(outputdir, exist_ok=True)
		run_command = None
		if not has_been_run(outputdir):
			warmup_frac = args.warmup_frac if 'wfrac' not in task_info else task_info['wfrac']
			run_command = "CUDA_VISIBLE_DEVICES={} python -u -m scripts.autoaux --prim-task-id {} --train_data_file {} --dev_data_file {} --test_data_file {} --output_dir {} --model_type roberta-base --model_name_or_path roberta-base  --tokenizer_name roberta-base --per_gpu_train_batch_size {}  --gradient_accumulation_steps {} --do_train --learning_rate {} --block_size 512 --logging_steps 10000 --classf_lr {} --classf_patience {} --num_train_epochs {} --classifier_dropout {} --overwrite_output_dir --classf_iter_batchsz  {} --classf_ft_lr 1e-6 --classf_max_seq_len 512 --seed {}  --classf_dev_wd {} --classf_dev_lr {} -searchspace-config {} -task-data {} -in-domain-data {} -num-config-samples {} --dev_batch_sz {} --eval_every 30 -prim-aux-lr {} -auxiliaries-lr {} --classf_warmup_frac {} --classf_wd {} --base_wd {} --dev_fit_iters {} -step-meta-every {} -token_temp {}  --classf-metric {} {} &> {}".format(gpuid, args.task, task_info['trainfile'], task_info['devfile'], task_info['testfile'], outputdir, pergpubsz, args.grad_accum_steps, args.lr, config['classflr'], args.patience, args.iters, args.classfdp, primiterbsz, seed, args.dev_wd, args.devlr, args.base_spconfig, task_info['taskdata'], task_info['domaindata'], config['nconf_samp'], args.devbsz, config['soptlr'], config['auxlr'], warmup_frac, args.classf_wd, args.base_wd, args.dev_ft_iters, args.step_meta_every, args.tokentform_temp, task_info['metric'], pure_transform_str, logfile)

		outputdir_retrain = None
		if args.do_retrain:
			searchOpts_path = os.path.join(outputdir, 'searchOpts.pth')
			logfile = "{}/retrain.seed={}.txt".format(logdir, seed)
			outputdir_retrain ="{}/{}/{}/retrain.seed={}".format(this_output_dir, args.task, hyper_id, seed)
			os.makedirs(outputdir_retrain, exist_ok=True)
			if not has_been_run(outputdir_retrain):
				warmup_frac = args.warmup_frac if 'wfrac' not in task_info else task_info['wfrac']
				retrain_run_command = "CUDA_VISIBLE_DEVICES={} python -u -m scripts.autoaux --prim-task-id {} --train_data_file {} --dev_data_file {} --test_data_file {} --output_dir {} --model_type roberta-base --model_name_or_path roberta-base  --tokenizer_name roberta-base --per_gpu_train_batch_size {}  --gradient_accumulation_steps {} --do_train --learning_rate {} --block_size 512 --logging_steps 10000 --classf_lr {} --classf_patience {} --num_train_epochs {} --classifier_dropout {} --overwrite_output_dir --classf_iter_batchsz  {} --classf_ft_lr 1e-6 --classf_max_seq_len 512 --seed {}  --classf_dev_wd {} --classf_dev_lr {} -searchspace-config {} -task-data {} -in-domain-data {} -num-config-samples {} --dev_batch_sz {} --eval_every 30 -prim-aux-lr {} -auxiliaries-lr {} --classf_warmup_frac {} --classf_wd {} --base_wd {} --dev_fit_iters {} -step-meta-every {} -token_temp {} --share-output-heads --classf-metric {} -warmstart-path {} {} &> {}".format(gpuid, args.task, task_info['trainfile'], task_info['devfile'], task_info['testfile'], outputdir_retrain, pergpubsz, args.grad_accum_steps, args.lr, config['classflr'], args.patience, args.iters, args.classfdp, primiterbsz, seed, args.dev_wd, args.devlr, args.base_spconfig, task_info['taskdata'], task_info['domaindata'], config['nconf_samp'], args.devbsz, config['soptlr'], config['auxlr'], warmup_frac, args.classf_wd, args.base_wd, args.dev_ft_iters, args.step_meta_every, args.tokentform_temp, task_info['metric'], searchOpts_path, pure_transform_str, logfile)
				run_command = "{}\n{}".format(run_command, retrain_run_command) if run_command else retrain_run_command


		if run_command:
			run_commands.append(run_command)
		outdirs.append((outputdir, outputdir_retrain))
	
	return hyper_id, run_commands, outdirs


import pdb
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	add_hyperparam_options(parser)
	args = parser.parse_args()
	task_info = get_task_info(args)
	this_hyper_config = get_hyper_config(args.hyperconfig)
	all_hyperconfigs = get_all_hyperconfigs(this_hyper_config)

	gpu_list = eval(str(args.gpu_list))
	num_gpus = len(gpu_list)

	all_threads = []
	all_conf_results = {} 
	print('This is run threads bool : ', args.runthreads)
	print('Generating configs and sending them to threads')
	chunks = np.array_split(all_hyperconfigs, num_gpus)
	print('These are the chunks : ', [len(x) for x in chunks])

	for gpu_num, gpuid in enumerate(gpu_list):
		hyperconfigs = chunks[gpu_num]
		this_commands = []

		for config_ in hyperconfigs:
			hyper_id, run_commands, outdirs = get_base_runstring(args, gpuid, config_, task_info)
			this_commands.extend(run_commands)
			all_conf_results[hyper_id] = outdirs, config_
		if args.runthreads:
			this_thread = MyThreadClass(this_commands)
			all_threads.append(this_thread)
			this_thread.start()

	if args.runthreads:
		for thread in all_threads:
			thread.join()

	# We can now write the results to a csv
	print('All threads are done. Gather the configs and generate csv of results')
	timestr = time.strftime("%Y%m%d-%H%M%S")
	fname = "resultsSheets/{}_{}_{}.csv".format("{}{}".format(args.exp_name, '.retrain' if args.do_retrain else ''), args.task, timestr)
	with open(fname, 'w') as fhandle:
		headnames = list(this_hyper_config.keys())
		headnames.extend(['preft.f1.mean', 'preft.f1.std', 'postft.f1.mean', 'postft.f1.std'])
		headnames.extend(['preft.acc.mean', 'preft.acc.std', 'postft.acc.mean', 'postft.acc.std',])
		header = ",".join(headnames)
		fhandle.write("{}\n".format(header))
		# Now we just gather the results
		for hyper_id, (outdirs, config_) in all_conf_results.items():
			preft_f1, preft_acc, postft_f1, postft_acc = [], [], [], []
			for outdir in outdirs:
				if args.do_retrain:
					_, outdir = outdir
				else:
					outdir, _ = outdir
				try:
					handle = open('{}/ftmodel.bestperfs.pkl'.format(outdir), 'rb')
					info = pkl.load(handle)
					pre_ft, post_ft = info
					preft_f1.append(pre_ft['f1'])
					postft_f1.append(post_ft['f1'])
					preft_acc.append(pre_ft['accuracy'])
					postft_acc.append(post_ft['accuracy'])
				except:
					print('Could not load results. Had to skip : {}'.format(outdir))

			this_results = [config_[k] for k in list(this_hyper_config.keys())]
			this_results.extend([np.mean(preft_f1), np.std(preft_f1), np.mean(postft_f1), np.std(postft_f1)])
			this_results.extend([np.mean(preft_acc), np.std(preft_acc), np.mean(postft_acc), np.std(postft_acc)])
			this_entry = ",".join([str(x) for x in this_results])
			fhandle.write("{}\n".format(this_entry))

