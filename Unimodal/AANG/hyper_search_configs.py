from copy import deepcopy


HYPER_CONFIG_PARTIAL_BIG = {
		'auxlr': [0.1, 1.0],
		'soptlr': [0.01, 0.1],
		'classflr': [1e-4, 1e-3],
		'wfrac': [0.06],
		'nconf_samp': [3, 6],
		'primbsz': [128],
		'auxbsz': [256]
}
HYPER_CONFIG_PARTIAL_BIG_1 = deepcopy(HYPER_CONFIG_PARTIAL_BIG)
HYPER_CONFIG_PARTIAL_BIG_1['nconf_samp'] = [3]


HYPER_CONFIG_HYPERPARTISAN = deepcopy(HYPER_CONFIG_PARTIAL_BIG)
HYPER_CONFIG_HYPERPARTISAN['primbsz'] = [64]
HYPER_CONFIG_HYPERPARTISAN['auxbsz'] = [128]

HYPER_CONFIG_PARTIAL_BIG_1 = {
		'auxlr': [1.0, 0.1],
		'soptlr': [0.1],
		'classflr': [1e-3, 1e-4],
		'wfrac': [0.06],
		'nconf_samp': [6],
		'primbsz': [128],
		'auxbsz': [512, 1024]
}


HYPER_CONFIG_HYPERPARTISAN_1 = deepcopy(HYPER_CONFIG_PARTIAL_BIG_1)
HYPER_CONFIG_HYPERPARTISAN_1['primbsz'] = [64]
HYPER_CONFIG_HYPERPARTISAN_1['auxbsz'] = [128]


HYPER_CONFIG_PARTIAL_MULTI = {
		'auxlr': [0],
		'soptlr': [0],
		'classflr': [1e-4, 1e-3],
		'wfrac': [0.06],
		'nconf_samp': [3, 6],
		'primbsz': [128],
		'auxbsz': [256]
}


CT_BEST_OURS = {
	'auxlr': [0.1],
	'soptlr': [1e-1],
	'classflr': [1e-3],
	'wfrac': [0.06],
	'nconf_samp': [6],
	'primbsz': [128],
	'auxbsz': [256]
}

CT_BEST_JOINT = {
	'auxlr': [1.0],
	'soptlr': [0.1],
	'classflr': [1e-4],
	'wfrac': [0.06],
	'nconf_samp': [3],
	'primbsz': [128],
	'auxbsz': [256]
}

CT_BEST_GPT = {
	'auxlr': [0.1],
	'soptlr': [1.0],
	'classflr': [1e-3],
	'wfrac': [0.06],
	'nconf_samp': [1],
	'primbsz': [128],
	'auxbsz': [256]
}

CT_BEST_XLNET = {
	'auxlr': [0.1],
	'soptlr': [1e-1],
	'classflr': [1e-4],
	'wfrac': [0.06],
	'nconf_samp': [1],
	'primbsz': [128],
	'auxbsz': [256]
}

CT_BEST_TAPT = {
	'auxlr': [1.0],
	'soptlr': [0.1],
	'classflr': [1e-4],
	'wfrac': [0.06],
	'nconf_samp': [1],
	'primbsz': [16],#[128],
	'auxbsz': [32]#[256]
}


HYPER_CONFIG_PARTIAL_ONETASK = {
		'auxlr': [0.1],
		'soptlr': [0.01, 0.1, 1.0],
		'wfrac': [0.06],
		'primbsz': [128],
		'auxbsz': [256]
}
# deepcopy(HYPER_CONFIG_PARTIAL_BIG)
HYPER_CONFIG_PARTIAL_ONETASK['nconf_samp'] = [1]
HYPER_CONFIG_PARTIAL_ONETASK['classflr'] = [1e-3, 1e-4, 5e-5]

HYPER_CONFIG_HYPERPARTISAN_ONETASK = deepcopy(HYPER_CONFIG_PARTIAL_ONETASK)
HYPER_CONFIG_HYPERPARTISAN_ONETASK['primbsz'] = [64]
HYPER_CONFIG_HYPERPARTISAN_ONETASK['auxbsz'] = [128]


HYPER_CONFIG_FULL = {
		'auxlr': [0.1, 5e-1, 1.0],
		'soptlr': [1e-1],
		'classflr': [1e-3, 1e-4, 3e-3, 5e-3, 1e-2],
		'nconf_samp': [3, 6],
		'primbsz': [128],
		'auxbsz': [256]
}


HYPER_CONFIG_TEST = {
		'auxlr': [0.1],
		'soptlr': [1e-1],
		'classflr': [3e-3],
		'wfrac': [0.06],
		'nconf_samp': [1],
		'primbsz': [128],
		'auxbsz': [256]
}

CONFIG_NAMES = [
	"full",  "partial",
	"partial_big", "partial_onetask",
	"partial_hyperpartisan", "partial_big_1",
	"partial_big_multi", 'partial_hyperpartisan_onetask',
	'partial_hyperpartisan_1', 'all_data',
	'ct_best_ours', 'ct_best_gpt', 'ct_best_joint', 
	'ct_best_xlnet', 'ct_best_tapt'
]

def get_hyper_config(config_name):
	if config_name == 'full':
		return HYPER_CONFIG_FULL
	elif config_name == 'partial':
		return HYPER_CONFIG_PARTIAL
	elif config_name == 'partial_big':
		return HYPER_CONFIG_PARTIAL_BIG
	elif config_name == 'partial_big_1':
		return HYPER_CONFIG_PARTIAL_BIG_1
	elif config_name == 'partial_hyperpartisan_1':
		return HYPER_CONFIG_HYPERPARTISAN_1
	elif config_name == 'partial_onetask':
		return HYPER_CONFIG_PARTIAL_ONETASK
	elif config_name == 'partial_hyperpartisan':
		return HYPER_CONFIG_HYPERPARTISAN
	elif config_name == 'partial_hyperpartisan_onetask':
		return HYPER_CONFIG_HYPERPARTISAN_ONETASK
	elif config_name == 'partial_big_multi':
		return HYPER_CONFIG_PARTIAL_MULTI
	elif config_name == 'ct_best_ours':
		return CT_BEST_OURS
	elif config_name == 'ct_best_gpt':
		return CT_BEST_GPT
	elif config_name == 'ct_best_joint':
		return CT_BEST_JOINT
	elif config_name == 'ct_best_xlnet':
		return CT_BEST_XLNET
	elif config_name == 'ct_best_tapt':
		return CT_BEST_TAPT



# Modified appropriately
CITATION_INTENT = {
	'primtaskid': 'citation_intent',
	'trainfile':  'datasets/citation_intent/train.jsonl',
	'devfile':    'datasets/citation_intent/dev.jsonl',
	'testfile':   'datasets/citation_intent/test.jsonl',
	'taskdata':   'datasets/citation_intent/train.txt',
	'domaindata': 'datasets/citation_intent/domain.10xTAPT.txt',
	'metric':     'f1',
}


SCIIE = {
	'primtaskid': 'sciie',
	'trainfile':  '/home/ldery/internship/dsp/datasets/sciie/train.jsonl',
	'devfile':    '/home/ldery/internship/dsp/datasets/sciie/dev.jsonl',
	'testfile':   '/home/ldery/internship/dsp/datasets/sciie/test.jsonl',
	'taskdata':   '/home/ldery/internship/dsp/datasets/sciie/train.txt',
	'domaindata': '/home/ldery/internship/dsp/datasets/sciie/domain.10xTAPT.txt',
	'metric':     'f1',
}

CHEMPROT = {
	'primtaskid': 'chemprot',
	'trainfile':  '/work/sakter/AANG/temp_datasets/chemprot/train.jsonl',
	'devfile':    '/work/sakter/AANG/temp_datasets/chemprot/dev.jsonl',
	'testfile':   '/work/sakter/AANG/temp_datasets/chemprot/test.jsonl',
	'taskdata':   '/work/sakter/AANG/temp_datasets/chemprot/train.txt',
	'domaindata': '/work/sakter/AANG/temp_datasets/chemprot/domain.10xTAPT.txt',
	'metric':     'accuracy',
}

HYPERPARTISAN = {
	'primtaskid': 'hyperpartisan',
	'trainfile':  '/home/ldery/internship/dsp/datasets/hyperpartisan/train.jsonl',
	'devfile':    '/home/ldery/internship/dsp/datasets/hyperpartisan/dev.jsonl',
	'testfile':   '/home/ldery/internship/dsp/datasets/hyperpartisan/test.jsonl',
	'taskdata':   '/home/ldery/internship/dsp/datasets/hyperpartisan/train.txt',
	'domaindata': '/home/ldery/internship/dsp/datasets/hyperpartisan/domain.10xTAPT.txt',
	'metric':     'f1',
}

RCT = {
	'primtaskid': 'rct',
	'trainfile':  '/home/ldery/internship/dsp/datasets/rct/train.jsonl',
	'devfile':    '/home/ldery/internship/dsp/datasets/rct/dev.jsonl',
	'testfile':   '/home/ldery/internship/dsp/datasets/rct/test.jsonl',
	'taskdata':   '/home/ldery/internship/dsp/datasets/rct/train.txt',
	'domaindata': '/home/ldery/internship/dsp/datasets/rct/domain.10xTAPT.txt',
	'metric':     'accuracy',
}

SemEval2016Task6 = {
	'primtaskid': 'SemEval2016Task6',
	'trainfile':  '/home/ldery/projects/ml-stance-detection/datasets/SemEval2016Task6/train.jsonl',
	'devfile':    '/home/ldery/projects/ml-stance-detection/datasets/SemEval2016Task6/dev.jsonl',
	'testfile':   '/home/ldery/projects/ml-stance-detection/datasets/SemEval2016Task6/test.jsonl',
	'taskdata':   '/home/ldery/projects/ml-stance-detection/datasets/SemEval2016Task6/train.txt',
	'domaindata': '/home/ldery/projects/ml-stance-detection/datasets/SemEval2016Task6/train.txt', # Todo [ldery] - change this up
	'metric':     'accuracy',
}

PERSPECTRUM = {
	'primtaskid': 'PERSPECTRUM',
	'trainfile':  '/home/ldery/projects/ml-stance-detection/datasets/PERSPECTRUM/perspectrum_train.jsonl',
	'devfile':    '/home/ldery/projects/ml-stance-detection/datasets/PERSPECTRUM/perspectrum_dev.jsonl',
	'testfile':   '/home/ldery/projects/ml-stance-detection/datasets/PERSPECTRUM/perspectrum_test.jsonl',
	'taskdata':   '/home/ldery/projects/ml-stance-detection/datasets/PERSPECTRUM/perspectrum_train.txt',
	'domaindata': '/home/ldery/projects/ml-stance-detection/datasets/PERSPECTRUM/perspectrum_train.txt', # Todo [ldery] - change this up
	'metric':     'accuracy',
}
